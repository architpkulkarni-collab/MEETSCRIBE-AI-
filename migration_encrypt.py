"""
AirScribe — Migration Script: Encrypt Existing Notes
=======================================================
One-time migration to encrypt all existing plaintext notes
and generate semantic embeddings.

Usage:
    python migration_encrypt.py              # Run migration
    python migration_encrypt.py --verify     # Verify migration only
    python migration_encrypt.py --dry-run    # Preview without modifying

IMPORTANT:
    - BACK UP YOUR DATABASE BEFORE RUNNING THIS SCRIPT
    - This is a DESTRUCTIVE operation: plaintext fields are replaced
      with ciphertext. If you lose the encryption key, the data is
      PERMANENTLY UNREADABLE.
    - Run this ONCE after deploying Feature #6.
    - The script is idempotent: already-encrypted notes are skipped.

What it does:
    1. Connects to MongoDB using MONGO_URI from .env
    2. Iterates over all notes in the 'notes' collection
    3. For each unencrypted note:
       a. Encrypts 'text' and 'content' fields
       b. Generates a 384-dim semantic embedding
       c. Writes the encrypted document back
       d. Removes the plaintext fields
    4. Same for the 'trash' collection
    5. Verifies that migrated notes can be decrypted
    6. Prints progress and summary
"""

import os
import sys
import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from pymongo import MongoClient
from bson.objectid import ObjectId

from encryption import derive_key, encrypt, decrypt, is_encrypted
from note_embeddings import generate_embedding, generate_embeddings_batch


def _get_connection():
    """Validate environment and return (key, notes_col, trash_col)."""
    master_key = os.getenv("NOTE_ENCRYPTION_MASTER_KEY", "")
    salt = os.getenv("NOTE_ENCRYPTION_SALT", "")
    mongo_uri = os.getenv("MONGO_URI", "")
    db_name = os.getenv("DB_NAME", "airscribe")

    if not master_key or not salt:
        print("ERROR: NOTE_ENCRYPTION_MASTER_KEY and NOTE_ENCRYPTION_SALT must be set in .env")
        sys.exit(1)
    if not mongo_uri:
        print("ERROR: MONGO_URI must be set in .env")
        sys.exit(1)

    key = derive_key(master_key, salt)
    client = MongoClient(mongo_uri)
    db = client[db_name]
    return key, db["notes"], db["trash"], db_name


def verify(key, notes_col, trash_col):
    """
    Verify that all encrypted notes in the database can be decrypted.
    Does NOT modify any data.
    """
    print(f"\n{'='*55}")
    print(f"  AirScribe — Migration Verification")
    print(f"  Time: {datetime.utcnow().isoformat()}Z")
    print(f"{'='*55}\n")

    verified = 0
    failed = 0
    plaintext_found = 0

    for collection_name, col in [("notes", notes_col), ("trash", trash_col)]:
        docs = list(col.find())
        print(f"📋 Verifying {len(docs)} documents in '{collection_name}'...")

        for doc in docs:
            doc_id = str(doc["_id"])

            if not is_encrypted(doc):
                # Document still has plaintext — migration did not run or was partial
                if "text" in doc or "content" in doc:
                    plaintext_found += 1
                continue

            try:
                # Try decrypting text
                decrypted_text = decrypt(
                    doc["encrypted_text"],
                    doc["encryption_nonce_text"],
                    key,
                    doc_id
                )
                # Try decrypting content
                if doc.get("encrypted_content") and doc.get("encryption_nonce_content"):
                    decrypt(
                        doc["encrypted_content"],
                        doc["encryption_nonce_content"],
                        key,
                        doc_id
                    )
                # Verify embedding exists
                if not doc.get("embedding"):
                    print(f"  ⚠️  Note {doc_id}: encrypted but missing embedding")

                verified += 1

            except Exception as e:
                failed += 1
                print(f"  ❌ Note {doc_id}: decryption FAILED — {e}")

    print(f"\n{'='*55}")
    print(f"  Verification Results")
    print(f"{'='*55}")
    print(f"  Verified OK:         {verified}")
    print(f"  Failed:              {failed}")
    print(f"  Still plaintext:     {plaintext_found}")
    print(f"{'='*55}\n")

    if failed > 0:
        print("❌ VERIFICATION FAILED — some notes cannot be decrypted.")
        return False
    elif plaintext_found > 0:
        print(f"⚠️  {plaintext_found} notes still have plaintext. Run migration first.")
        return False
    else:
        print("✅ All encrypted notes verified successfully.")
        return True


def migrate(dry_run=False):
    """Run the encryption migration."""
    key, notes_col, trash_col, db_name = _get_connection()
    print(f"✅ Encryption key derived")

    print(f"\n{'='*55}")
    print(f"  AirScribe — Encryption Migration{' (DRY RUN)' if dry_run else ''}")
    print(f"  Database: {db_name}")
    print(f"  Time: {datetime.utcnow().isoformat()}Z")
    print(f"{'='*55}\n")

    encrypted_count = 0
    skipped_count = 0
    error_count = 0

    # Process notes collection
    all_notes = list(notes_col.find())
    total = len(all_notes)
    print(f"📋 Found {total} notes to process")

    for i, note in enumerate(all_notes):
        note_id = str(note["_id"])
        name = note.get("name", "Untitled")

        # Skip already-encrypted notes
        if is_encrypted(note):
            skipped_count += 1
            continue

        try:
            plaintext = note.get("text", "")
            html_content = note.get("content", "")

            # Encrypt text
            ct_text, nonce_text = encrypt(plaintext, key, note_id)

            # Encrypt content
            ct_content, nonce_content = encrypt(html_content, key, note_id)

            # Generate embedding
            embedding = generate_embedding(plaintext)

            if not dry_run:
                # Update the document
                update = {
                    "$set": {
                        "encrypted_text": ct_text,
                        "encryption_nonce_text": nonce_text,
                        "encrypted_content": ct_content,
                        "encryption_nonce_content": nonce_content,
                        "encryption_version": 1,
                        "embedding": embedding if embedding else [],
                    },
                    "$unset": {
                        "text": "",
                        "content": "",
                    }
                }
                notes_col.update_one({"_id": note["_id"]}, update)

            encrypted_count += 1

            if (i + 1) % 10 == 0 or i == total - 1:
                print(f"  Progress: {i+1}/{total} ({encrypted_count} encrypted, {skipped_count} skipped)")

        except Exception as e:
            error_count += 1
            print(f"  ❌ Error encrypting note '{name}' ({note_id}): {e}")

    # Process trash collection
    all_trash = list(trash_col.find())
    trash_total = len(all_trash)
    trash_encrypted = 0
    print(f"\n📋 Found {trash_total} trash notes to process")

    for note in all_trash:
        note_id = str(note["_id"])

        if is_encrypted(note):
            continue

        try:
            plaintext = note.get("text", "")
            html_content = note.get("content", "")

            ct_text, nonce_text = encrypt(plaintext, key, note_id)
            ct_content, nonce_content = encrypt(html_content, key, note_id)
            embedding = generate_embedding(plaintext)

            if not dry_run:
                update = {
                    "$set": {
                        "encrypted_text": ct_text,
                        "encryption_nonce_text": nonce_text,
                        "encrypted_content": ct_content,
                        "encryption_nonce_content": nonce_content,
                        "encryption_version": 1,
                        "embedding": embedding if embedding else [],
                    },
                    "$unset": {
                        "text": "",
                        "content": "",
                    }
                }
                trash_col.update_one({"_id": note["_id"]}, update)

            trash_encrypted += 1

        except Exception as e:
            error_count += 1
            print(f"  ❌ Error encrypting trash note ({note_id}): {e}")

    # ── Summary ──
    print(f"\n{'='*55}")
    print(f"  Migration {'Preview' if dry_run else 'Complete'}")
    print(f"{'='*55}")
    print(f"  Notes encrypted:     {encrypted_count}")
    print(f"  Notes skipped:       {skipped_count} (already encrypted)")
    print(f"  Trash encrypted:     {trash_encrypted}")
    print(f"  Errors:              {error_count}")
    print(f"{'='*55}\n")

    if dry_run:
        print("ℹ️  Dry run complete. No data was modified.")
        return

    if error_count > 0:
        print("⚠️  Some notes failed to encrypt. Review the errors above.")
        sys.exit(1)

    # ── Post-migration verification ──
    print("🔍 Running post-migration verification...")
    success = verify(key, notes_col, trash_col)

    if success:
        print("✅ All notes encrypted and verified successfully.")
        print("   Plaintext has been permanently removed from the database.")
        print("   KEEP YOUR ENCRYPTION KEYS SAFE — losing them means losing all note data.")
    else:
        print("⚠️  Verification found issues. Review the output above.")
        sys.exit(1)


if __name__ == "__main__":
    if "--verify" in sys.argv:
        key, notes_col, trash_col, db_name = _get_connection()
        verify(key, notes_col, trash_col)
    elif "--dry-run" in sys.argv:
        migrate(dry_run=True)
    else:
        migrate()

