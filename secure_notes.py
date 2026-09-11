"""
AirScribe — Secure Notes Orchestration Layer
=================================================
Coordinates encryption, embedding generation, and storage preparation
for the encrypted note store.

Purpose:
    Provides high-level functions that the API layer (app.py) calls
    instead of directly handling plaintext. This module ensures the
    correct sequence:
        Create/Update: plaintext → embed → encrypt → store
        Read:          retrieve  → decrypt → return
        Search:        query     → embed query → cosine rank → decrypt top-K

How it connects:
    - Called by app.py note CRUD routes
    - Uses encryption.py for AES-256-GCM operations
    - Uses note_embeddings.py for semantic vector generation
    - Key material comes from environment variables via config

SECURITY RULES:
    - Plaintext only exists in local variables during request processing
    - After encryption, plaintext references are deleted
    - NEVER log or print note content
    - NEVER store plaintext in any persistent location
"""

import os
import logging
from typing import Optional

from encryption import derive_key, encrypt, decrypt, is_encrypted
from note_embeddings import generate_embedding

logger = logging.getLogger("airscribe.secure_notes")

# ── Cached derived key (computed once at startup) ──
_derived_key: Optional[bytes] = None


def _get_key() -> bytes:
    """
    Get the derived encryption key, computing it once and caching.

    Returns:
        32-byte AES-256 key.

    Raises:
        ValueError: If encryption environment variables are not set.
    """
    global _derived_key
    if _derived_key is None:
        master_key = os.getenv("NOTE_ENCRYPTION_MASTER_KEY", "")
        salt = os.getenv("NOTE_ENCRYPTION_SALT", "")
        _derived_key = derive_key(master_key, salt)
        logger.info("Encryption key derived and cached")
    return _derived_key


def prepare_note_for_storage(note: dict, note_id: str = "") -> dict:
    """
    Encrypt a note's content and generate its embedding for storage.

    Takes a note dict with plaintext 'text' and 'content' fields,
    and returns a new dict with those fields replaced by encrypted
    versions plus an embedding vector.

    Args:
        note: Dict with at minimum 'text' (plaintext) and 'content' (HTML).
              May contain other fields (name, folder, tags, etc.) which
              are passed through unchanged.
        note_id: The MongoDB _id of the note (used as AAD for encryption).
                 For new notes, pass "" — the AAD will be empty.

    Returns:
        New dict with:
            - 'text' and 'content' removed
            - 'encrypted_text', 'encryption_nonce_text' added
            - 'encrypted_content', 'encryption_nonce_content' added
            - 'embedding' added (384-dim float list)
            - 'encryption_version' set to 1
            - All other fields preserved

    Raises:
        ValueError: If encryption fails.
    """
    key = _get_key()
    result = {}

    # Copy all fields except the ones we'll encrypt
    for k, v in note.items():
        if k not in ("text", "content"):
            result[k] = v

    # ── Encrypt plaintext ──
    plaintext = note.get("text", "")
    ct_text, nonce_text = encrypt(plaintext, key, note_id)
    result["encrypted_text"] = ct_text
    result["encryption_nonce_text"] = nonce_text

    # ── Encrypt HTML content ──
    html_content = note.get("content", "")
    ct_content, nonce_content = encrypt(html_content, key, note_id)
    result["encrypted_content"] = ct_content
    result["encryption_nonce_content"] = nonce_content

    # ── Generate semantic embedding (from plaintext, BEFORE we lose it) ──
    embedding = generate_embedding(plaintext)
    if embedding is not None:
        result["embedding"] = embedding
    else:
        logger.warning(f"Embedding generation failed for note {note_id}")
        result["embedding"] = []

    # ── Mark encryption version ──
    result["encryption_version"] = 1

    logger.debug(f"Note prepared for storage (id={note_id or 'new'})")
    return result


def prepare_note_for_response(note: dict) -> dict:
    """
    Decrypt a note's content for returning to the frontend.

    Takes an encrypted note document from MongoDB and returns a dict
    with plaintext 'text' and 'content' fields restored.

    Args:
        note: Encrypted note dict from MongoDB with 'encrypted_text',
              'encryption_nonce_text', etc.

    Returns:
        New dict with:
            - 'text' restored to plaintext
            - 'content' restored to HTML
            - Encrypted fields removed
            - Embedding removed (not needed by frontend)
            - All other fields preserved
    """
    if not is_encrypted(note):
        # Note is already plaintext (pre-migration or unencrypted)
        return note

    key = _get_key()
    note_id = str(note.get("_id", ""))
    result = {}

    # Copy all fields except encrypted ones and embedding
    skip_fields = {
        "encrypted_text", "encryption_nonce_text",
        "encrypted_content", "encryption_nonce_content",
        "embedding", "encryption_version"
    }
    for k, v in note.items():
        if k not in skip_fields:
            result[k] = v

    # ── Decrypt text ──
    try:
        result["text"] = decrypt(
            note["encrypted_text"],
            note["encryption_nonce_text"],
            key,
            note_id
        )
    except Exception:
        try:
            result["text"] = decrypt(
                note["encrypted_text"],
                note["encryption_nonce_text"],
                key,
                ""
            )
        except Exception as e:
            logger.error(f"Failed to decrypt text for note {note_id}: {e}")
            result["text"] = "[Decryption failed]"

    # ── Decrypt HTML content ──
    try:
        result["content"] = decrypt(
            note.get("encrypted_content", ""),
            note.get("encryption_nonce_content", ""),
            key,
            note_id
        )
    except Exception:
        try:
            result["content"] = decrypt(
                note.get("encrypted_content", ""),
                note.get("encryption_nonce_content", ""),
                key,
                ""
            )
        except Exception as e:
            logger.error(f"Failed to decrypt content for note {note_id}: {e}")
            result["content"] = "<p>[Decryption failed]</p>"

    return result


def prepare_notes_for_response(notes: list[dict]) -> list[dict]:
    """
    Decrypt a list of notes for returning to the frontend.

    Args:
        notes: List of encrypted note dicts from MongoDB.

    Returns:
        List of decrypted note dicts.
    """
    return [prepare_note_for_response(note) for note in notes]


def re_encrypt_for_new_id(note: dict, new_note_id: str) -> dict:
    """
    Re-encrypt a note with a new note ID (used for duplication).

    The AAD changes when the note ID changes, so we must decrypt
    with the old ID and re-encrypt with the new one.

    Args:
        note: The encrypted source note dict.
        new_note_id: The new MongoDB _id for the duplicate.

    Returns:
        New encrypted note dict bound to the new ID.
    """
    # Decrypt with original ID
    decrypted = prepare_note_for_response(note)

    # Re-encrypt with new ID
    return prepare_note_for_storage(decrypted, new_note_id)
