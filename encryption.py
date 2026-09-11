"""
AirScribe — Note Encryption Service
======================================
AES-256-GCM encryption for note content with HKDF key derivation.

Purpose:
    Provides encrypt/decrypt functions for note text and HTML content.
    Uses AES-256-GCM (authenticated encryption with associated data) to
    ensure both confidentiality and integrity of note content.

Security guarantees:
    - Each encryption operation uses a unique random 96-bit nonce
    - Associated data (AAD) binds ciphertext to a specific note ID,
      preventing ciphertext-swapping attacks
    - HKDF derives the AES key from a high-entropy master secret,
      so the master key is never used directly
    - Tampered ciphertext is detected and rejected (GCM auth tag)

How it connects:
    Called by secure_notes.py to encrypt before storage and decrypt
    after retrieval. The master key comes from environment variables
    via ai_server/config.py.

SECURITY RULES:
    - NEVER log plaintext content
    - NEVER log encryption keys
    - NEVER use a hardcoded key
    - NEVER reuse a nonce
"""

import os
import base64
import logging
from typing import Tuple

from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from cryptography.hazmat.primitives import hashes
from cryptography.exceptions import InvalidTag

logger = logging.getLogger("airscribe.encryption")

# ── Constants ──
_NONCE_SIZE = 12       # 96-bit nonce (recommended for AES-GCM)
_KEY_SIZE = 32         # 256-bit key
_INFO = b"airscribe-note-encryption-v1"


def derive_key(master_key_hex: str, salt_hex: str) -> bytes:
    """
    Derive a 256-bit AES key from a master secret using HKDF-SHA256.

    Args:
        master_key_hex: The master encryption key as a hex string (64 chars = 32 bytes).
        salt_hex: The derivation salt as a hex string (32 chars = 16 bytes).

    Returns:
        32-byte derived key suitable for AES-256-GCM.

    Raises:
        ValueError: If master_key_hex or salt_hex are missing or malformed.
    """
    if not master_key_hex or not salt_hex:
        raise ValueError(
            "Encryption master key and salt must be set. "
            "Set NOTE_ENCRYPTION_MASTER_KEY and NOTE_ENCRYPTION_SALT in .env"
        )

    try:
        master_key_bytes = bytes.fromhex(master_key_hex)
        salt_bytes = bytes.fromhex(salt_hex)
    except ValueError as e:
        raise ValueError(f"Encryption key/salt must be valid hex strings: {e}") from e

    if len(master_key_bytes) < 16:
        raise ValueError("Master key must be at least 16 bytes (32 hex chars)")

    hkdf = HKDF(
        algorithm=hashes.SHA256(),
        length=_KEY_SIZE,
        salt=salt_bytes,
        info=_INFO,
    )

    derived = hkdf.derive(master_key_bytes)
    logger.debug("Encryption key derived successfully")
    return derived


def encrypt(plaintext: str, key: bytes, note_id: str = "") -> Tuple[str, str]:
    """
    Encrypt a plaintext string using AES-256-GCM.

    Args:
        plaintext: The text to encrypt (note content).
        key: 32-byte AES key (from derive_key).
        note_id: The note's MongoDB _id as a string. Used as Associated
                 Authenticated Data (AAD) to bind ciphertext to this note.

    Returns:
        Tuple of (ciphertext_base64, nonce_base64).
        Both are standard base64-encoded strings safe for JSON/BSON storage.

    Raises:
        ValueError: If key is wrong size.
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")

    plaintext_bytes = plaintext.encode("utf-8")
    aad = note_id.encode("utf-8") if note_id else None

    # Generate a unique random nonce for this encryption
    nonce = os.urandom(_NONCE_SIZE)

    aesgcm = AESGCM(key)
    ciphertext = aesgcm.encrypt(nonce, plaintext_bytes, aad)

    # Encode for storage
    ciphertext_b64 = base64.b64encode(ciphertext).decode("ascii")
    nonce_b64 = base64.b64encode(nonce).decode("ascii")

    return ciphertext_b64, nonce_b64


def decrypt(ciphertext_b64: str, nonce_b64: str, key: bytes, note_id: str = "") -> str:
    """
    Decrypt AES-256-GCM ciphertext back to plaintext.

    Args:
        ciphertext_b64: Base64-encoded ciphertext (from encrypt).
        nonce_b64: Base64-encoded nonce (from encrypt).
        key: 32-byte AES key (same key used for encryption).
        note_id: The note's MongoDB _id (must match the AAD used during encryption).

    Returns:
        The decrypted plaintext string.

    Raises:
        cryptography.exceptions.InvalidTag: If the ciphertext is tampered,
            the key is wrong, or the note_id doesn't match.
        ValueError: If inputs are malformed.
    """
    if len(key) != _KEY_SIZE:
        raise ValueError(f"Key must be {_KEY_SIZE} bytes, got {len(key)}")

    try:
        ciphertext = base64.b64decode(ciphertext_b64)
        nonce = base64.b64decode(nonce_b64)
    except Exception as e:
        raise ValueError(f"Invalid base64 input: {e}") from e

    aad = note_id.encode("utf-8") if note_id else None

    aesgcm = AESGCM(key)
    plaintext_bytes = aesgcm.decrypt(nonce, ciphertext, aad)

    return plaintext_bytes.decode("utf-8")


def is_encrypted(note: dict) -> bool:
    """
    Check if a note document is in encrypted format.

    Args:
        note: A MongoDB note document dict.

    Returns:
        True if the note contains encrypted fields.
    """
    return "encrypted_text" in note and "encryption_nonce_text" in note
