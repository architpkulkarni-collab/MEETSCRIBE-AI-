"""
Tests for AirScribe encryption module (encryption.py).
"""

import os
import pytest
from unittest.mock import patch


# ── Test fixtures ──

@pytest.fixture
def master_key():
    """A valid 64-char hex master key for testing."""
    return "a" * 64  # 32 bytes in hex


@pytest.fixture
def salt():
    """A valid 32-char hex salt for testing."""
    return "b" * 32  # 16 bytes in hex


@pytest.fixture
def key(master_key, salt):
    """Derived encryption key."""
    from encryption import derive_key
    return derive_key(master_key, salt)


# ── Key Derivation Tests ──

class TestKeyDerivation:
    def test_derive_key_produces_32_bytes(self, master_key, salt):
        from encryption import derive_key
        key = derive_key(master_key, salt)
        assert len(key) == 32
        assert isinstance(key, bytes)

    def test_derive_key_is_deterministic(self, master_key, salt):
        from encryption import derive_key
        key1 = derive_key(master_key, salt)
        key2 = derive_key(master_key, salt)
        assert key1 == key2

    def test_different_salts_produce_different_keys(self, master_key):
        from encryption import derive_key
        key1 = derive_key(master_key, "c" * 32)
        key2 = derive_key(master_key, "d" * 32)
        assert key1 != key2

    def test_different_master_keys_produce_different_keys(self, salt):
        from encryption import derive_key
        key1 = derive_key("e" * 64, salt)
        key2 = derive_key("f" * 64, salt)
        assert key1 != key2

    def test_missing_master_key_raises(self, salt):
        from encryption import derive_key
        with pytest.raises(ValueError, match="must be set"):
            derive_key("", salt)

    def test_missing_salt_raises(self, master_key):
        from encryption import derive_key
        with pytest.raises(ValueError, match="must be set"):
            derive_key(master_key, "")

    def test_short_master_key_raises(self, salt):
        from encryption import derive_key
        with pytest.raises(ValueError, match="at least 16 bytes"):
            derive_key("ab" * 8, salt)  # 8 bytes, too short

    def test_invalid_hex_raises(self, salt):
        from encryption import derive_key
        with pytest.raises(ValueError, match="valid hex"):
            derive_key("not_hex_string_at_all_!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!", salt)


# ── Encryption / Decryption Round-Trip Tests ──

class TestEncryptDecrypt:
    def test_round_trip_basic(self, key):
        from encryption import encrypt, decrypt
        plaintext = "Hello, World!"
        ct, nonce = encrypt(plaintext, key)
        result = decrypt(ct, nonce, key)
        assert result == plaintext

    def test_round_trip_with_note_id(self, key):
        from encryption import encrypt, decrypt
        plaintext = "Secret note content"
        note_id = "507f1f77bcf86cd799439011"
        ct, nonce = encrypt(plaintext, key, note_id)
        result = decrypt(ct, nonce, key, note_id)
        assert result == plaintext

    def test_round_trip_empty_string(self, key):
        from encryption import encrypt, decrypt
        ct, nonce = encrypt("", key)
        result = decrypt(ct, nonce, key)
        assert result == ""

    def test_round_trip_unicode(self, key):
        from encryption import encrypt, decrypt
        plaintext = "Hello 🌍 مرحبا 你好 こんにちは"
        ct, nonce = encrypt(plaintext, key)
        result = decrypt(ct, nonce, key)
        assert result == plaintext

    def test_round_trip_large_content(self, key):
        from encryption import encrypt, decrypt
        plaintext = "A" * 100_000  # 100KB
        ct, nonce = encrypt(plaintext, key)
        result = decrypt(ct, nonce, key)
        assert result == plaintext

    def test_round_trip_html_content(self, key):
        from encryption import encrypt, decrypt
        html = '<p>Hello <strong>World</strong></p><ul><li>Item 1</li></ul>'
        ct, nonce = encrypt(html, key)
        result = decrypt(ct, nonce, key)
        assert result == html

    def test_unique_nonces(self, key):
        from encryption import encrypt
        nonces = set()
        for _ in range(100):
            _, nonce = encrypt("test", key)
            nonces.add(nonce)
        # All 100 nonces should be unique
        assert len(nonces) == 100

    def test_different_ciphertext_same_plaintext(self, key):
        from encryption import encrypt
        ct1, _ = encrypt("same text", key)
        ct2, _ = encrypt("same text", key)
        # Different nonces → different ciphertext
        assert ct1 != ct2


# ── Tamper Detection Tests ──

class TestTamperDetection:
    def test_wrong_key_raises(self, master_key, salt):
        from encryption import derive_key, encrypt, decrypt
        key1 = derive_key(master_key, salt)
        key2 = derive_key("c" * 64, salt)  # Different key
        ct, nonce = encrypt("secret", key1)

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(ct, nonce, key2)

    def test_wrong_nonce_raises(self, key):
        from encryption import encrypt, decrypt
        import base64
        ct, nonce = encrypt("secret", key)
        # Use a different nonce
        wrong_nonce = base64.b64encode(os.urandom(12)).decode("ascii")

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(ct, wrong_nonce, key)

    def test_tampered_ciphertext_raises(self, key):
        from encryption import encrypt, decrypt
        import base64
        ct, nonce = encrypt("secret", key)
        # Tamper with ciphertext
        ct_bytes = bytearray(base64.b64decode(ct))
        ct_bytes[0] ^= 0xFF  # Flip bits
        tampered_ct = base64.b64encode(bytes(ct_bytes)).decode("ascii")

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(tampered_ct, nonce, key)

    def test_wrong_note_id_raises(self, key):
        from encryption import encrypt, decrypt
        ct, nonce = encrypt("secret", key, note_id="note_1")

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(ct, nonce, key, note_id="note_2")

    def test_missing_note_id_when_expected_raises(self, key):
        from encryption import encrypt, decrypt
        ct, nonce = encrypt("secret", key, note_id="note_1")

        from cryptography.exceptions import InvalidTag
        with pytest.raises(InvalidTag):
            decrypt(ct, nonce, key, note_id="")  # No AAD


# ── Helper Tests ──

class TestIsEncrypted:
    def test_encrypted_note_detected(self):
        from encryption import is_encrypted
        note = {
            "encrypted_text": "abc",
            "encryption_nonce_text": "def",
        }
        assert is_encrypted(note) is True

    def test_plaintext_note_not_detected(self):
        from encryption import is_encrypted
        note = {
            "text": "hello",
            "content": "<p>hello</p>",
        }
        assert is_encrypted(note) is False

    def test_empty_note_not_detected(self):
        from encryption import is_encrypted
        assert is_encrypted({}) is False
