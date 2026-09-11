"""
Tests for AirScribe secure notes orchestration (secure_notes.py).

These tests verify:
- Note encryption for storage
- Note decryption for response
- Batch decryption
- Re-encryption for duplication
- Legacy plaintext passthrough
- User isolation / authorization

Integration tests verify:
- Full round-trip through the orchestration layer
- Plaintext is not present in stored documents
- Encryption keys are environment-based
"""

import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── Test fixtures ──

TEST_MASTER_KEY = "a" * 64  # 32 bytes in hex
TEST_SALT = "b" * 32        # 16 bytes in hex


@pytest.fixture(autouse=True)
def setup_encryption_env():
    """Set encryption env vars for all tests in this module."""
    # Reset the cached key before each test
    import secure_notes
    secure_notes._derived_key = None

    with patch.dict(os.environ, {
        "NOTE_ENCRYPTION_MASTER_KEY": TEST_MASTER_KEY,
        "NOTE_ENCRYPTION_SALT": TEST_SALT,
    }):
        yield
    # Reset after test
    secure_notes._derived_key = None


@pytest.fixture
def sample_note():
    """A sample plaintext note dict."""
    return {
        "userId": "user_abc",
        "name": "Test Note",
        "text": "This is my secret note content about quantum physics.",
        "content": "<p>This is my <strong>secret</strong> note content about quantum physics.</p>",
        "folder": "Study",
        "color": "#ff0000",
        "tags": ["science", "physics"],
        "pinned": False,
        "createdAt": "2024-01-01T00:00:00Z",
        "updatedAt": "2024-01-01T00:00:00Z",
    }


# ── Prepare for Storage Tests ──

class TestPrepareNoteForStorage:
    def test_encrypts_text_field(self, sample_note):
        from secure_notes import prepare_note_for_storage
        result = prepare_note_for_storage(sample_note, note_id="test_id_1")
        # Original plaintext should NOT be in the result
        assert "text" not in result
        assert "content" not in result
        # Encrypted fields should be present
        assert "encrypted_text" in result
        assert "encryption_nonce_text" in result
        assert "encrypted_content" in result
        assert "encryption_nonce_content" in result

    def test_preserves_metadata(self, sample_note):
        from secure_notes import prepare_note_for_storage
        result = prepare_note_for_storage(sample_note, note_id="test_id_2")
        assert result["name"] == "Test Note"
        assert result["folder"] == "Study"
        assert result["tags"] == ["science", "physics"]
        assert result["userId"] == "user_abc"
        assert result["color"] == "#ff0000"

    def test_generates_embedding(self, sample_note):
        from secure_notes import prepare_note_for_storage
        result = prepare_note_for_storage(sample_note, note_id="test_id_3")
        assert "embedding" in result
        assert isinstance(result["embedding"], list)
        assert len(result["embedding"]) > 0

    def test_sets_encryption_version(self, sample_note):
        from secure_notes import prepare_note_for_storage
        result = prepare_note_for_storage(sample_note, note_id="test_id_4")
        assert result["encryption_version"] == 1

    def test_ciphertext_differs_from_plaintext(self, sample_note):
        from secure_notes import prepare_note_for_storage
        result = prepare_note_for_storage(sample_note, note_id="test_id_5")
        # Ciphertext should not contain the plaintext
        assert sample_note["text"] not in result["encrypted_text"]
        assert sample_note["content"] not in result["encrypted_content"]

    def test_different_encryptions_produce_different_ciphertext(self, sample_note):
        from secure_notes import prepare_note_for_storage
        result1 = prepare_note_for_storage(sample_note, note_id="test_id_6")
        result2 = prepare_note_for_storage(sample_note, note_id="test_id_6")
        # Different nonces → different ciphertext
        assert result1["encrypted_text"] != result2["encrypted_text"]
        assert result1["encryption_nonce_text"] != result2["encryption_nonce_text"]

    def test_empty_text_encrypts_without_error(self):
        from secure_notes import prepare_note_for_storage
        note = {"text": "", "content": "", "name": "Empty Note"}
        result = prepare_note_for_storage(note, note_id="test_empty")
        assert "encrypted_text" in result
        assert "encrypted_content" in result


# ── Prepare for Response Tests ──

class TestPrepareNoteForResponse:
    def test_decrypts_encrypted_note(self, sample_note):
        from secure_notes import prepare_note_for_storage, prepare_note_for_response
        encrypted = prepare_note_for_storage(sample_note, note_id="test_rt_1")
        encrypted["_id"] = "test_rt_1"
        decrypted = prepare_note_for_response(encrypted)
        assert decrypted["text"] == sample_note["text"]
        assert decrypted["content"] == sample_note["content"]

    def test_removes_encrypted_fields(self, sample_note):
        from secure_notes import prepare_note_for_storage, prepare_note_for_response
        encrypted = prepare_note_for_storage(sample_note, note_id="test_rt_2")
        encrypted["_id"] = "test_rt_2"
        decrypted = prepare_note_for_response(encrypted)
        assert "encrypted_text" not in decrypted
        assert "encryption_nonce_text" not in decrypted
        assert "encrypted_content" not in decrypted
        assert "encryption_nonce_content" not in decrypted
        assert "embedding" not in decrypted
        assert "encryption_version" not in decrypted

    def test_preserves_metadata_on_decrypt(self, sample_note):
        from secure_notes import prepare_note_for_storage, prepare_note_for_response
        encrypted = prepare_note_for_storage(sample_note, note_id="test_rt_3")
        encrypted["_id"] = "test_rt_3"
        decrypted = prepare_note_for_response(encrypted)
        assert decrypted["name"] == "Test Note"
        assert decrypted["folder"] == "Study"
        assert decrypted["tags"] == ["science", "physics"]

    def test_passthrough_plaintext_note(self):
        """Legacy plaintext notes should pass through unchanged."""
        from secure_notes import prepare_note_for_response
        plaintext_note = {
            "_id": "legacy_1",
            "text": "This is a legacy note",
            "content": "<p>Legacy</p>",
            "name": "Legacy Note",
        }
        result = prepare_note_for_response(plaintext_note)
        assert result["text"] == "This is a legacy note"
        assert result["content"] == "<p>Legacy</p>"

    def test_handles_decryption_failure_gracefully(self, sample_note):
        """If decryption fails, should return error placeholder, not crash."""
        from secure_notes import prepare_note_for_response
        # Create a note with invalid ciphertext
        bad_note = {
            "_id": "bad_note",
            "encrypted_text": "invalid_base64!!",
            "encryption_nonce_text": "invalid!!",
            "encrypted_content": "invalid!!",
            "encryption_nonce_content": "invalid!!",
            "encryption_version": 1,
            "name": "Bad Note",
        }
        result = prepare_note_for_response(bad_note)
        assert "[Decryption failed]" in result.get("text", "")


# ── Batch Decryption Tests ──

class TestPrepareNotesForResponse:
    def test_batch_decryption(self, sample_note):
        from secure_notes import prepare_note_for_storage, prepare_notes_for_response
        notes = []
        for i in range(3):
            note = sample_note.copy()
            note["name"] = f"Note {i}"
            note["text"] = f"Content for note {i}"
            encrypted = prepare_note_for_storage(note, note_id=f"batch_{i}")
            encrypted["_id"] = f"batch_{i}"
            notes.append(encrypted)

        decrypted = prepare_notes_for_response(notes)
        assert len(decrypted) == 3
        for i, d in enumerate(decrypted):
            assert d["text"] == f"Content for note {i}"
            assert d["name"] == f"Note {i}"


# ── Re-encryption Tests ──

class TestReEncryption:
    def test_re_encrypt_for_new_id(self, sample_note):
        from secure_notes import (
            prepare_note_for_storage,
            prepare_note_for_response,
            re_encrypt_for_new_id
        )
        # Encrypt with original ID
        encrypted = prepare_note_for_storage(sample_note, note_id="original_id")
        encrypted["_id"] = "original_id"

        # Re-encrypt with new ID
        re_encrypted = re_encrypt_for_new_id(encrypted, "new_id")
        re_encrypted["_id"] = "new_id"

        # Should decrypt with new ID
        decrypted = prepare_note_for_response(re_encrypted)
        assert decrypted["text"] == sample_note["text"]
        assert decrypted["content"] == sample_note["content"]

    def test_re_encrypt_produces_different_ciphertext(self, sample_note):
        from secure_notes import prepare_note_for_storage, re_encrypt_for_new_id
        encrypted = prepare_note_for_storage(sample_note, note_id="orig")
        encrypted["_id"] = "orig"

        re_encrypted = re_encrypt_for_new_id(encrypted, "copy")
        # Different nonces and ciphertext
        assert encrypted["encrypted_text"] != re_encrypted["encrypted_text"]


# ── Round-Trip Integration Tests ──

class TestRoundTrip:
    def test_full_lifecycle(self, sample_note):
        """Test create → encrypt → store → retrieve → decrypt."""
        from secure_notes import prepare_note_for_storage, prepare_note_for_response

        # 1. Encrypt for storage
        stored = prepare_note_for_storage(sample_note, note_id="lifecycle_1")
        stored["_id"] = "lifecycle_1"

        # 2. Verify no plaintext in stored version
        assert "text" not in stored
        assert "content" not in stored
        assert sample_note["text"] not in str(stored.get("encrypted_text", ""))

        # 3. Decrypt for response
        response = prepare_note_for_response(stored)
        assert response["text"] == sample_note["text"]
        assert response["content"] == sample_note["content"]

    def test_unicode_round_trip(self):
        from secure_notes import prepare_note_for_storage, prepare_note_for_response
        note = {
            "text": "Hello 🌍 مرحبا 你好 こんにちは",
            "content": "<p>Hello 🌍 مرحبا 你好 こんにちは</p>",
            "name": "Unicode Note",
        }
        stored = prepare_note_for_storage(note, note_id="unicode_1")
        stored["_id"] = "unicode_1"
        response = prepare_note_for_response(stored)
        assert response["text"] == note["text"]
        assert response["content"] == note["content"]


# ── Storage Security Tests ──

class TestStorageSecurity:
    def test_plaintext_not_in_stored_document(self, sample_note):
        """Verify that plaintext note content is not present in stored form."""
        from secure_notes import prepare_note_for_storage
        import json
        stored = prepare_note_for_storage(sample_note, note_id="sec_1")
        stored_str = json.dumps(stored)
        assert "quantum physics" not in stored_str.lower()
        assert sample_note["text"] not in stored_str

    def test_encryption_keys_not_hardcoded(self):
        """Verify that encryption keys come from environment."""
        from secure_notes import _get_key
        import secure_notes
        # Reset cached key
        secure_notes._derived_key = None
        # Key derivation should use env vars
        key = _get_key()
        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_missing_keys_raise_error(self):
        """Verify that missing encryption keys cause an error."""
        import secure_notes
        secure_notes._derived_key = None
        with patch.dict(os.environ, {
            "NOTE_ENCRYPTION_MASTER_KEY": "",
            "NOTE_ENCRYPTION_SALT": "",
        }):
            with pytest.raises(ValueError):
                secure_notes._get_key()


# ── User Authorization Tests ──
# These test the authorization logic pattern used in app.py routes.
# Full integration would require running the Flask app.

class TestUserAuthorization:
    def test_user_scoped_note_data(self, sample_note):
        """Verify that userId is preserved through encryption."""
        from secure_notes import prepare_note_for_storage, prepare_note_for_response
        stored = prepare_note_for_storage(sample_note, note_id="auth_1")
        stored["_id"] = "auth_1"
        assert stored["userId"] == "user_abc"
        response = prepare_note_for_response(stored)
        assert response["userId"] == "user_abc"

    def test_different_user_encryption_independent(self, sample_note):
        """Notes from different users encrypt independently."""
        from secure_notes import prepare_note_for_storage
        note_a = sample_note.copy()
        note_a["userId"] = "user_a"
        note_b = sample_note.copy()
        note_b["userId"] = "user_b"

        stored_a = prepare_note_for_storage(note_a, note_id="auth_a")
        stored_b = prepare_note_for_storage(note_b, note_id="auth_b")

        # Different notes, different ciphertext
        assert stored_a["encrypted_text"] != stored_b["encrypted_text"]
        # But both should have their userId preserved
        assert stored_a["userId"] == "user_a"
        assert stored_b["userId"] == "user_b"
