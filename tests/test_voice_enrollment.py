"""
Tests for ai_server.voice_enrollment Flask blueprint.

Tests the enrollment API routes including validation, error handling,
and profile CRUD operations.
"""

import os
import sys
import json
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_server.voice_enrollment import voice_enrollment_bp
from flask import Flask


def create_test_app():
    """Create a test Flask app with the voice enrollment blueprint."""
    app = Flask(__name__)
    app.config["TESTING"] = True
    app.register_blueprint(voice_enrollment_bp)
    return app


class TestEnrollVoiceValidation(unittest.TestCase):
    """Tests for POST /ai/voice/enroll validation."""

    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()

    def test_no_audio_file(self):
        """Should return 400 when no audio file is provided."""
        res = self.client.post("/ai/voice/enroll", data={
            "userId": "user1",
            "userName": "Test"
        })
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("error", data)

    def test_no_user_id(self):
        """Should return 400 when userId is missing."""
        from io import BytesIO
        res = self.client.post("/ai/voice/enroll", data={
            "audio": (BytesIO(b"fake audio data" * 100), "test.webm"),
            "userName": "Test"
        })
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("userId", data["error"])

    def test_no_user_name(self):
        """Should return 400 when userName is missing."""
        from io import BytesIO
        res = self.client.post("/ai/voice/enroll", data={
            "audio": (BytesIO(b"fake audio data" * 100), "test.webm"),
            "userId": "user1"
        })
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("userName", data["error"])

    def test_invalid_file_extension(self):
        """Should return 400 for unsupported file types."""
        from io import BytesIO
        res = self.client.post("/ai/voice/enroll", data={
            "audio": (BytesIO(b"not audio"), "test.txt"),
            "userId": "user1",
            "userName": "Test"
        })
        self.assertEqual(res.status_code, 400)
        data = json.loads(res.data)
        self.assertIn("Invalid file type", data["error"])

    def test_empty_audio_file(self):
        """Should return 400 when audio file is empty."""
        from io import BytesIO
        res = self.client.post("/ai/voice/enroll", data={
            "audio": (BytesIO(b""), "test.webm"),
            "userId": "user1",
            "userName": "Test"
        })
        self.assertEqual(res.status_code, 400)


class TestListProfiles(unittest.TestCase):
    """Tests for GET /ai/voice/profiles."""

    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()

    @patch("ai_server.voice_enrollment.embeddings")
    def test_returns_empty_list(self, mock_embeddings):
        """Should return empty list when no profiles exist."""
        mock_embeddings.get_all_profiles_metadata.return_value = []
        res = self.client.get("/ai/voice/profiles")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data, [])

    @patch("ai_server.voice_enrollment.embeddings")
    def test_returns_profiles(self, mock_embeddings):
        """Should return profile metadata without embeddings."""
        mock_embeddings.get_all_profiles_metadata.return_value = [
            {
                "_id": "abc123",
                "userId": "user1",
                "userName": "Alice",
                "embeddingModel": "speechbrain-ecapa-voxceleb",
                "sampleDuration": 30.0,
                "createdAt": "2024-01-01T00:00:00Z"
            }
        ]
        res = self.client.get("/ai/voice/profiles")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["userName"], "Alice")
        # Embedding should NOT be present
        self.assertNotIn("voiceEmbedding", data[0])

    @patch("ai_server.voice_enrollment.embeddings")
    def test_filter_by_user_id(self, mock_embeddings):
        """Should pass userId filter to embeddings module."""
        mock_embeddings.get_all_profiles_metadata.return_value = []
        self.client.get("/ai/voice/profiles?userId=user1")
        mock_embeddings.get_all_profiles_metadata.assert_called_with("user1")


class TestDeleteProfile(unittest.TestCase):
    """Tests for DELETE /ai/voice/profile/<id>."""

    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()

    @patch("ai_server.voice_enrollment.embeddings")
    def test_successful_deletion(self, mock_embeddings):
        """Should return 200 on successful deletion."""
        mock_embeddings.delete_embedding.return_value = True
        res = self.client.delete("/ai/voice/profile/abc123")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertTrue(data["success"])

    @patch("ai_server.voice_enrollment.embeddings")
    def test_not_found_deletion(self, mock_embeddings):
        """Should return 404 when profile doesn't exist."""
        mock_embeddings.delete_embedding.return_value = False
        res = self.client.delete("/ai/voice/profile/nonexistent")
        self.assertEqual(res.status_code, 404)


class TestGetProfile(unittest.TestCase):
    """Tests for GET /ai/voice/profile/<id>."""

    def setUp(self):
        self.app = create_test_app()
        self.client = self.app.test_client()

    @patch("ai_server.voice_enrollment.embeddings")
    def test_profile_found(self, mock_embeddings):
        """Should return profile metadata without embedding."""
        mock_embeddings.get_embedding.return_value = {
            "_id": "abc123",
            "userId": "user1",
            "userName": "Alice",
            "voiceEmbedding": [0.1, 0.2, 0.3],  # Should be stripped
            "embeddingModel": "speechbrain-ecapa-voxceleb"
        }
        res = self.client.get("/ai/voice/profile/abc123")
        self.assertEqual(res.status_code, 200)
        data = json.loads(res.data)
        self.assertEqual(data["userName"], "Alice")
        self.assertNotIn("voiceEmbedding", data)

    @patch("ai_server.voice_enrollment.embeddings")
    def test_profile_not_found(self, mock_embeddings):
        """Should return 404 when profile doesn't exist."""
        mock_embeddings.get_embedding.return_value = None
        res = self.client.get("/ai/voice/profile/nonexistent")
        self.assertEqual(res.status_code, 404)


if __name__ == "__main__":
    unittest.main()
