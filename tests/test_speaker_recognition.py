"""
Tests for ai_server.speaker_recognition module.

Tests embedding extraction, cosine similarity, and speaker identification.
Uses mock audio where needed to avoid requiring actual audio files.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestCosineSimilarity(unittest.TestCase):
    """Tests for the cosine_similarity function."""

    def setUp(self):
        from ai_server.speaker_recognition import cosine_similarity
        self.cosine_similarity = cosine_similarity

    def test_identical_vectors(self):
        v = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(self.cosine_similarity(v, v), 1.0, places=6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([-1.0, 0.0, 0.0])
        self.assertAlmostEqual(self.cosine_similarity(a, b), -1.0, places=6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        self.assertAlmostEqual(self.cosine_similarity(a, b), 0.0, places=6)

    def test_zero_vector(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.zeros(3)
        self.assertEqual(self.cosine_similarity(a, b), 0.0)

    def test_both_zero_vectors(self):
        a = np.zeros(3)
        b = np.zeros(3)
        self.assertEqual(self.cosine_similarity(a, b), 0.0)

    def test_high_dimensional(self):
        """Test with 192-dim vectors (same as ECAPA-TDNN output)."""
        np.random.seed(42)
        a = np.random.randn(192)
        b = a * 0.9 + np.random.randn(192) * 0.1  # Similar to a
        score = self.cosine_similarity(a, b)
        self.assertGreater(score, 0.5)  # Should be similar

    def test_return_type(self):
        a = np.array([1.0, 2.0])
        b = np.array([3.0, 4.0])
        result = self.cosine_similarity(a, b)
        self.assertIsInstance(result, float)


class TestIdentifySpeakerFromEmbedding(unittest.TestCase):
    """Tests for identify_speaker_from_embedding function."""

    def setUp(self):
        from ai_server.speaker_recognition import identify_speaker_from_embedding
        self.identify = identify_speaker_from_embedding

    def test_empty_profiles(self):
        emb = np.random.randn(192)
        name, score = self.identify(emb, [])
        self.assertEqual(name, "Unknown Speaker")
        self.assertEqual(score, 0.0)

    def test_exact_match(self):
        emb = np.random.randn(192)
        profiles = [{"userName": "Alice", "voiceEmbedding": emb.tolist()}]
        name, score = self.identify(emb, profiles, threshold=0.70)
        self.assertEqual(name, "Alice")
        self.assertAlmostEqual(score, 1.0, places=4)

    def test_below_threshold(self):
        """When similarity is below threshold, should return Unknown Speaker."""
        np.random.seed(99)
        emb_a = np.random.randn(192)
        emb_b = np.random.randn(192)  # Completely different
        profiles = [{"userName": "Bob", "voiceEmbedding": emb_b.tolist()}]
        name, score = self.identify(emb_a, profiles, threshold=0.90)
        self.assertEqual(name, "Unknown Speaker")

    def test_best_match_selected(self):
        """When multiple profiles, the best match should be selected."""
        np.random.seed(42)
        target = np.random.randn(192)
        # Profile A: very similar
        close = target * 0.95 + np.random.randn(192) * 0.05
        # Profile B: different
        far = np.random.randn(192)

        profiles = [
            {"userName": "Far", "voiceEmbedding": far.tolist()},
            {"userName": "Close", "voiceEmbedding": close.tolist()},
        ]
        name, score = self.identify(target, profiles, threshold=0.70)
        self.assertEqual(name, "Close")

    def test_missing_embedding_in_profile(self):
        """Profile with None embedding should be skipped."""
        emb = np.random.randn(192)
        profiles = [{"userName": "Ghost", "voiceEmbedding": None}]
        name, score = self.identify(emb, profiles)
        self.assertEqual(name, "Unknown Speaker")

    def test_numpy_array_in_profile(self):
        """Profile embedding can be a numpy array (not just a list)."""
        emb = np.random.randn(192)
        profiles = [{"userName": "Direct", "voiceEmbedding": emb.copy()}]
        name, score = self.identify(emb, profiles, threshold=0.70)
        self.assertEqual(name, "Direct")


class TestExtractEmbedding(unittest.TestCase):
    """Tests for extract_embedding function (requires mock since we don't have audio)."""

    @patch("ai_server.speaker_recognition.get_speaker_encoder")
    def test_returns_numpy_array(self, mock_encoder):
        """Should return a numpy array when extraction succeeds."""
        import torch
        mock_model = MagicMock()
        fake_embedding = torch.randn(1, 192)
        mock_model.encode_batch.return_value = fake_embedding
        mock_model.load_audio.return_value = torch.randn(16000)
        mock_encoder.return_value = mock_model

        from ai_server.speaker_recognition import extract_embedding
        result = extract_embedding("fake_audio.wav")

        self.assertIsNotNone(result)
        self.assertEqual(result.shape, (192,))
        self.assertIsInstance(result, np.ndarray)

    @patch("ai_server.speaker_recognition.get_speaker_encoder")
    def test_returns_none_on_error(self, mock_encoder):
        """Should return None when extraction fails."""
        mock_encoder.side_effect = RuntimeError("Model not loaded")

        from ai_server.speaker_recognition import extract_embedding
        result = extract_embedding("fake_audio.wav")

        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
