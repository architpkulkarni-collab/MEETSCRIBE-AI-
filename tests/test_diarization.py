"""
Tests for ai_server.diarization module.

Tests diarization fallback, Whisper-to-diarization mapping, and
simple silence-gap diarization.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestSimpleDiarizationFallback(unittest.TestCase):
    """Tests for the simple_diarization_fallback function."""

    def setUp(self):
        from ai_server.diarization import simple_diarization_fallback
        self.diarize = simple_diarization_fallback

    def test_empty_segments(self):
        result = self.diarize([])
        self.assertEqual(result, [])

    def test_single_segment(self):
        segments = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        result = self.diarize(segments, num_speakers=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[0]["text"], "Hello")

    def test_speaker_alternation_on_gap(self):
        """Speaker should change after a silence gap > 1.5 seconds."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "First speaker"},
            {"start": 4.0, "end": 6.0, "text": "Second speaker"},  # 2s gap
        ]
        result = self.diarize(segments, num_speakers=2)
        self.assertEqual(len(result), 2)
        self.assertNotEqual(result[0]["speaker"], result[1]["speaker"])

    def test_same_speaker_no_gap(self):
        """Speaker should NOT change when gap < 1.5 seconds."""
        segments = [
            {"start": 0.0, "end": 2.0, "text": "Same speaker"},
            {"start": 2.5, "end": 4.0, "text": "still talking"},  # 0.5s gap
        ]
        result = self.diarize(segments, num_speakers=2)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["speaker"], result[1]["speaker"])

    def test_wraps_around_speaker_count(self):
        """Speaker index should wrap around num_speakers."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 3.0, "end": 4.0, "text": "B"},  # Gap → speaker 1
            {"start": 6.0, "end": 7.0, "text": "C"},  # Gap → speaker 0 (wrap)
        ]
        result = self.diarize(segments, num_speakers=2)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")
        self.assertEqual(result[1]["speaker"], "SPEAKER_01")
        self.assertEqual(result[2]["speaker"], "SPEAKER_00")

    def test_empty_text_filtered(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": ""},
            {"start": 2.0, "end": 3.0, "text": "   "},
        ]
        result = self.diarize(segments)
        self.assertEqual(len(result), 1)

    def test_three_speakers(self):
        segments = [
            {"start": 0.0, "end": 1.0, "text": "A"},
            {"start": 3.0, "end": 4.0, "text": "B"},
            {"start": 6.0, "end": 7.0, "text": "C"},
            {"start": 9.0, "end": 10.0, "text": "D"},
        ]
        result = self.diarize(segments, num_speakers=3)
        speakers = [r["speaker"] for r in result]
        self.assertEqual(speakers, ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_00"])


class TestMapWhisperToDiarization(unittest.TestCase):
    """Tests for the map_whisper_to_diarization function."""

    def setUp(self):
        from ai_server.diarization import map_whisper_to_diarization
        self.map_fn = map_whisper_to_diarization

    def test_empty_diarization(self):
        whisper = [{"start": 0, "end": 2, "text": "Hello"}]
        result = self.map_fn(whisper, [])
        self.assertEqual(result, [])

    def test_perfect_overlap(self):
        whisper = [{"start": 0.0, "end": 2.0, "text": "Hello"}]
        diarization = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"}]
        result = self.map_fn(whisper, diarization)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")

    def test_partial_overlap_selects_best(self):
        """Should assign the speaker with the most overlap."""
        whisper = [{"start": 1.0, "end": 5.0, "text": "Long utterance"}]
        diarization = [
            {"start": 0.0, "end": 2.0, "speaker": "SPEAKER_00"},  # 1s overlap
            {"start": 2.0, "end": 6.0, "speaker": "SPEAKER_01"},  # 3s overlap
        ]
        result = self.map_fn(whisper, diarization)
        self.assertEqual(result[0]["speaker"], "SPEAKER_01")

    def test_no_overlap_defaults_to_speaker_00(self):
        whisper = [{"start": 10.0, "end": 12.0, "text": "No overlap"}]
        diarization = [{"start": 0.0, "end": 2.0, "speaker": "SPEAKER_01"}]
        result = self.map_fn(whisper, diarization)
        self.assertEqual(result[0]["speaker"], "SPEAKER_00")

    def test_empty_text_filtered(self):
        whisper = [
            {"start": 0, "end": 2, "text": "Hello"},
            {"start": 2, "end": 4, "text": ""},
        ]
        diarization = [{"start": 0, "end": 4, "speaker": "SPEAKER_00"}]
        result = self.map_fn(whisper, diarization)
        self.assertEqual(len(result), 1)


class TestGetDiarizationPipeline(unittest.TestCase):
    """Tests for pipeline loading and fallback behavior."""

    @patch("ai_server.diarization.HF_TOKEN", "")
    def test_fallback_without_token(self):
        """Should fall back when HF_TOKEN is not set."""
        # Reset the cached pipeline
        import ai_server.diarization as mod
        mod._diarization_pipeline = None
        mod._diarization_mode = "not_loaded"

        result = mod.get_diarization_pipeline()
        self.assertEqual(result, "fallback")
        self.assertEqual(mod.get_mode(), "fallback")


if __name__ == "__main__":
    unittest.main()
