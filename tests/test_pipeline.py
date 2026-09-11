"""
Tests for ai_server.pipeline module.

Tests the unified processing pipeline including segment merging,
speaker identification flow, and fallback behavior.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestMergeConsecutiveSpeakers(unittest.TestCase):
    """Tests for the merge_consecutive_speakers function."""

    def setUp(self):
        from ai_server.pipeline import merge_consecutive_speakers
        self.merge = merge_consecutive_speakers

    def test_empty_list(self):
        self.assertEqual(self.merge([]), [])

    def test_single_segment(self):
        segs = [{"speakerIndex": 0, "speaker": "A", "text": "Hello", "start": 0, "end": 1, "confidence": 0.8}]
        result = self.merge(segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello")

    def test_merge_same_speaker(self):
        """Consecutive segments from same speaker should merge."""
        segs = [
            {"speakerIndex": 0, "speaker": "Alice", "text": "Hello", "start": 0, "end": 1, "confidence": 0.8},
            {"speakerIndex": 0, "speaker": "Alice", "text": "world", "start": 1, "end": 2, "confidence": 0.9},
        ]
        result = self.merge(segs)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["text"], "Hello world")
        self.assertEqual(result[0]["start"], 0)
        self.assertEqual(result[0]["end"], 2)
        self.assertEqual(result[0]["confidence"], 0.9)  # Higher confidence kept

    def test_no_merge_different_speakers(self):
        """Segments from different speakers should NOT merge."""
        segs = [
            {"speakerIndex": 0, "speaker": "Alice", "text": "Hello", "start": 0, "end": 1, "confidence": 0.8},
            {"speakerIndex": 1, "speaker": "Bob", "text": "Hi", "start": 1, "end": 2, "confidence": 0.7},
        ]
        result = self.merge(segs)
        self.assertEqual(len(result), 2)

    def test_merge_preserves_speaker_transitions(self):
        """A-A-B-B should become A-B."""
        segs = [
            {"speakerIndex": 0, "speaker": "A", "text": "one", "start": 0, "end": 1, "confidence": 0.8},
            {"speakerIndex": 0, "speaker": "A", "text": "two", "start": 1, "end": 2, "confidence": 0.8},
            {"speakerIndex": 1, "speaker": "B", "text": "three", "start": 2, "end": 3, "confidence": 0.7},
            {"speakerIndex": 1, "speaker": "B", "text": "four", "start": 3, "end": 4, "confidence": 0.7},
        ]
        result = self.merge(segs)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["text"], "one two")
        self.assertEqual(result[1]["text"], "three four")

    def test_does_not_mutate_original(self):
        """Merge should not mutate the original segments list."""
        segs = [
            {"speakerIndex": 0, "speaker": "A", "text": "one", "start": 0, "end": 1, "confidence": 0.8},
            {"speakerIndex": 0, "speaker": "A", "text": "two", "start": 1, "end": 2, "confidence": 0.8},
        ]
        original_text = segs[0]["text"]
        self.merge(segs)
        self.assertEqual(segs[0]["text"], original_text)


class MockMeetingSession:
    """Minimal mock for MeetingSession to test pipeline without full server."""

    def __init__(self, speaker_count=2):
        self.speaker_count = speaker_count
        self.diarization_to_index = {}
        self.next_speaker_index = 0
        self.manual_names = {}
        self.auto_labels = {}
        self.enrolled_profiles = []

    def get_or_assign_speaker_index(self, diarization_label):
        if diarization_label in self.diarization_to_index:
            return self.diarization_to_index[diarization_label]
        idx = self.next_speaker_index
        self.diarization_to_index[diarization_label] = idx
        self.auto_labels[idx] = f"Speaker {idx + 1}"
        self.next_speaker_index += 1
        return idx

    def get_speaker_name(self, idx):
        if idx in self.manual_names:
            return self.manual_names[idx]
        return self.auto_labels.get(idx, f"Speaker {idx + 1}")


class TestPipelineFallbackBehavior(unittest.TestCase):
    """Tests that the pipeline falls back correctly when modules are unavailable."""

    @patch("ai_server.pipeline.get_diarization_mode", return_value="fallback")
    @patch("ai_server.pipeline.transcribe_audio")
    @patch("ai_server.pipeline.get_all_embeddings", return_value=[])
    def test_fallback_diarization_used(self, mock_emb, mock_whisper, mock_mode):
        """When pyannote is not available, should use simple fallback."""
        mock_whisper.return_value = [
            {"start": 0.0, "end": 2.0, "text": "Hello"},
            {"start": 4.0, "end": 6.0, "text": "World"},
        ]

        from ai_server.pipeline import process_audio_chunk
        session = MockMeetingSession(speaker_count=2)
        result = process_audio_chunk("fake.wav", session, [])

        self.assertEqual(len(result), 2)
        # First segment should be Speaker 1
        self.assertIn("speaker", result[0])
        self.assertIn("text", result[0])

    @patch("ai_server.pipeline.get_diarization_mode", return_value="fallback")
    @patch("ai_server.pipeline.transcribe_audio")
    @patch("ai_server.pipeline.get_all_embeddings", return_value=[])
    def test_no_speech_returns_empty(self, mock_emb, mock_whisper, mock_mode):
        """When Whisper returns no segments, pipeline should return empty."""
        mock_whisper.return_value = []

        from ai_server.pipeline import process_audio_chunk
        session = MockMeetingSession()
        result = process_audio_chunk("silence.wav", session, [])

        self.assertEqual(result, [])


class TestPipelineWithRecognition(unittest.TestCase):
    """Tests for pipeline with enrolled profiles (speaker recognition)."""

    @patch("ai_server.pipeline.get_diarization_mode", return_value="fallback")
    @patch("ai_server.pipeline.transcribe_audio")
    @patch("ai_server.pipeline.is_recognition_loaded", return_value=False)
    @patch("ai_server.pipeline.get_all_embeddings")
    def test_unknown_speaker_when_no_recognition(self, mock_emb, mock_recog, mock_whisper, mock_mode):
        """When recognition is not loaded, speakers should get default labels."""
        mock_whisper.return_value = [
            {"start": 0.0, "end": 2.0, "text": "Hello there"},
        ]

        from ai_server.pipeline import process_audio_chunk
        session = MockMeetingSession()
        result = process_audio_chunk("fake.wav", session, [])

        self.assertEqual(len(result), 1)
        # Should have a speaker name (even if it's the default)
        self.assertIn("speaker", result[0])
        self.assertIsNotNone(result[0]["speaker"])


if __name__ == "__main__":
    unittest.main()
