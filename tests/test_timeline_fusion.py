"""
Tests for Feature #3: Multi-Modal Timeline Fusion
===================================================
Covers: event model, Whisper chunk normalization, OCR timestamp
correction, note integration, and full timeline fusion.
"""

import time
import pytest
import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from timeline_fusion import (
    TimelineEvent,
    create_event,
    WhisperTimestampNormalizer,
    MeetingTimeline,
    correct_ocr_timestamp,
    correct_whisper_chunk_timestamp,
    MAX_CORRECTION_SECONDS,
    VALID_EVENT_TYPES,
)


# ══════════════════════════════════════════════════════
#  EVENT MODEL TESTS
# ══════════════════════════════════════════════════════

class TestEventModel:
    def test_create_transcript_event(self):
        e = create_event("transcript", "whisper", "Hello everyone")
        assert e.type == "transcript"
        assert e.source == "whisper"
        assert e.content == "Hello everyone"
        assert e.id  # non-empty
        assert e.timestamp > 0
        assert e.raw_timestamp == e.timestamp
        assert e.timestamp_correction == 0.0

    def test_create_note_event(self):
        e = create_event("note", "user", "Action item: follow up", timestamp=1000.0)
        assert e.type == "note"
        assert e.timestamp == 1000.0

    def test_create_screen_event(self):
        e = create_event("screen", "screen_capture", "screenshot_data")
        assert e.type == "screen"

    def test_create_ocr_event(self):
        e = create_event("ocr", "ocr_engine", "Detected text from slide")
        assert e.type == "ocr"

    def test_invalid_event_type_raises(self):
        with pytest.raises(ValueError, match="Invalid event type"):
            create_event("invalid_type", "whisper", "text")

    def test_event_to_dict(self):
        e = create_event("transcript", "whisper", "Hello", timestamp=5000.0,
                         metadata={"speaker": "Alice", "chunkIndex": 0})
        d = e.to_dict()
        assert d["type"] == "transcript"
        assert d["content"] == "Hello"
        assert d["metadata"]["speaker"] == "Alice"
        assert d["timestamp"] == 5000.0
        # None values should be excluded
        assert "corrected_timestamp" not in d

    def test_event_preserves_custom_id(self):
        e = create_event("note", "user", "test", event_id="custom-123")
        assert e.id == "custom-123"

    def test_raw_timestamp_defaults_to_timestamp(self):
        e = create_event("note", "user", "test", timestamp=42000.0)
        assert e.raw_timestamp == 42000.0

    def test_all_valid_types_accepted(self):
        for t in VALID_EVENT_TYPES:
            e = create_event(t, "user", "content")
            assert e.type == t


# ══════════════════════════════════════════════════════
#  WHISPER CHUNK TIMESTAMP NORMALIZATION TESTS
# ══════════════════════════════════════════════════════

class TestWhisperTimestampNormalizer:
    def test_first_chunk_no_offset(self):
        """First chunk timestamps should be meeting_start + chunk_relative."""
        n = WhisperTimestampNormalizer(meeting_start_ms=1000000.0)
        n.register_chunk(0, 8.0)
        result = n.normalize(0, 2.4)
        assert result == 1000000.0 + (2.4 * 1000)  # 1002400.0

    def test_second_chunk_offset_by_first_duration(self):
        """Second chunk's 0.0s should map to meeting_start + 8.0s."""
        n = WhisperTimestampNormalizer(meeting_start_ms=1000000.0)
        n.register_chunk(0, 8.0)
        n.register_chunk(1, 8.0)
        result = n.normalize(1, 0.0)
        assert result == 1000000.0 + (8.0 * 1000)  # 1008000.0

    def test_second_chunk_does_not_reset_to_zero(self):
        """Critical: chunk 2 at 3.1s must NOT appear at meeting second 3.1."""
        n = WhisperTimestampNormalizer(meeting_start_ms=0.0)
        n.register_chunk(0, 8.0)
        n.register_chunk(1, 8.0)

        chunk0_event = n.normalize(0, 2.4)  # 2.4s into meeting
        chunk1_event = n.normalize(1, 3.1)  # 8.0 + 3.1 = 11.1s into meeting

        assert chunk0_event == 2400.0
        assert chunk1_event == 11100.0
        assert chunk1_event > chunk0_event

    def test_third_chunk_cumulative(self):
        """Third chunk should accumulate offsets from chunks 0 and 1."""
        n = WhisperTimestampNormalizer(meeting_start_ms=0.0)
        n.register_chunk(0, 8.0)
        n.register_chunk(1, 8.0)
        n.register_chunk(2, 8.0)

        result = n.normalize(2, 1.5)
        expected = (8.0 + 8.0 + 1.5) * 1000  # 17500.0
        assert result == expected

    def test_variable_chunk_durations(self):
        """Chunks can have different durations."""
        n = WhisperTimestampNormalizer(meeting_start_ms=0.0)
        n.register_chunk(0, 5.0)
        n.register_chunk(1, 10.0)
        n.register_chunk(2, 3.0)

        assert n.normalize(0, 2.0) == 2000.0
        assert n.normalize(1, 4.0) == 9000.0   # 5.0 + 4.0
        assert n.normalize(2, 1.0) == 16000.0  # 5.0 + 10.0 + 1.0

    def test_unregistered_chunk_defaults_to_zero_offset(self):
        """Unregistered chunk uses 0 offset (graceful degradation)."""
        n = WhisperTimestampNormalizer(meeting_start_ms=1000.0)
        result = n.normalize(99, 5.0)
        assert result == 1000.0 + 5000.0

    def test_get_chunk_offset(self):
        n = WhisperTimestampNormalizer(meeting_start_ms=0.0)
        n.register_chunk(0, 8.0)
        n.register_chunk(1, 8.0)
        assert n.get_chunk_offset(0) == 0.0
        assert n.get_chunk_offset(1) == 8.0

    def test_duplicate_chunk_registration_idempotent(self):
        """Registering the same chunk twice should not change offset."""
        n = WhisperTimestampNormalizer(meeting_start_ms=0.0)
        n.register_chunk(0, 8.0)
        n.register_chunk(0, 99.0)  # Should be ignored
        n.register_chunk(1, 8.0)
        assert n.get_chunk_offset(1) == 8.0  # Not 99.0


# ══════════════════════════════════════════════════════
#  OCR TIMESTAMP CORRECTION TESTS
# ══════════════════════════════════════════════════════

class TestOCRTimestampCorrection:
    def test_ocr_uses_capture_time_as_authoritative(self):
        """Screenshot at 10:32:15 with OCR at 10:32:20 → timeline = 10:32:15."""
        capture_ms = 1000000.0
        ocr_done_ms = 1005000.0  # 5 seconds later

        correction = correct_ocr_timestamp(capture_ms, ocr_done_ms)
        assert correction["timestamp"] == capture_ms
        assert correction["raw_timestamp"] == ocr_done_ms
        assert correction["corrected_timestamp"] == capture_ms
        assert correction["timestamp_correction"] == -5000.0
        assert "processing delay" in correction["correction_reason"].lower()
        assert correction["correction_confidence"] > 0

    def test_no_delay_no_correction_needed(self):
        """If capture and OCR are simultaneous, correction is zero."""
        capture_ms = 1000000.0
        correction = correct_ocr_timestamp(capture_ms, capture_ms)
        assert correction["timestamp"] == capture_ms
        assert correction["timestamp_correction"] == 0.0

    def test_bounded_correction_rejects_huge_delta(self):
        """If OCR delay exceeds MAX_CORRECTION_SECONDS, reject the correction."""
        capture_ms = 1000000.0
        ocr_done_ms = capture_ms + (MAX_CORRECTION_SECONDS * 1000) + 1000  # Over bound

        correction = correct_ocr_timestamp(capture_ms, ocr_done_ms)
        assert correction["timestamp"] == ocr_done_ms  # Falls back to OCR time
        assert correction["correction_confidence"] == 0.0
        assert "rejected" in correction["correction_reason"].lower()


# ══════════════════════════════════════════════════════
#  WHISPER CORRECTION METADATA TESTS
# ══════════════════════════════════════════════════════

class TestWhisperCorrectionMetadata:
    def test_chunk_0_near_zero_delta_no_correction(self):
        """First chunk with negligible delta produces empty correction."""
        correction = correct_whisper_chunk_timestamp(
            raw_chunk_relative_ms=2400.0,
            normalized_meeting_ms=2400.0,
            chunk_index=0,
        )
        assert correction == {}

    def test_chunk_1_produces_correction(self):
        """Later chunks produce correction metadata."""
        correction = correct_whisper_chunk_timestamp(
            raw_chunk_relative_ms=3100.0,  # 3.1s into chunk
            normalized_meeting_ms=11100.0,  # 8.0 + 3.1 = 11.1s into meeting
            chunk_index=1,
        )
        assert correction["timestamp"] == 11100.0
        assert correction["raw_timestamp"] == 3100.0
        assert correction["correction_confidence"] == 1.0
        assert "chunk 1" in correction["correction_reason"].lower()


# ══════════════════════════════════════════════════════
#  TIMELINE FUSION TESTS
# ══════════════════════════════════════════════════════

class TestMeetingTimeline:
    def test_chronological_ordering(self):
        """Events must be in chronological order regardless of insertion order."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        tl.add_event(create_event("note", "user", "Note B", timestamp=2000.0))
        tl.add_event(create_event("transcript", "whisper", "Speech A", timestamp=1000.0))
        tl.add_event(create_event("note", "user", "Note C", timestamp=3000.0))

        events = tl.get_events()
        assert len(events) == 3
        assert events[0].content == "Speech A"
        assert events[1].content == "Note B"
        assert events[2].content == "Note C"

    def test_simultaneous_timestamps_preserved(self):
        """Events at the same timestamp should all be present."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        tl.add_event(create_event("transcript", "whisper", "A", timestamp=1000.0, event_id="e1"))
        tl.add_event(create_event("note", "user", "B", timestamp=1000.0, event_id="e2"))

        events = tl.get_events()
        assert len(events) == 2

    def test_out_of_order_arrival(self):
        """Late-arriving events are inserted at correct position."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        tl.add_event(create_event("transcript", "whisper", "First", timestamp=1000.0))
        tl.add_event(create_event("transcript", "whisper", "Third", timestamp=3000.0))
        tl.add_event(create_event("note", "user", "Second (late)", timestamp=2000.0))

        events = tl.get_events()
        assert events[0].content == "First"
        assert events[1].content == "Second (late)"
        assert events[2].content == "Third"

    def test_duplicate_event_rejected(self):
        """Same event ID cannot be added twice."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        e = create_event("note", "user", "Unique", timestamp=1000.0, event_id="dup-1")
        assert tl.add_event(e) is True
        assert tl.add_event(e) is False
        assert len(tl.get_events()) == 1

    def test_missing_timestamp_defaults_to_now(self):
        """Event without explicit timestamp gets current time."""
        tl = MeetingTimeline()
        e = create_event("note", "user", "Auto-timestamped")
        tl.add_event(e)
        assert e.timestamp > 0

    def test_filter_by_event_type(self):
        tl = MeetingTimeline(meeting_start_ms=0.0)
        tl.add_event(create_event("transcript", "whisper", "T1", timestamp=1000.0))
        tl.add_event(create_event("note", "user", "N1", timestamp=2000.0))
        tl.add_event(create_event("transcript", "whisper", "T2", timestamp=3000.0))

        transcripts = tl.get_events(event_type="transcript")
        assert len(transcripts) == 2

        notes = tl.get_events(event_type="note")
        assert len(notes) == 1

    def test_filter_since_timestamp(self):
        tl = MeetingTimeline(meeting_start_ms=0.0)
        tl.add_event(create_event("transcript", "whisper", "Old", timestamp=1000.0))
        tl.add_event(create_event("note", "user", "New", timestamp=5000.0))

        events = tl.get_events(since_ms=3000.0)
        assert len(events) == 1
        assert events[0].content == "New"

    def test_event_count_by_type(self):
        tl = MeetingTimeline(meeting_start_ms=0.0)
        tl.add_event(create_event("transcript", "whisper", "T1", timestamp=1000.0))
        tl.add_event(create_event("transcript", "whisper", "T2", timestamp=2000.0))
        tl.add_event(create_event("note", "user", "N1", timestamp=3000.0))

        counts = tl.get_event_count()
        assert counts["transcript"] == 2
        assert counts["note"] == 1
        assert counts["total"] == 3

    def test_export_to_list(self):
        tl = MeetingTimeline(meeting_start_ms=0.0)
        tl.add_event(create_event("note", "user", "N1", timestamp=1000.0))
        exported = tl.to_list()
        assert isinstance(exported, list)
        assert len(exported) == 1
        assert exported[0]["content"] == "N1"

    def test_clear(self):
        tl = MeetingTimeline(meeting_start_ms=0.0)
        tl.add_event(create_event("note", "user", "N1", timestamp=1000.0))
        tl.clear()
        assert len(tl.get_events()) == 0


# ══════════════════════════════════════════════════════
#  FULL FUSION INTEGRATION TESTS
# ══════════════════════════════════════════════════════

class TestFusionIntegration:
    def test_transcript_note_ocr_together(self):
        """All three event types fuse into correct chronological order."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        # Transcript at 2s
        tl.add_event(create_event("transcript", "whisper", "Let's review the slides",
                                  timestamp=2000.0, metadata={"speaker": "Alice"}))

        # Screenshot at 5s, OCR at 10s (but capture is authoritative)
        ocr_correction = correct_ocr_timestamp(5000.0, 10000.0)
        screen_event = create_event("screen", "screen_capture", "[screenshot]", timestamp=5000.0)
        tl.add_event(screen_event)

        ocr_event = create_event("ocr", "ocr_engine", "Q3 Revenue: $2.4M",
                                 timestamp=ocr_correction["timestamp"],
                                 metadata={"processing_delay_ms": 5000})
        ocr_event.raw_timestamp = ocr_correction["raw_timestamp"]
        ocr_event.corrected_timestamp = ocr_correction["corrected_timestamp"]
        ocr_event.timestamp_correction = ocr_correction["timestamp_correction"]
        ocr_event.correction_reason = ocr_correction["correction_reason"]
        tl.add_event(ocr_event)

        # Note at 8s
        tl.add_event(create_event("note", "user", "Action: verify Q3 numbers",
                                  timestamp=8000.0))

        events = tl.get_events()
        assert len(events) == 4

        # Verify chronological order
        assert events[0].type == "transcript"
        assert events[1].type == "screen"
        assert events[2].type == "ocr"       # OCR corrected to capture time (5s)
        assert events[3].type == "note"

        # Verify OCR correction metadata preserved
        assert events[2].raw_timestamp == 10000.0
        assert events[2].corrected_timestamp == 5000.0
        assert events[2].timestamp_correction == -5000.0

    def test_whisper_normalization_in_fusion(self):
        """Whisper chunks normalize correctly in a full timeline."""
        tl = MeetingTimeline(meeting_start_ms=100000.0)
        normalizer = tl.normalizer

        # Chunk 0: 8 seconds of audio
        normalizer.register_chunk(0, 8.0)
        ts0 = normalizer.normalize(0, 2.4)
        tl.add_event(create_event("transcript", "whisper", "Hello everyone",
                                  timestamp=ts0, metadata={"chunkIndex": 0}))

        # Chunk 1: another 8 seconds
        normalizer.register_chunk(1, 8.0)
        ts1 = normalizer.normalize(1, 3.1)
        tl.add_event(create_event("transcript", "whisper", "Let's begin",
                                  timestamp=ts1, metadata={"chunkIndex": 1}))

        # User note between chunks
        tl.add_event(create_event("note", "user", "Important point", timestamp=105000.0))

        events = tl.get_events()
        assert len(events) == 3

        # Chunk 0 event: 100000 + 2400 = 102400
        assert events[0].timestamp == 102400.0
        assert events[0].type == "transcript"

        # Note at 105000
        assert events[1].timestamp == 105000.0
        assert events[1].type == "note"

        # Chunk 1 event: 100000 + (8.0 + 3.1) * 1000 = 111100
        assert events[2].timestamp == 111100.0
        assert events[2].type == "transcript"

    def test_delayed_note_does_not_corrupt_ordering(self):
        """A note that arrives late is inserted at its correct timestamp position."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        tl.add_event(create_event("transcript", "whisper", "First speech", timestamp=1000.0))
        tl.add_event(create_event("transcript", "whisper", "Later speech", timestamp=5000.0))

        # Note at 3s arrives after "Later speech" was already added
        tl.add_event(create_event("note", "user", "Typed note", timestamp=3000.0))

        events = tl.get_events()
        assert events[0].timestamp == 1000.0
        assert events[1].timestamp == 3000.0  # Note correctly positioned
        assert events[2].timestamp == 5000.0

    def test_existing_transcription_behavior_preserved(self):
        """Transcript events with speaker/chunk metadata work correctly."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        tl.add_event(create_event(
            "transcript", "whisper", "Good morning",
            timestamp=2000.0,
            metadata={"speaker": "Speaker 1", "speakerIndex": 0, "chunkIndex": 0,
                       "start": 0.0, "end": 2.0, "confidence": 0.95}
        ))

        events = tl.get_events()
        assert len(events) == 1
        assert events[0].metadata["speaker"] == "Speaker 1"
        assert events[0].metadata["speakerIndex"] == 0
        assert events[0].metadata["confidence"] == 0.95

    def test_deterministic_reconstruction(self):
        """Same events added in different order produce identical timeline."""
        events_data = [
            ("transcript", "whisper", "A", 1000.0, "e1"),
            ("note", "user", "B", 2000.0, "e2"),
            ("ocr", "ocr_engine", "C", 1500.0, "e3"),
            ("screen", "screen_capture", "D", 500.0, "e4"),
        ]

        tl1 = MeetingTimeline(meeting_start_ms=0.0)
        for t, s, c, ts, eid in events_data:
            tl1.add_event(create_event(t, s, c, timestamp=ts, event_id=eid))

        tl2 = MeetingTimeline(meeting_start_ms=0.0)
        for t, s, c, ts, eid in reversed(events_data):
            tl2.add_event(create_event(t, s, c, timestamp=ts, event_id=eid))

        list1 = tl1.to_list()
        list2 = tl2.to_list()
        assert len(list1) == len(list2) == 4
        for e1, e2 in zip(list1, list2):
            assert e1["id"] == e2["id"]
            assert e1["timestamp"] == e2["timestamp"]

    def test_correction_metadata_in_export(self):
        """Exported timeline preserves timestamp correction metadata."""
        tl = MeetingTimeline(meeting_start_ms=0.0)

        ocr_correction = correct_ocr_timestamp(5000.0, 8000.0)
        e = create_event("ocr", "ocr_engine", "Slide text", timestamp=ocr_correction["timestamp"])
        e.raw_timestamp = ocr_correction["raw_timestamp"]
        e.corrected_timestamp = ocr_correction["corrected_timestamp"]
        e.timestamp_correction = ocr_correction["timestamp_correction"]
        e.correction_reason = ocr_correction["correction_reason"]
        e.correction_confidence = ocr_correction["correction_confidence"]
        tl.add_event(e)

        exported = tl.to_list()
        assert len(exported) == 1
        assert exported[0]["raw_timestamp"] == 8000.0
        assert exported[0]["corrected_timestamp"] == 5000.0
        assert exported[0]["timestamp_correction"] == -3000.0
