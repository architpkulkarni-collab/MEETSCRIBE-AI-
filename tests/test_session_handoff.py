"""
Tests for Feature #7: Cross-Device Session Handoff Protocol
==============================================================
Verifies: snapshot/restore integrity, handoff state machine,
conflict/duplicate protection, failure safety, active-device
enforcement, timeline preservation, and performance.
"""

import time
import pytest
import sys
import os
import hashlib
import threading

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from session_handoff import (
    HandoffManager,
    SessionSnapshot,
    HandoffPhase,
    DeviceRole,
    restore_session_from_snapshot,
    HANDOFF_TIMEOUT_SECONDS,
)
from timeline_fusion import (
    MeetingTimeline,
    create_event,
    TimelineEvent,
    WhisperTimestampNormalizer,
)


# ══════════════════════════════════════════════════════
#  HELPER: Minimal MeetingSession for testing
# ══════════════════════════════════════════════════════

class MockMeetingSession:
    """Minimal MeetingSession stub matching the real class interface."""

    def __init__(self, session_id, speaker_count=2):
        self.session_id = session_id
        self.speaker_count = speaker_count
        self.chunks_processed = 0
        self.total_duration = 0.0
        self.meeting_start_ms = time.time() * 1000

        self.diarization_to_index = {}
        self.next_speaker_index = 0
        self.manual_names = {}
        self.auto_labels = {}
        self.transcript = []
        self.enrolled_profiles = []
        self.created_at = None  # not needed for handoff

        # Feature #3: Timeline
        self.timeline = MeetingTimeline(meeting_start_ms=self.meeting_start_ms)

    def get_speaker_name(self, index):
        if index in self.manual_names:
            return self.manual_names[index]
        return self.auto_labels.get(index, f"Speaker {index + 1}")

    def to_dict(self):
        speakers = {}
        for idx in range(self.next_speaker_index):
            speakers[str(idx)] = {
                "index": idx,
                "label": self.get_speaker_name(idx),
                "isManual": idx in self.manual_names,
            }
        return {
            "sessionId": self.session_id,
            "speakerCount": self.next_speaker_index,
            "speakers": speakers,
            "chunksProcessed": self.chunks_processed,
            "totalDuration": round(self.total_duration, 1),
        }


def _make_session_with_data(session_id="test-session"):
    """Create a session with realistic transcript/timeline data."""
    s = MockMeetingSession(session_id, speaker_count=2)
    s.next_speaker_index = 2
    s.auto_labels = {0: "Speaker 1", 1: "Speaker 2"}
    s.manual_names = {0: "Alice"}
    s.diarization_to_index = {"SPEAKER_00": 0, "SPEAKER_01": 1}
    s.chunks_processed = 3
    s.total_duration = 24.0

    # Add transcript segments
    s.transcript = [
        {"speaker": "Alice", "speakerIndex": 0, "text": "Hello everyone", "start": 0.0, "end": 2.1, "chunkIndex": 0},
        {"speaker": "Speaker 2", "speakerIndex": 1, "text": "Hi, let's begin", "start": 2.5, "end": 4.0, "chunkIndex": 0},
        {"speaker": "Alice", "speakerIndex": 0, "text": "Let me share the agenda", "start": 8.0, "end": 10.5, "chunkIndex": 1},
        {"speaker": "Speaker 2", "speakerIndex": 1, "text": "Sounds good", "start": 11.0, "end": 12.0, "chunkIndex": 1},
        {"speaker": "Alice", "speakerIndex": 0, "text": "Moving to next topic", "start": 16.0, "end": 18.5, "chunkIndex": 2},
    ]

    # Add timeline events
    start_ms = s.meeting_start_ms
    norm = s.timeline.normalizer
    norm.register_chunk(0, 8.0)
    norm.register_chunk(1, 8.0)
    norm.register_chunk(2, 8.0)

    for seg in s.transcript:
        evt = create_event(
            event_type="transcript",
            source="whisper",
            content=seg["text"],
            timestamp=start_ms + seg["start"] * 1000,
            metadata={"speaker": seg["speaker"], "speakerIndex": seg["speakerIndex"]},
        )
        s.timeline.add_event(evt)

    # Add a typed note
    note_evt = create_event("note", "user", "Action item: review slides",
                            timestamp=start_ms + 5000)
    s.timeline.add_event(note_evt)

    return s


# ══════════════════════════════════════════════════════
#  1. SUCCESSFUL HANDOFF (Device A → Device B)
# ══════════════════════════════════════════════════════

class TestSuccessfulHandoff:
    def test_full_handoff_flow(self):
        """Complete Device A → Device B handoff."""
        mgr = HandoffManager()
        session = _make_session_with_data("session-001")

        # Step 1: Device A creates pair code
        result = mgr.create_pair_code("session-001", "device-a")
        assert "pair_code" in result
        assert len(result["pair_code"]) == 6
        pair_code = result["pair_code"]

        # Step 2: Device B joins
        result = mgr.join_with_code(pair_code, "device-b")
        assert "handoff_token" in result
        assert result["session_id"] == "session-001"
        token = result["handoff_token"]

        # Step 3: Device A initiates handoff
        result = mgr.initiate_handoff("session-001", "device-a", session)
        assert "snapshot" in result
        snapshot_dict = result["snapshot"]

        # Step 4: Device B retrieves snapshot
        result = mgr.get_snapshot("session-001", token)
        assert "snapshot" in result

        # Step 5: Device B acknowledges
        snap = SessionSnapshot.from_dict(result["snapshot"])
        result = mgr.acknowledge_handoff("session-001", token, "device-b", snap.checksum)
        assert result.get("success") is True

        # Step 6: Device A commits
        result = mgr.commit_handoff("session-001", "device-a", session)
        assert result.get("success") is True
        assert result["active_device"] == DeviceRole.DEVICE_B_ACTIVE
        assert result["handoff_epoch"] == 1
        assert "handoff_duration_ms" in result


# ══════════════════════════════════════════════════════
#  2. TRANSCRIPT PRESERVED
# ══════════════════════════════════════════════════════

class TestTranscriptPreserved:
    def test_transcript_count_matches(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap = result["snapshot"]
        assert len(snap["transcript"]) == len(session.transcript)
        assert len(snap["transcript"]) == 5

    def test_transcript_content_matches(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap_transcript = result["snapshot"]["transcript"]
        for orig, snapped in zip(session.transcript, snap_transcript):
            assert orig["text"] == snapped["text"]
            assert orig["speaker"] == snapped["speaker"]
            assert orig["speakerIndex"] == snapped["speakerIndex"]


# ══════════════════════════════════════════════════════
#  3. NOTES PRESERVED
# ══════════════════════════════════════════════════════

class TestNotesPreserved:
    def test_note_events_in_snapshot(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        timeline_events = result["snapshot"]["timeline_events"]
        note_events = [e for e in timeline_events if e["type"] == "note"]
        assert len(note_events) == 1
        assert "Action item: review slides" in note_events[0]["content"]


# ══════════════════════════════════════════════════════
#  4. FEATURE #3 UNIFIED TIMELINE PRESERVED
# ══════════════════════════════════════════════════════

class TestTimelinePreserved:
    def test_timeline_event_count(self):
        session = _make_session_with_data()
        original_count = session.timeline.get_event_count()

        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap = result["snapshot"]

        # 5 transcript + 1 note = 6 events
        assert len(snap["timeline_events"]) == original_count["total"]

    def test_timeline_restored_correctly(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap_dict = result["snapshot"]

        # Restore
        restored = restore_session_from_snapshot(snap_dict, MockMeetingSession)

        # Verify timeline events match
        orig_events = session.timeline.get_events()
        rest_events = restored.timeline.get_events()
        assert len(orig_events) == len(rest_events)
        for o, r in zip(orig_events, rest_events):
            assert o.id == r.id
            assert o.content == r.content
            assert o.type == r.type
            assert o.timestamp == r.timestamp

    def test_normalizer_state_preserved(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap_dict = result["snapshot"]

        restored = restore_session_from_snapshot(snap_dict, MockMeetingSession)
        orig_norm = session.timeline.normalizer
        rest_norm = restored.timeline.normalizer

        for ci in [0, 1, 2]:
            assert orig_norm.get_chunk_offset(ci) == rest_norm.get_chunk_offset(ci)


# ══════════════════════════════════════════════════════
#  5. NEW EVENTS AFTER HANDOFF CONTINUE CORRECTLY
# ══════════════════════════════════════════════════════

class TestPostHandoffContinuation:
    def test_new_transcript_after_restore(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap_dict = result["snapshot"]

        restored = restore_session_from_snapshot(snap_dict, MockMeetingSession)

        # Simulate Device B adding new transcript
        new_seg = {"speaker": "Alice", "speakerIndex": 0, "text": "Continuing on Device B",
                   "start": 24.0, "end": 26.0, "chunkIndex": 3}
        restored.transcript.append(new_seg)

        # Add to timeline
        new_evt = create_event("transcript", "whisper", "Continuing on Device B",
                               timestamp=restored.meeting_start_ms + 24000)
        restored.timeline.add_event(new_evt)

        assert len(restored.transcript) == 6  # 5 original + 1 new
        assert restored.timeline.get_event_count()["total"] == 7  # 6 original + 1 new
        assert restored.session_id == session.session_id  # Same session ID


# ══════════════════════════════════════════════════════
#  6. NO DUPLICATE EVENTS
# ══════════════════════════════════════════════════════

class TestNoDuplicateEvents:
    def test_timeline_deduplication(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        snap_dict = result["snapshot"]

        restored = restore_session_from_snapshot(snap_dict, MockMeetingSession)

        # Try to re-add an existing event (should be rejected by MeetingTimeline)
        existing_events = session.timeline.get_events()
        if existing_events:
            duplicate = create_event(
                event_type=existing_events[0].type,
                source=existing_events[0].source,
                content=existing_events[0].content,
                timestamp=existing_events[0].timestamp,
                event_id=existing_events[0].id,  # Same ID
            )
            added = restored.timeline.add_event(duplicate)
            assert added is False  # Duplicate rejected

    def test_snapshot_event_ids_unique(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        mgr.join_with_code(
            mgr.get_state(session.session_id).pair_code, "dev-b"
        )
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)
        event_ids = [e["id"] for e in result["snapshot"]["timeline_events"]]
        assert len(event_ids) == len(set(event_ids))  # All unique


# ══════════════════════════════════════════════════════
#  7. OUT-OF-ORDER MESSAGES
# ══════════════════════════════════════════════════════

class TestOutOfOrderMessages:
    def test_ack_before_snapshot_sent_fails(self):
        mgr = HandoffManager()
        session = _make_session_with_data("ooo-session")
        mgr.create_pair_code("ooo-session", "dev-a")
        state = mgr.get_state("ooo-session")
        mgr.join_with_code(state.pair_code, "dev-b")

        # Try to ACK before snapshot is created
        result = mgr.acknowledge_handoff("ooo-session", state.handoff_token,
                                          "dev-b", "fake-checksum")
        assert "error" in result

    def test_commit_before_ack_fails(self):
        mgr = HandoffManager()
        session = _make_session_with_data("ooo-session2")
        mgr.create_pair_code("ooo-session2", "dev-a")
        state = mgr.get_state("ooo-session2")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("ooo-session2", "dev-a", session)

        # Try to commit before ACK
        result = mgr.commit_handoff("ooo-session2", "dev-a", session)
        assert "error" in result


# ══════════════════════════════════════════════════════
#  8. DUPLICATE MESSAGES
# ══════════════════════════════════════════════════════

class TestDuplicateMessages:
    def test_duplicate_pair_code_request(self):
        mgr = HandoffManager()
        session = _make_session_with_data("dup-session")
        r1 = mgr.create_pair_code("dup-session", "dev-a")
        assert "pair_code" in r1

        # Second pair request while first is active → error
        r2 = mgr.create_pair_code("dup-session", "dev-a")
        assert "error" in r2

    def test_duplicate_join_attempt(self):
        mgr = HandoffManager()
        session = _make_session_with_data("dup-session2")
        r1 = mgr.create_pair_code("dup-session2", "dev-a")
        code = r1["pair_code"]

        # First join succeeds
        r2 = mgr.join_with_code(code, "dev-b")
        assert "handoff_token" in r2

        # Second join with same code fails (already paired)
        r3 = mgr.join_with_code(code, "dev-c")
        assert "error" in r3


# ══════════════════════════════════════════════════════
#  9. MISSING UPDATE DETECTION
# ══════════════════════════════════════════════════════

class TestMissingUpdateDetection:
    def test_wrong_checksum_rejected(self):
        mgr = HandoffManager()
        session = _make_session_with_data("missing-session")
        mgr.create_pair_code("missing-session", "dev-a")
        state = mgr.get_state("missing-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("missing-session", "dev-a", session)
        mgr.get_snapshot("missing-session", state.handoff_token)

        # ACK with wrong checksum → rejected
        result = mgr.acknowledge_handoff("missing-session", state.handoff_token,
                                          "dev-b", "wrong-checksum")
        assert "error" in result
        assert "checksum" in result["error"].lower() or "Checksum" in result["error"]

    def test_snapshot_checksum_integrity(self):
        session = _make_session_with_data()
        snap = SessionSnapshot(
            session_id="test", sequence_number=5,
            transcript=[{"text": "hello"}],
            timeline_events=[{"id": "1", "type": "note"}],
            chunks_processed=2, handoff_epoch=0,
        )
        snap.seal()
        assert snap.verify()

        # Tamper with data
        snap.sequence_number = 999
        assert not snap.verify()


# ══════════════════════════════════════════════════════
# 10. DEVICE B ACK REQUIRED BEFORE COMMIT
# ══════════════════════════════════════════════════════

class TestAckRequiredBeforeCommit:
    def test_commit_without_ack_fails(self):
        mgr = HandoffManager()
        session = _make_session_with_data("ack-session")
        mgr.create_pair_code("ack-session", "dev-a")
        state = mgr.get_state("ack-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("ack-session", "dev-a", session)

        # Skip ACK → commit should fail
        result = mgr.commit_handoff("ack-session", "dev-a", session)
        assert "error" in result
        assert "Cannot commit" in result["error"]


# ══════════════════════════════════════════════════════
# 11. FAILURE LEAVES DEVICE A SAFE
# ══════════════════════════════════════════════════════

class TestFailureSafety:
    def test_abort_preserves_device_a(self):
        mgr = HandoffManager()
        session = _make_session_with_data("fail-session")
        mgr.create_pair_code("fail-session", "dev-a")
        state = mgr.get_state("fail-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("fail-session", "dev-a", session)

        # Abort handoff
        result = mgr.abort_handoff("fail-session", "Network error")
        assert result.get("success") is True
        assert result["active_device"] == DeviceRole.DEVICE_A_ACTIVE

        # Session data should be intact
        assert len(session.transcript) == 5
        assert session.timeline.get_event_count()["total"] == 6

    def test_timeout_preserves_device_a(self):
        mgr = HandoffManager()
        session = _make_session_with_data("timeout-session")
        mgr.create_pair_code("timeout-session", "dev-a")
        state = mgr.get_state("timeout-session")

        # Force expiry
        state.expires_at = time.time() - 1

        result = mgr.join_with_code(state.pair_code, "dev-b")
        assert "error" in result

        # Device A should remain usable
        status = mgr.get_handoff_status("timeout-session")
        assert status["active_device"] == DeviceRole.DEVICE_A_ACTIVE

    def test_invalid_token_rejected(self):
        mgr = HandoffManager()
        session = _make_session_with_data("token-session")
        mgr.create_pair_code("token-session", "dev-a")
        state = mgr.get_state("token-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("token-session", "dev-a", session)

        # Try snapshot retrieval with wrong token
        result = mgr.get_snapshot("token-session", "wrong-token-here")
        assert "error" in result


# ══════════════════════════════════════════════════════
# 12. RECONNECTION
# ══════════════════════════════════════════════════════

class TestReconnection:
    def test_new_pair_after_failure(self):
        mgr = HandoffManager()
        session = _make_session_with_data("recon-session")

        # First attempt fails
        mgr.create_pair_code("recon-session", "dev-a")
        mgr.abort_handoff("recon-session", "First attempt failed")

        # Can create new pair code after failure
        result = mgr.create_pair_code("recon-session", "dev-a")
        assert "pair_code" in result


# ══════════════════════════════════════════════════════
# 13. REPEATED HANDOFF ATTEMPT
# ══════════════════════════════════════════════════════

class TestRepeatedHandoffAttempt:
    def test_cannot_initiate_twice(self):
        mgr = HandoffManager()
        session = _make_session_with_data("repeat-session")
        mgr.create_pair_code("repeat-session", "dev-a")
        state = mgr.get_state("repeat-session")
        mgr.join_with_code(state.pair_code, "dev-b")

        # First initiation
        r1 = mgr.initiate_handoff("repeat-session", "dev-a", session)
        assert "snapshot" in r1

        # Second initiation in wrong phase
        r2 = mgr.initiate_handoff("repeat-session", "dev-a", session)
        assert "error" in r2


# ══════════════════════════════════════════════════════
# 14. STALE DEVICE A CANNOT OVERWRITE AFTER HANDOFF
# ══════════════════════════════════════════════════════

class TestStaleDeviceProtection:
    def test_stale_device_a_blocked(self):
        mgr = HandoffManager()
        session = _make_session_with_data("stale-session")

        # Complete full handoff
        mgr.create_pair_code("stale-session", "dev-a")
        state = mgr.get_state("stale-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("stale-session", "dev-a", session)
        mgr.get_snapshot("stale-session", state.handoff_token)
        snap = state.snapshot
        mgr.acknowledge_handoff("stale-session", state.handoff_token,
                                "dev-b", snap.checksum)
        mgr.commit_handoff("stale-session", "dev-a", session)

        # Device A is no longer active
        assert not mgr.validate_active_device("stale-session", "dev-a")
        # Device B is active
        assert mgr.validate_active_device("stale-session", "dev-b")

    def test_handoff_epoch_increments(self):
        mgr = HandoffManager()
        session = _make_session_with_data("epoch-session")

        # Complete handoff
        mgr.create_pair_code("epoch-session", "dev-a")
        state = mgr.get_state("epoch-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("epoch-session", "dev-a", session)
        mgr.get_snapshot("epoch-session", state.handoff_token)
        snap = state.snapshot
        mgr.acknowledge_handoff("epoch-session", state.handoff_token,
                                "dev-b", snap.checksum)
        result = mgr.commit_handoff("epoch-session", "dev-a", session)

        assert result["handoff_epoch"] == 1
        assert mgr.get_active_epoch("epoch-session") == 1


# ══════════════════════════════════════════════════════
# 15. SESSION ID REMAINS UNCHANGED
# ══════════════════════════════════════════════════════

class TestSessionIdUnchanged:
    def test_session_id_preserved_after_restore(self):
        session = _make_session_with_data("preserve-id")
        mgr = HandoffManager()
        mgr.create_pair_code("preserve-id", "dev-a")
        state = mgr.get_state("preserve-id")
        mgr.join_with_code(state.pair_code, "dev-b")
        result = mgr.initiate_handoff("preserve-id", "dev-a", session)
        snap_dict = result["snapshot"]

        restored = restore_session_from_snapshot(snap_dict, MockMeetingSession)
        assert restored.session_id == "preserve-id"
        assert restored.session_id == session.session_id


# ══════════════════════════════════════════════════════
# 16. HANDOFF TIMING
# ══════════════════════════════════════════════════════

class TestHandoffTiming:
    def test_handoff_completes_under_2_seconds(self):
        """Measure handoff duration — target ≤ 2 seconds."""
        mgr = HandoffManager()
        session = _make_session_with_data("perf-session")

        start = time.time()

        mgr.create_pair_code("perf-session", "dev-a")
        state = mgr.get_state("perf-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("perf-session", "dev-a", session)
        mgr.get_snapshot("perf-session", state.handoff_token)
        snap = state.snapshot
        mgr.acknowledge_handoff("perf-session", state.handoff_token,
                                "dev-b", snap.checksum)
        result = mgr.commit_handoff("perf-session", "dev-a", session)

        elapsed_ms = (time.time() - start) * 1000

        assert result.get("success") is True
        # Protocol-level handoff should be well under 2 seconds
        assert elapsed_ms < 2000, f"Handoff took {elapsed_ms:.0f}ms (target: <2000ms)"
        print(f"  Handoff timing: {elapsed_ms:.1f}ms")

    def test_snapshot_includes_timing(self):
        mgr = HandoffManager()
        session = _make_session_with_data("timing-session")
        mgr.create_pair_code("timing-session", "dev-a")
        state = mgr.get_state("timing-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("timing-session", "dev-a", session)
        mgr.get_snapshot("timing-session", state.handoff_token)
        snap = state.snapshot
        mgr.acknowledge_handoff("timing-session", state.handoff_token,
                                "dev-b", snap.checksum)
        result = mgr.commit_handoff("timing-session", "dev-a", session)

        assert "handoff_duration_ms" in result
        assert result["handoff_duration_ms"] >= 0


# ══════════════════════════════════════════════════════
# 17. EXISTING FUNCTIONALITY NON-REGRESSION
# ══════════════════════════════════════════════════════

class TestNonRegression:
    def test_session_without_handoff_works(self):
        """Verify normal session operations work without any handoff."""
        session = MockMeetingSession("normal-session", speaker_count=2)
        session.auto_labels[0] = "Speaker 1"
        session.next_speaker_index = 1
        session.transcript.append({
            "speaker": "Speaker 1", "speakerIndex": 0,
            "text": "Normal meeting", "start": 0, "end": 2,
        })

        # Timeline should work normally
        evt = create_event("transcript", "whisper", "Normal meeting",
                           timestamp=time.time() * 1000)
        added = session.timeline.add_event(evt)
        assert added is True
        assert session.timeline.get_event_count()["total"] == 1

    def test_handoff_manager_no_state_returns_defaults(self):
        mgr = HandoffManager()
        status = mgr.get_handoff_status("nonexistent")
        assert status["phase"] == HandoffPhase.IDLE
        assert status["active_device"] == DeviceRole.DEVICE_A_ACTIVE

    def test_validate_active_device_no_handoff(self):
        mgr = HandoffManager()
        # Without handoff, any device is considered active
        assert mgr.validate_active_device("any-session", "any-device") is True

    def test_restored_session_speaker_names_preserved(self):
        session = _make_session_with_data()
        mgr = HandoffManager()
        mgr.create_pair_code(session.session_id, "dev-a")
        state = mgr.get_state(session.session_id)
        mgr.join_with_code(state.pair_code, "dev-b")
        result = mgr.initiate_handoff(session.session_id, "dev-a", session)

        restored = restore_session_from_snapshot(result["snapshot"], MockMeetingSession)

        # Manual name should be preserved
        assert restored.manual_names[0] == "Alice"
        assert restored.auto_labels[0] == "Speaker 1"
        assert restored.auto_labels[1] == "Speaker 2"


# ══════════════════════════════════════════════════════
#  ADDITIONAL EDGE CASES
# ══════════════════════════════════════════════════════

class TestEdgeCases:
    def test_empty_session_handoff(self):
        """Handoff of a session with no transcript/events."""
        session = MockMeetingSession("empty-session")
        mgr = HandoffManager()
        mgr.create_pair_code("empty-session", "dev-a")
        state = mgr.get_state("empty-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        result = mgr.initiate_handoff("empty-session", "dev-a", session)

        assert "snapshot" in result
        assert len(result["snapshot"]["transcript"]) == 0
        assert len(result["snapshot"]["timeline_events"]) == 0

        # Restore should work
        restored = restore_session_from_snapshot(result["snapshot"], MockMeetingSession)
        assert restored.session_id == "empty-session"
        assert len(restored.transcript) == 0

    def test_cannot_abort_committed_handoff(self):
        mgr = HandoffManager()
        session = _make_session_with_data("committed-session")

        # Complete handoff
        mgr.create_pair_code("committed-session", "dev-a")
        state = mgr.get_state("committed-session")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("committed-session", "dev-a", session)
        mgr.get_snapshot("committed-session", state.handoff_token)
        snap = state.snapshot
        mgr.acknowledge_handoff("committed-session", state.handoff_token,
                                "dev-b", snap.checksum)
        mgr.commit_handoff("committed-session", "dev-a", session)

        # Abort should fail
        result = mgr.abort_handoff("committed-session")
        assert "error" in result

    def test_wrong_device_a_cannot_initiate(self):
        mgr = HandoffManager()
        session = _make_session_with_data("wrong-dev")
        mgr.create_pair_code("wrong-dev", "dev-a")
        state = mgr.get_state("wrong-dev")
        mgr.join_with_code(state.pair_code, "dev-b")

        # Wrong device tries to initiate
        result = mgr.initiate_handoff("wrong-dev", "dev-c", session)
        assert "error" in result

    def test_wrong_device_b_ack_rejected(self):
        mgr = HandoffManager()
        session = _make_session_with_data("wrong-ack")
        mgr.create_pair_code("wrong-ack", "dev-a")
        state = mgr.get_state("wrong-ack")
        mgr.join_with_code(state.pair_code, "dev-b")
        mgr.initiate_handoff("wrong-ack", "dev-a", session)
        mgr.get_snapshot("wrong-ack", state.handoff_token)

        # Wrong device tries to ACK
        result = mgr.acknowledge_handoff("wrong-ack", state.handoff_token,
                                          "dev-c", state.snapshot.checksum)
        assert "error" in result

    def test_thread_safety(self):
        """Concurrent operations should not crash."""
        mgr = HandoffManager()
        errors = []

        def create_and_abort(i):
            try:
                sid = f"thread-{i}"
                session = MockMeetingSession(sid)
                mgr.create_pair_code(sid, f"dev-a-{i}")
                mgr.abort_handoff(sid)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_and_abort, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
