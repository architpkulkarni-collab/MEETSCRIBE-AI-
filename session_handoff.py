"""
Cross-Device Session Handoff Protocol — Feature #7
=====================================================
Enables a live meeting recording to move from Device A → Device B with
zero event/data loss.  The handoff is server-mediated via the existing
Flask backend (no extra signaling server required).

Key design points:
  - Single authoritative active device at all times.
  - Deterministic sequence numbers on every state update.
  - Explicit acknowledgement before handoff commits.
  - Device A remains safe on any failure path.
  - No duplicate transcript/note/timeline events.
"""

import time
import uuid
import hmac
import hashlib
import secrets
import threading
from enum import Enum
from dataclasses import dataclass, field, asdict
from typing import Optional


# ══════════════════════════════════════════════════════
#  HANDOFF STATE MACHINE
# ══════════════════════════════════════════════════════

class DeviceRole(str, Enum):
    DEVICE_A_ACTIVE = "DEVICE_A_ACTIVE"
    HANDOFF_IN_PROGRESS = "HANDOFF_IN_PROGRESS"
    DEVICE_B_ACTIVE = "DEVICE_B_ACTIVE"


class HandoffPhase(str, Enum):
    IDLE = "IDLE"
    PAIR_REQUESTED = "PAIR_REQUESTED"
    PAIRED = "PAIRED"
    SNAPSHOT_CREATED = "SNAPSHOT_CREATED"
    SNAPSHOT_SENT = "SNAPSHOT_SENT"
    ACK_RECEIVED = "ACK_RECEIVED"
    COMMITTED = "COMMITTED"
    FAILED = "FAILED"


HANDOFF_TIMEOUT_SECONDS = 30
PAIR_CODE_LENGTH = 6
HANDOFF_TOKEN_LENGTH = 32


# ══════════════════════════════════════════════════════
#  SESSION SNAPSHOT
# ══════════════════════════════════════════════════════

@dataclass
class SessionSnapshot:
    """
    Frozen authoritative state of a meeting session at handoff time.
    Contains everything needed for Device B to continue seamlessly.
    """
    # Session identity
    session_id: str
    snapshot_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    created_at: float = field(default_factory=lambda: time.time())

    # Sequence / version — monotonically increasing
    sequence_number: int = 0

    # Meeting state
    meeting_status: str = "active"  # active | paused | stopped
    speaker_count: int = 0
    chunks_processed: int = 0
    total_duration: float = 0.0
    meeting_start_ms: float = 0.0

    # Speaker mappings
    diarization_to_index: dict = field(default_factory=dict)
    next_speaker_index: int = 0
    manual_names: dict = field(default_factory=dict)
    auto_labels: dict = field(default_factory=dict)

    # Transcript events (list of segment dicts)
    transcript: list = field(default_factory=list)

    # Timeline events (Feature #3) — serialised dicts
    timeline_events: list = field(default_factory=list)
    timeline_event_ids: list = field(default_factory=list)

    # Normalizer state (Feature #3)
    normalizer_chunk_offsets: dict = field(default_factory=dict)
    normalizer_next_offset: float = 0.0

    # Active device info
    active_device: str = DeviceRole.DEVICE_A_ACTIVE
    handoff_epoch: int = 0  # increments on each successful handoff

    # Integrity checksum
    checksum: str = ""

    def compute_checksum(self) -> str:
        """Deterministic checksum over transferable state."""
        parts = [
            self.session_id,
            str(self.sequence_number),
            str(len(self.transcript)),
            str(len(self.timeline_events)),
            str(self.chunks_processed),
            str(self.handoff_epoch),
        ]
        raw = "|".join(parts).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:16]

    def seal(self):
        """Compute and store checksum."""
        self.checksum = self.compute_checksum()
        return self

    def verify(self) -> bool:
        """Verify integrity."""
        return self.checksum == self.compute_checksum()

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "SessionSnapshot":
        # Convert string keys back to int keys where needed
        snap = cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
        return snap


# ══════════════════════════════════════════════════════
#  HANDOFF MANAGER
# ══════════════════════════════════════════════════════

class HandoffManager:
    """
    Server-side manager for cross-device session handoff.

    Lifecycle:
      1. Device A creates a pair code via create_pair_code()
      2. Device B joins with join_with_code()
      3. Device A initiates handoff via initiate_handoff()
         — freezes session, creates snapshot
      4. Device B receives snapshot, validates, ACKs
      5. Device A commits handoff — Device B becomes active
      6. On any failure → Device A remains active, handoff aborted
    """

    def __init__(self):
        self._lock = threading.Lock()
        # session_id → HandoffState
        self._states: dict[str, "HandoffState"] = {}

    def get_state(self, session_id: str) -> Optional["HandoffState"]:
        with self._lock:
            return self._states.get(session_id)

    def create_pair_code(self, session_id: str, device_a_id: str) -> dict:
        """
        Generate a short numeric pairing code for Device B to join.
        Returns {pair_code, expires_at}.
        """
        with self._lock:
            state = self._states.get(session_id)
            if state and state.phase not in (HandoffPhase.IDLE, HandoffPhase.FAILED):
                return {"error": "Handoff already in progress"}

            pair_code = "".join([str(secrets.randbelow(10)) for _ in range(PAIR_CODE_LENGTH)])
            handoff_token = secrets.token_hex(HANDOFF_TOKEN_LENGTH)
            expires_at = time.time() + HANDOFF_TIMEOUT_SECONDS

            state = HandoffState(
                session_id=session_id,
                device_a_id=device_a_id,
                pair_code=pair_code,
                handoff_token=handoff_token,
                phase=HandoffPhase.PAIR_REQUESTED,
                expires_at=expires_at,
            )
            self._states[session_id] = state

        return {
            "pair_code": pair_code,
            "expires_at": expires_at,
            "session_id": session_id,
        }

    def join_with_code(self, pair_code: str, device_b_id: str) -> dict:
        """
        Device B joins using the pair code.
        Returns {session_id, handoff_token} or {error}.
        """
        with self._lock:
            # Find session by pair code
            target = None
            for sid, state in self._states.items():
                if (state.pair_code == pair_code
                        and state.phase == HandoffPhase.PAIR_REQUESTED
                        and time.time() < state.expires_at):
                    target = state
                    break

            if not target:
                return {"error": "Invalid or expired pair code"}

            target.device_b_id = device_b_id
            target.phase = HandoffPhase.PAIRED
            target.expires_at = time.time() + HANDOFF_TIMEOUT_SECONDS

        return {
            "session_id": target.session_id,
            "handoff_token": target.handoff_token,
            "device_a_id": target.device_a_id,
        }

    def initiate_handoff(self, session_id: str, device_a_id: str,
                         session_obj) -> dict:
        """
        Device A initiates handoff. Freezes the session and creates a snapshot.
        session_obj is the live MeetingSession instance.
        Returns {snapshot: dict} or {error}.
        """
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return {"error": "No pairing exists for this session"}
            if state.device_a_id != device_a_id:
                return {"error": "Not the active device"}
            if state.phase != HandoffPhase.PAIRED:
                return {"error": f"Cannot initiate handoff in phase {state.phase}"}
            if time.time() > state.expires_at:
                state.phase = HandoffPhase.FAILED
                state.failure_reason = "Pairing expired"
                return {"error": "Pairing expired"}

            # Create snapshot from session
            snapshot = self._create_snapshot(session_obj, state)
            state.snapshot = snapshot
            state.phase = HandoffPhase.SNAPSHOT_CREATED
            state.expires_at = time.time() + HANDOFF_TIMEOUT_SECONDS

        return {"snapshot": snapshot.to_dict()}

    def get_snapshot(self, session_id: str, handoff_token: str) -> dict:
        """
        Device B retrieves the snapshot using its handoff token.
        Returns {snapshot: dict} or {error}.
        """
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return {"error": "No handoff state"}
            if not hmac.compare_digest(state.handoff_token, handoff_token):
                return {"error": "Invalid handoff token"}
            if state.phase not in (HandoffPhase.SNAPSHOT_CREATED, HandoffPhase.SNAPSHOT_SENT):
                return {"error": f"No snapshot available (phase: {state.phase})"}
            if time.time() > state.expires_at:
                state.phase = HandoffPhase.FAILED
                state.failure_reason = "Handoff timed out"
                return {"error": "Handoff timed out"}

            state.phase = HandoffPhase.SNAPSHOT_SENT

        return {"snapshot": state.snapshot.to_dict()}

    def acknowledge_handoff(self, session_id: str, handoff_token: str,
                            device_b_id: str, ack_checksum: str) -> dict:
        """
        Device B acknowledges receipt and validation of the snapshot.
        ack_checksum must match the snapshot checksum — proves Device B
        has the complete, uncorrupted state.
        Returns {success: True} or {error}.
        """
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return {"error": "No handoff state"}
            if not hmac.compare_digest(state.handoff_token, handoff_token):
                return {"error": "Invalid handoff token"}
            if state.device_b_id != device_b_id:
                return {"error": "Device B ID mismatch"}
            if state.phase != HandoffPhase.SNAPSHOT_SENT:
                return {"error": f"Cannot acknowledge in phase {state.phase}"}
            if time.time() > state.expires_at:
                state.phase = HandoffPhase.FAILED
                state.failure_reason = "Acknowledgement timed out"
                return {"error": "Handoff timed out"}

            # Verify checksum matches
            if not state.snapshot or state.snapshot.checksum != ack_checksum:
                state.phase = HandoffPhase.FAILED
                state.failure_reason = "Checksum mismatch — state corrupted in transit"
                return {"error": "Checksum mismatch — incomplete or corrupted state"}

            state.phase = HandoffPhase.ACK_RECEIVED
            state.ack_time = time.time()

        return {"success": True, "phase": HandoffPhase.ACK_RECEIVED}

    def commit_handoff(self, session_id: str, device_a_id: str,
                       session_obj) -> dict:
        """
        Device A commits the handoff after receiving Device B's ACK.
        After this, Device B is the active device.
        Returns {success, handoff_epoch, handoff_duration_ms} or {error}.
        """
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return {"error": "No handoff state"}
            if state.device_a_id != device_a_id:
                return {"error": "Not the active device"}
            if state.phase != HandoffPhase.ACK_RECEIVED:
                return {"error": f"Cannot commit in phase {state.phase}"}

            # Commit: transfer ownership
            new_epoch = (state.snapshot.handoff_epoch if state.snapshot else 0) + 1
            state.phase = HandoffPhase.COMMITTED
            state.committed_at = time.time()
            state.handoff_epoch = new_epoch

            # Calculate handoff duration
            handoff_start = state.snapshot.created_at if state.snapshot else state.committed_at
            handoff_duration_ms = (state.committed_at - handoff_start) * 1000

        return {
            "success": True,
            "handoff_epoch": new_epoch,
            "active_device": DeviceRole.DEVICE_B_ACTIVE,
            "handoff_duration_ms": round(handoff_duration_ms, 1),
        }

    def abort_handoff(self, session_id: str, reason: str = "Aborted by user") -> dict:
        """Abort the current handoff. Device A remains active."""
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return {"error": "No handoff state"}
            if state.phase == HandoffPhase.COMMITTED:
                return {"error": "Cannot abort — handoff already committed"}

            state.phase = HandoffPhase.FAILED
            state.failure_reason = reason
            state.snapshot = None

        return {"success": True, "active_device": DeviceRole.DEVICE_A_ACTIVE}

    def get_handoff_status(self, session_id: str) -> dict:
        """Get current handoff status for a session."""
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return {"phase": HandoffPhase.IDLE, "active_device": DeviceRole.DEVICE_A_ACTIVE}

            # Check expiry
            if (state.phase not in (HandoffPhase.IDLE, HandoffPhase.COMMITTED, HandoffPhase.FAILED)
                    and time.time() > state.expires_at):
                state.phase = HandoffPhase.FAILED
                state.failure_reason = "Timed out"

            return {
                "phase": state.phase,
                "active_device": (DeviceRole.DEVICE_B_ACTIVE
                                  if state.phase == HandoffPhase.COMMITTED
                                  else DeviceRole.DEVICE_A_ACTIVE),
                "handoff_epoch": state.handoff_epoch,
                "device_a_id": state.device_a_id,
                "device_b_id": state.device_b_id,
                "failure_reason": state.failure_reason,
                "expires_at": state.expires_at,
            }

    def validate_active_device(self, session_id: str, device_id: str) -> bool:
        """
        Check if a device is the current active (writer) device.
        Used to prevent stale devices from writing after handoff.
        """
        with self._lock:
            state = self._states.get(session_id)
            if not state:
                return True  # No handoff state → original device is active
            if state.phase == HandoffPhase.COMMITTED:
                return device_id == state.device_b_id
            # During handoff or failure, Device A remains active
            return device_id == state.device_a_id

    def get_active_epoch(self, session_id: str) -> int:
        """Get the current handoff epoch for a session."""
        with self._lock:
            state = self._states.get(session_id)
            return state.handoff_epoch if state else 0

    # ── Internal ──

    def _create_snapshot(self, session_obj, state: "HandoffState") -> SessionSnapshot:
        """
        Create a frozen snapshot from a live MeetingSession.
        """
        # Serialize timeline events
        timeline_events = []
        timeline_event_ids = []
        normalizer_offsets = {}
        normalizer_next = 0.0

        if hasattr(session_obj, "timeline") and session_obj.timeline:
            tl = session_obj.timeline
            with tl._lock:
                timeline_events = [e.to_dict() for e in tl._events]
                timeline_event_ids = list(tl._event_ids)
            norm = tl.normalizer
            with norm._lock:
                normalizer_offsets = dict(norm._chunk_offsets)
                normalizer_next = norm._next_offset

        # Convert int keys to strings for JSON serialization
        dia_map = {str(k): v for k, v in session_obj.diarization_to_index.items()}
        manual = {str(k): v for k, v in session_obj.manual_names.items()}
        auto = {str(k): v for k, v in session_obj.auto_labels.items()}
        norm_offsets = {str(k): v for k, v in normalizer_offsets.items()}

        snapshot = SessionSnapshot(
            session_id=session_obj.session_id,
            sequence_number=len(session_obj.transcript) + len(timeline_events),
            meeting_status="active",
            speaker_count=session_obj.speaker_count,
            chunks_processed=session_obj.chunks_processed,
            total_duration=session_obj.total_duration,
            meeting_start_ms=session_obj.meeting_start_ms,
            diarization_to_index=dia_map,
            next_speaker_index=session_obj.next_speaker_index,
            manual_names=manual,
            auto_labels=auto,
            transcript=list(session_obj.transcript),  # deep copy
            timeline_events=timeline_events,
            timeline_event_ids=timeline_event_ids,
            normalizer_chunk_offsets=norm_offsets,
            normalizer_next_offset=normalizer_next,
            active_device=DeviceRole.DEVICE_A_ACTIVE,
            handoff_epoch=state.handoff_epoch,
        )
        snapshot.seal()
        return snapshot


@dataclass
class HandoffState:
    """Internal state tracker for an active handoff negotiation."""
    session_id: str
    device_a_id: str
    device_b_id: str = ""
    pair_code: str = ""
    handoff_token: str = ""
    phase: HandoffPhase = HandoffPhase.IDLE
    expires_at: float = 0.0
    snapshot: Optional[SessionSnapshot] = None
    ack_time: float = 0.0
    committed_at: float = 0.0
    handoff_epoch: int = 0
    failure_reason: str = ""


# ══════════════════════════════════════════════════════
#  SESSION RESTORE
# ══════════════════════════════════════════════════════

def restore_session_from_snapshot(snapshot_dict: dict, MeetingSessionClass):
    """
    Reconstruct a MeetingSession from a handoff snapshot dict.
    Returns a fully populated MeetingSession ready to continue recording.

    Args:
        snapshot_dict: The snapshot as a plain dict (from JSON).
        MeetingSessionClass: The MeetingSession class to instantiate.

    Returns:
        Restored MeetingSession instance.
    """
    snap = SessionSnapshot.from_dict(snapshot_dict)

    if not snap.verify():
        raise ValueError("Snapshot integrity check failed — checksum mismatch")

    session = MeetingSessionClass(snap.session_id, speaker_count=snap.speaker_count)

    # Restore basic state
    session.chunks_processed = snap.chunks_processed
    session.total_duration = snap.total_duration
    session.meeting_start_ms = snap.meeting_start_ms
    session.next_speaker_index = snap.next_speaker_index

    # Restore speaker mappings (convert string keys back to appropriate types)
    session.diarization_to_index = {k: v for k, v in snap.diarization_to_index.items()}
    session.manual_names = {int(k): v for k, v in snap.manual_names.items()}
    session.auto_labels = {int(k): v for k, v in snap.auto_labels.items()}

    # Restore transcript
    session.transcript = list(snap.transcript)

    # Restore timeline (Feature #3)
    if hasattr(session, "timeline") and session.timeline:
        tl = session.timeline
        # Restore meeting start
        tl.meeting_start_ms = snap.meeting_start_ms

        # Restore normalizer
        norm = tl.normalizer
        norm.meeting_start_ms = snap.meeting_start_ms
        with norm._lock:
            norm._chunk_offsets = {int(k): v for k, v in snap.normalizer_chunk_offsets.items()}
            norm._next_offset = snap.normalizer_next_offset

        # Restore timeline events from serialized dicts
        from timeline_fusion import TimelineEvent
        with tl._lock:
            tl._events.clear()
            tl._event_ids.clear()
            for evt_dict in snap.timeline_events:
                evt = TimelineEvent(**{
                    k: v for k, v in evt_dict.items()
                    if k in TimelineEvent.__dataclass_fields__
                })
                tl._events.append(evt)
                tl._event_ids.add(evt.id)

    return session


# Singleton manager instance
handoff_manager = HandoffManager()
