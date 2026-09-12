"""
Multi-Modal Timeline Fusion — Event Model & Engine
=====================================================
Feature #3: Combines transcript, typed note, and screen/OCR events
into one unified chronological meeting timeline.

Event model, timestamp normalization, drift correction, and fusion.
"""

import time
import uuid
import threading
from dataclasses import dataclass, field, asdict
from typing import Optional


# ══════════════════════════════════════════════════════
#  EVENT MODEL
# ══════════════════════════════════════════════════════

VALID_EVENT_TYPES = {"transcript", "note", "screen", "ocr"}
VALID_SOURCES = {"whisper", "web_speech", "user", "screen_capture", "ocr_engine"}

# Maximum allowed correction in seconds (bounded drift)
MAX_CORRECTION_SECONDS = 30.0


@dataclass
class TimelineEvent:
    """
    Normalized event for the unified meeting timeline.

    Attributes:
        id: Unique event identifier.
        timestamp: Authoritative event time (epoch ms). For transcripts, this is
                   meeting-relative. For screenshots, this is capture time, NOT
                   OCR completion time.
        type: One of: transcript, note, screen, ocr.
        source: Origin system (whisper, web_speech, user, screen_capture, ocr_engine).
        content: The event payload (text, OCR result, etc.).
        metadata: Optional dict for speaker info, confidence, chunk details, etc.
        raw_timestamp: Original timestamp before any correction.
        corrected_timestamp: Timestamp after drift correction (None if uncorrected).
        timestamp_correction: Signed correction delta in ms (0 if none).
        correction_reason: Human-readable reason for correction (None if none).
        correction_confidence: 0.0–1.0 confidence in the correction (None if none).
    """
    id: str
    timestamp: float  # epoch ms
    type: str
    source: str
    content: str
    metadata: dict = field(default_factory=dict)
    raw_timestamp: Optional[float] = None
    corrected_timestamp: Optional[float] = None
    timestamp_correction: float = 0.0
    correction_reason: Optional[str] = None
    correction_confidence: Optional[float] = None

    def __post_init__(self):
        if self.type not in VALID_EVENT_TYPES:
            raise ValueError(f"Invalid event type '{self.type}'. Must be one of {VALID_EVENT_TYPES}")
        if self.raw_timestamp is None:
            self.raw_timestamp = self.timestamp

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/API responses."""
        d = asdict(self)
        # Remove None values for cleaner output
        return {k: v for k, v in d.items() if v is not None}


def create_event(
    event_type: str,
    source: str,
    content: str,
    timestamp: Optional[float] = None,
    metadata: Optional[dict] = None,
    event_id: Optional[str] = None,
) -> TimelineEvent:
    """
    Factory for creating timeline events with sensible defaults.

    Args:
        event_type: One of transcript, note, screen, ocr.
        source: Origin system identifier.
        content: Event payload text.
        timestamp: Epoch ms. Defaults to current time if not provided.
        metadata: Optional metadata dict.
        event_id: Optional custom ID. Auto-generated if not provided.

    Returns:
        TimelineEvent instance.
    """
    return TimelineEvent(
        id=event_id or str(uuid.uuid4())[:12],
        timestamp=timestamp or (time.time() * 1000),
        type=event_type,
        source=source,
        content=content,
        metadata=metadata or {},
    )


# ══════════════════════════════════════════════════════
#  WHISPER CHUNK TIMESTAMP NORMALIZER
# ══════════════════════════════════════════════════════

class WhisperTimestampNormalizer:
    """
    Normalizes Whisper chunk-relative timestamps into a single
    meeting-relative timeline.

    Problem: Whisper produces timestamps relative to each audio chunk
    (e.g., chunk 2 starts at 0.0s again). This class tracks the
    cumulative offset so that timestamps are meeting-absolute.

    Usage:
        normalizer = WhisperTimestampNormalizer(meeting_start_epoch_ms)
        normalizer.register_chunk(chunk_index, chunk_duration_seconds)
        abs_ms = normalizer.normalize(chunk_index, chunk_relative_seconds)
    """

    def __init__(self, meeting_start_ms: float):
        """
        Args:
            meeting_start_ms: Epoch ms when the meeting started.
        """
        self.meeting_start_ms = meeting_start_ms
        self._chunk_offsets: dict[int, float] = {}  # chunk_index -> cumulative offset in seconds
        self._next_offset: float = 0.0
        self._lock = threading.Lock()

    def register_chunk(self, chunk_index: int, chunk_duration_seconds: float):
        """
        Register a chunk and its duration. The offset for this chunk
        is the sum of all previous chunk durations.

        Must be called in order (chunk 0, 1, 2, ...).
        """
        with self._lock:
            if chunk_index not in self._chunk_offsets:
                self._chunk_offsets[chunk_index] = self._next_offset
                self._next_offset += chunk_duration_seconds

    def normalize(self, chunk_index: int, chunk_relative_seconds: float) -> float:
        """
        Convert a chunk-relative Whisper timestamp to meeting-absolute epoch ms.

        Args:
            chunk_index: Which audio chunk this timestamp came from.
            chunk_relative_seconds: Whisper's timestamp within the chunk (e.g. 2.4s).

        Returns:
            Epoch ms representing the absolute meeting time.
        """
        with self._lock:
            offset = self._chunk_offsets.get(chunk_index, 0.0)
        meeting_relative_seconds = offset + chunk_relative_seconds
        return self.meeting_start_ms + (meeting_relative_seconds * 1000)

    def get_chunk_offset(self, chunk_index: int) -> float:
        """Get the cumulative offset in seconds for a given chunk."""
        with self._lock:
            return self._chunk_offsets.get(chunk_index, 0.0)


# ══════════════════════════════════════════════════════
#  TIMING DRIFT CORRECTION
# ══════════════════════════════════════════════════════

def correct_ocr_timestamp(
    capture_timestamp_ms: float,
    ocr_completion_timestamp_ms: float,
) -> TimelineEvent:
    """
    Create correction metadata for an OCR event where capture time
    differs from processing completion time.

    The capture timestamp is authoritative. The correction records
    the delta and reason.

    Returns a dict of correction fields to apply to a TimelineEvent.
    """
    processing_delay_ms = ocr_completion_timestamp_ms - capture_timestamp_ms

    # Bound the correction
    if abs(processing_delay_ms) > MAX_CORRECTION_SECONDS * 1000:
        # Correction too large — something is wrong, don't correct
        return {
            "timestamp": ocr_completion_timestamp_ms,
            "raw_timestamp": ocr_completion_timestamp_ms,
            "corrected_timestamp": None,
            "timestamp_correction": 0.0,
            "correction_reason": f"Correction rejected: delta {processing_delay_ms:.0f}ms exceeds bound",
            "correction_confidence": 0.0,
        }

    return {
        "timestamp": capture_timestamp_ms,  # Use capture time as authoritative
        "raw_timestamp": ocr_completion_timestamp_ms,
        "corrected_timestamp": capture_timestamp_ms,
        "timestamp_correction": -processing_delay_ms,
        "correction_reason": f"OCR processing delay: {processing_delay_ms:.0f}ms",
        "correction_confidence": min(1.0, 1.0 - (processing_delay_ms / (MAX_CORRECTION_SECONDS * 1000))),
    }


def correct_whisper_chunk_timestamp(
    raw_chunk_relative_ms: float,
    normalized_meeting_ms: float,
    chunk_index: int,
) -> dict:
    """
    Create correction metadata for a Whisper timestamp that was
    normalized from chunk-relative to meeting-absolute.

    Args:
        raw_chunk_relative_ms: The original Whisper timestamp (chunk-relative, in ms).
        normalized_meeting_ms: The corrected meeting-absolute timestamp (in ms).
        chunk_index: Which chunk this came from.

    Returns:
        Dict of correction fields.
    """
    delta = normalized_meeting_ms - raw_chunk_relative_ms

    if chunk_index == 0 and abs(delta) < 1:
        # First chunk, no correction needed
        return {}

    return {
        "timestamp": normalized_meeting_ms,
        "raw_timestamp": raw_chunk_relative_ms,
        "corrected_timestamp": normalized_meeting_ms,
        "timestamp_correction": delta,
        "correction_reason": f"Whisper chunk {chunk_index} offset normalization",
        "correction_confidence": 1.0,  # Deterministic correction
    }


# ══════════════════════════════════════════════════════
#  TIMELINE FUSION ENGINE
# ══════════════════════════════════════════════════════

class MeetingTimeline:
    """
    Unified chronological timeline that fuses transcript, note,
    screen capture, and OCR events.

    Events are always maintained in chronological order by timestamp.
    Out-of-order arrivals are inserted at the correct position.
    Duplicate events (same ID) are rejected.

    Thread-safe.
    """

    def __init__(self, meeting_start_ms: Optional[float] = None):
        """
        Args:
            meeting_start_ms: Epoch ms when the meeting started. Defaults to now.
        """
        self.meeting_start_ms = meeting_start_ms or (time.time() * 1000)
        self._events: list[TimelineEvent] = []
        self._event_ids: set[str] = set()
        self._lock = threading.Lock()
        self._normalizer = WhisperTimestampNormalizer(self.meeting_start_ms)

    @property
    def normalizer(self) -> WhisperTimestampNormalizer:
        return self._normalizer

    def add_event(self, event: TimelineEvent) -> bool:
        """
        Add an event to the timeline. Inserts at the correct
        chronological position.

        Returns False if the event ID already exists (duplicate).
        """
        with self._lock:
            if event.id in self._event_ids:
                return False

            self._event_ids.add(event.id)

            # Binary search for insertion point to maintain chronological order
            lo, hi = 0, len(self._events)
            while lo < hi:
                mid = (lo + hi) // 2
                if self._events[mid].timestamp <= event.timestamp:
                    lo = mid + 1
                else:
                    hi = mid

            self._events.insert(lo, event)
            return True

    def get_events(
        self,
        event_type: Optional[str] = None,
        since_ms: Optional[float] = None,
        limit: Optional[int] = None,
    ) -> list[TimelineEvent]:
        """
        Get timeline events, optionally filtered.

        Args:
            event_type: Filter by type (transcript, note, screen, ocr).
            since_ms: Only return events after this timestamp.
            limit: Max number of events to return.

        Returns:
            List of TimelineEvent in chronological order.
        """
        with self._lock:
            result = self._events[:]

        if event_type:
            result = [e for e in result if e.type == event_type]
        if since_ms is not None:
            result = [e for e in result if e.timestamp >= since_ms]
        if limit:
            result = result[-limit:]

        return result

    def get_event_count(self) -> dict:
        """Return count of events by type."""
        with self._lock:
            counts = {}
            for e in self._events:
                counts[e.type] = counts.get(e.type, 0) + 1
            counts["total"] = len(self._events)
            return counts

    def to_list(self) -> list[dict]:
        """Export the full timeline as a list of dicts."""
        with self._lock:
            return [e.to_dict() for e in self._events]

    def clear(self):
        """Clear all events."""
        with self._lock:
            self._events.clear()
            self._event_ids.clear()
