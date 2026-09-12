"""
Task 3: Overlap & Temporal Clash Detection Engine
=================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution Suite

Patent Target: Multi-Modal Temporal Clash Detection & Probabilistic Voice Fingerprint Decomposition

Capabilities:
  - Scans multi-track speaker turns from Task 2 (pyannote) to identify collision zones.
  - Computes microsecond-accurate intersection boundaries [t_start, t_end] for simultaneous speech.
  - Cross-references clash intervals with Task 1 (faster-whisper) word timestamps to isolate colliding tokens.
  - Separates the audio stream into:
      1. `clean_segments`: Single-speaker references used to enroll voice fingerprints (Task 4).
      2. `overlap_segments`: Collision intervals requiring probabilistic attribution (Task 5).
"""

import os
import sys
import logging
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Set, Tuple

# Path bootstrap
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)
for p in [_root_dir, _parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from voice_intelligence.config import MIN_OVERLAP_DURATION_SEC, TEMP_DIR  # type: ignore
    from voice_intelligence.diarization_engine import SpeakerTurn, DiarizationResult  # type: ignore
    from voice_intelligence.whisper_transcriber import WordToken, TranscriptionResult  # type: ignore
    from voice_intelligence.overlap_suite.audio_utils import slice_wav_file  # type: ignore
except ImportError:
    from config import MIN_OVERLAP_DURATION_SEC, TEMP_DIR  # type: ignore
    from diarization_engine import SpeakerTurn, DiarizationResult  # type: ignore
    from whisper_transcriber import WordToken, TranscriptionResult  # type: ignore
    from overlap_suite.audio_utils import slice_wav_file  # type: ignore

logger = logging.getLogger("voice_intelligence.overlap_detector")


@dataclass
class OverlapSegment:
    """Represents an isolated temporal clash where two or more speakers spoke concurrently."""
    start: float
    end: float
    duration: float
    speakers: List[str]
    words: List[WordToken] = field(default_factory=list)
    text: str = ""
    audio_slice_path: Optional[str] = None
    confidence_level: str = "high"  # high, medium, low based on overlap duration and word coverage

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "speakers": self.speakers,
            "text": self.text,
            "words": [
                {
                    "word": w.word,
                    "start": round(w.start, 3),
                    "end": round(w.end, 3),
                    "probability": round(w.probability, 3),
                }
                for w in self.words
            ],
            "audio_slice_path": self.audio_slice_path,
            "confidence_level": self.confidence_level,
        }


@dataclass
class CollisionSummary:
    """Comprehensive analysis of overlapping speech across an audio timeline."""
    total_audio_duration: float = 0.0
    total_overlap_duration: float = 0.0
    overlap_ratio: float = 0.0  # Percentage of audio with speech collision
    total_collisions: int = 0
    clashing_speaker_pairs: List[Tuple[str, str]] = field(default_factory=list)
    clean_segments: List[SpeakerTurn] = field(default_factory=list)
    overlap_segments: List[OverlapSegment] = field(default_factory=list)

    @property
    def has_clashes(self) -> bool:
        return len(self.overlap_segments) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_audio_duration": round(self.total_audio_duration, 2),
            "total_overlap_duration": round(self.total_overlap_duration, 2),
            "overlap_percentage": round(self.overlap_ratio * 100.0, 2),
            "total_collisions": self.total_collisions,
            "clashing_speaker_pairs": [list(pair) for pair in self.clashing_speaker_pairs],
            "num_clean_segments": len(self.clean_segments),
            "num_overlap_segments": len(self.overlap_segments),
            "overlap_segments": [s.to_dict() for s in self.overlap_segments],
            "clean_segments": [s.to_dict() for s in self.clean_segments],
        }


class OverlapDetector:
    """
    Mathematical sweep-line collision detector that identifies overlapping speaker turns
    and correlates them with word tokens.
    """

    def __init__(self, min_overlap_sec: float = MIN_OVERLAP_DURATION_SEC):
        self.min_overlap_sec = float(min_overlap_sec)

    def detect_overlaps(
        self,
        diarization: DiarizationResult,
        transcription: Optional[TranscriptionResult] = None,
        audio_path: Optional[str] = None,
        slice_audio: bool = True,
    ) -> CollisionSummary:
        """
        Detect multi-speaker collision intervals from a DiarizationResult.
        """
        turns = diarization.turns
        total_duration = diarization.duration
        return self.detect_overlaps_from_turns(
            turns=turns,
            transcription=transcription,
            total_duration=total_duration,
            audio_path=audio_path,
            slice_audio=slice_audio,
        )

    def detect_overlaps_from_turns(
        self,
        turns: List[SpeakerTurn],
        transcription: Optional[TranscriptionResult] = None,
        total_duration: float = 0.0,
        audio_path: Optional[str] = None,
        slice_audio: bool = True,
    ) -> CollisionSummary:
        """
        Interval sweep-line algorithm:
        1. Deconstructs timeline into discrete atomic time boundaries.
        2. Identifies intervals where active_speakers >= 2.
        3. Merges contiguous or adjacent collision windows.
        4. Cross-references word tokens occurring in collision windows.
        """
        if not turns:
            return CollisionSummary(total_audio_duration=total_duration)

        if total_duration <= 0.0:
            total_duration = max(t.end for t in turns)

        # 1. Collect all boundary points
        time_points: Set[float] = set()
        for t in turns:
            time_points.add(round(t.start, 4))
            time_points.add(round(t.end, 4))
        sorted_points = sorted(list(time_points))

        # 2. Evaluate active speakers in each atomic interval [t_i, t_{i+1}]
        raw_intervals: List[Dict[str, Any]] = []
        clean_turns: List[SpeakerTurn] = []

        for i in range(len(sorted_points) - 1):
            t_start = sorted_points[i]
            t_end = sorted_points[i + 1]
            dur = t_end - t_start
            if dur < 1e-4:
                continue

            # Midpoint test to determine speaker activity
            mid = (t_start + t_end) / 2.0
            active = set()
            for t in turns:
                if t.start <= mid <= t.end:
                    active.add(t.speaker_id)

            if len(active) >= 2:
                raw_intervals.append({
                    "start": t_start,
                    "end": t_end,
                    "speakers": sorted(list(active)),
                })
            elif len(active) == 1:
                spk = list(active)[0]
                clean_turns.append(SpeakerTurn(speaker_id=spk, start=t_start, end=t_end))

        # 3. Merge contiguous collision intervals with identical or overlapping speakers
        merged_collisions: List[Dict[str, Any]] = []
        for interval in raw_intervals:
            if not merged_collisions:
                merged_collisions.append({
                    "start": interval["start"],
                    "end": interval["end"],
                    "speakers": set(interval["speakers"]),
                })
            else:
                prev = merged_collisions[-1]
                # If contiguous and same speakers or adjacent
                if abs(prev["end"] - interval["start"]) < 0.05:
                    prev["end"] = interval["end"]
                    prev["speakers"].update(interval["speakers"])
                else:
                    merged_collisions.append({
                        "start": interval["start"],
                        "end": interval["end"],
                        "speakers": set(interval["speakers"]),
                    })

        # 4. Merge contiguous clean turns for the same speaker
        merged_clean: List[SpeakerTurn] = []
        for ct in clean_turns:
            if not merged_clean:
                merged_clean.append(ct)
            else:
                prev_ct = merged_clean[-1]
                if prev_ct.speaker_id == ct.speaker_id and abs(prev_ct.end - ct.start) < 0.05:
                    prev_ct.end = ct.end
                else:
                    merged_clean.append(ct)

        # 5. Filter collisions by minimum threshold and pair extraction
        clashing_pairs: Set[Tuple[str, str]] = set()
        overlap_segments: List[OverlapSegment] = []
        total_overlap_sec = 0.0

        for idx, mc in enumerate(merged_collisions):
            dur = mc["end"] - mc["start"]
            if dur < self.min_overlap_sec:
                continue

            spk_list = sorted(list(mc["speakers"]))
            # Record clashing pairs
            for p1 in range(len(spk_list)):
                for p2 in range(p1 + 1, len(spk_list)):
                    clashing_pairs.add((spk_list[p1], spk_list[p2]))

            total_overlap_sec += dur

            # Extract colliding words from transcription if available
            colliding_words: List[WordToken] = []
            colliding_text = ""
            if transcription is not None:
                colliding_words = transcription.get_words_in_timerange(mc["start"], mc["end"])
                colliding_text = " ".join(w.word for w in colliding_words).strip()

            # Slice collision audio if requested and audio_path exists
            slice_path = None
            if slice_audio and audio_path and os.path.exists(audio_path):
                try:
                    slice_name = f"overlap_{idx+1}_{int(mc['start']*1000)}ms_{int(mc['end']*1000)}ms.wav"
                    target_slice = os.path.join(TEMP_DIR, slice_name)
                    slice_path = slice_wav_file(audio_path, mc["start"], mc["end"], target_slice)
                except Exception as slice_err:
                    logger.warning(f"Audio slicing skipped: {slice_err}")

            overlap_segments.append(
                OverlapSegment(
                    start=round(mc["start"], 3),
                    end=round(mc["end"], 3),
                    duration=round(dur, 3),
                    speakers=spk_list,
                    words=colliding_words,
                    text=colliding_text,
                    audio_slice_path=slice_path,
                    confidence_level="high" if dur >= 0.5 else "medium",
                )
            )

        ratio = (total_overlap_sec / total_duration) if total_duration > 0 else 0.0

        summary = CollisionSummary(
            total_audio_duration=round(total_duration, 3),
            total_overlap_duration=round(total_overlap_sec, 3),
            overlap_ratio=round(ratio, 4),
            total_collisions=len(overlap_segments),
            clashing_speaker_pairs=sorted(list(clashing_pairs)),
            clean_segments=merged_clean,
            overlap_segments=overlap_segments,
        )
        return summary


# Global singleton factory
_detector_instance: Optional[OverlapDetector] = None


def get_overlap_detector(min_overlap_sec: float = MIN_OVERLAP_DURATION_SEC) -> OverlapDetector:
    """Return a shared singleton OverlapDetector instance."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = OverlapDetector(min_overlap_sec=min_overlap_sec)
    return _detector_instance
