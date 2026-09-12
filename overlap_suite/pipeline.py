"""
End-to-End Overlapping-Speech Attribution Pipeline
===================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution Suite

Orchestrates all 5 functional stages:
  Stage 1: Word-Level Transcription (faster-whisper)
  Stage 2: Speaker Diarization (pyannote)
  Stage 3: Collision Interval Detection (sweep-line algorithm)
  Stage 4: Voice Fingerprinting (ECAPA-TDNN & acoustic enrollment)
  Stage 5: Soft Attribution & Probabilistic Decomposition
"""

import os
import sys
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# Path bootstrap
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)
for p in [_root_dir, _parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from voice_intelligence.whisper_transcriber import (  # type: ignore
        FasterWhisperTranscriber,
        TranscriptionResult,
        get_transcriber,
    )
    from voice_intelligence.diarization_engine import (  # type: ignore
        PyannoteDiarizer,
        DiarizationResult,
        SpeakerTurn,
        get_diarizer,
    )
    from voice_intelligence.overlap_suite.overlap_detector import (  # type: ignore
        OverlapDetector,
        CollisionSummary,
        OverlapSegment,
        get_overlap_detector,
    )
    from voice_intelligence.overlap_suite.voice_fingerprint import (  # type: ignore
        VoiceFingerprintEngine,
        SpeakerProfileRegistry,
        get_fingerprint_engine,
    )
    from voice_intelligence.overlap_suite.attribution_engine import (  # type: ignore
        AttributionEngine,
        AttributedOverlapSegment,
        AttributionReport,
        get_attribution_engine,
    )
except ImportError:
    from whisper_transcriber import (  # type: ignore
        FasterWhisperTranscriber,
        TranscriptionResult,
        get_transcriber,
    )
    from diarization_engine import (  # type: ignore
        PyannoteDiarizer,
        DiarizationResult,
        SpeakerTurn,
        get_diarizer,
    )
    from overlap_suite.overlap_detector import (  # type: ignore
        OverlapDetector,
        CollisionSummary,
        OverlapSegment,
        get_overlap_detector,
    )
    from overlap_suite.voice_fingerprint import (  # type: ignore
        VoiceFingerprintEngine,
        SpeakerProfileRegistry,
        get_fingerprint_engine,
    )
    from overlap_suite.attribution_engine import (  # type: ignore
        AttributionEngine,
        AttributedOverlapSegment,
        AttributionReport,
        get_attribution_engine,
    )

logger = logging.getLogger("voice_intelligence.pipeline")


@dataclass
class UnifiedTranscriptItem:
    """A single chronological element in the meeting transcript."""
    start: float
    end: float
    text: str
    speaker_id: str
    is_overlapping: bool = False
    dominant_speaker: Optional[str] = None
    secondary_speaker: Optional[str] = None
    confidence_distribution: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "text": self.text,
            "speaker": self.speaker_id,
            "is_overlapping": self.is_overlapping,
            "dominant_speaker": self.dominant_speaker,
            "secondary_speaker": self.secondary_speaker,
            "confidence_distribution": {
                k: round(v, 2) for k, v in self.confidence_distribution.items()
            },
        }


@dataclass
class PipelineExecutionResult:
    """Complete output of the Overlapping-Speech Attribution Pipeline."""
    audio_path: str
    duration_sec: float
    total_processing_time: float
    num_speakers: int
    speakers: List[str]
    has_overlaps: bool
    total_collisions: int
    transcription: TranscriptionResult
    diarization: DiarizationResult
    collision_summary: CollisionSummary
    attribution_report: AttributionReport
    unified_transcript: List[UnifiedTranscriptItem] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "audio_path": self.audio_path,
            "duration_sec": round(self.duration_sec, 2),
            "total_processing_time": round(self.total_processing_time, 2),
            "num_speakers": self.num_speakers,
            "speakers": self.speakers,
            "has_overlaps": self.has_overlaps,
            "total_collisions": self.total_collisions,
            "collision_summary": self.collision_summary.to_dict(),
            "attribution_report": self.attribution_report.to_dict(),
            "unified_transcript": [item.to_dict() for item in self.unified_transcript],
        }


class OverlappingAttributionPipeline:
    """
    End-to-End Orchestrator executing the complete 5-stage attribution flow.
    """

    def __init__(
        self,
        transcriber: Optional[FasterWhisperTranscriber] = None,
        diarizer: Optional[PyannoteDiarizer] = None,
        overlap_detector: Optional[OverlapDetector] = None,
        fp_engine: Optional[VoiceFingerprintEngine] = None,
        attribution_engine: Optional[AttributionEngine] = None,
    ):
        self.transcriber = transcriber or get_transcriber()
        self.diarizer = diarizer or get_diarizer()
        self.overlap_detector = overlap_detector or get_overlap_detector()
        self.fp_engine = fp_engine or get_fingerprint_engine()
        self.attribution_engine = attribution_engine or get_attribution_engine()

    def process(self, audio_path: str) -> PipelineExecutionResult:
        """
        Runs full pipeline on the given audio path.
        """
        t0 = time.time()
        logger.info(f"Starting End-to-End Overlapping-Speech Pipeline on '{audio_path}'...")

        # Stage 1: Word-level speech-to-text
        trans_result = self.transcriber.transcribe(audio_path, word_timestamps=True)

        # Stage 2: Multi-track speaker diarization
        diar_result = self.diarizer.diarize(audio_path)

        # Stage 3: Collision interval detection & word alignment
        collision_summary = self.overlap_detector.detect_overlaps(
            diarization=diar_result,
            transcription=trans_result,
            audio_path=audio_path,
        )

        # Stage 4: Voice fingerprinting on clean segments
        registry = self.fp_engine.enroll_speakers_from_clean_segments(
            audio_path=audio_path,
            clean_segments=collision_summary.clean_segments,
        )

        # Stage 5: Soft probabilistic attribution on colliding speech
        attr_report = self.attribution_engine.attribute_all(
            collision_summary=collision_summary,
            audio_path=audio_path,
            registry=registry,
        )

        # Assemble unified chronological transcript
        unified = self._assemble_unified_transcript(
            trans_result=trans_result,
            clean_turns=collision_summary.clean_segments,
            attributed_overlaps=attr_report.attributed_segments,
        )

        total_time = time.time() - t0
        return PipelineExecutionResult(
            audio_path=audio_path,
            duration_sec=trans_result.duration or diar_result.duration,
            total_processing_time=total_time,
            num_speakers=len(diar_result.speakers),
            speakers=diar_result.speakers,
            has_overlaps=collision_summary.has_clashes,
            total_collisions=collision_summary.total_collisions,
            transcription=trans_result,
            diarization=diar_result,
            collision_summary=collision_summary,
            attribution_report=attr_report,
            unified_transcript=unified,
        )

    def _assemble_unified_transcript(
        self,
        trans_result: TranscriptionResult,
        clean_turns: List[SpeakerTurn],
        attributed_overlaps: List[AttributedOverlapSegment],
    ) -> List[UnifiedTranscriptItem]:
        """
        Combines clean single-speaker segments and attributed collision zones
        into a coherent, chronologically ordered meeting transcript.
        """
        items: List[UnifiedTranscriptItem] = []

        # 1. Add clean single-speaker segments with their transcribed text
        for ct in clean_turns:
            words = trans_result.get_words_in_timerange(ct.start, ct.end)
            text = " ".join(w.word for w in words).strip()
            if not text:
                text = f"[Speech active by {ct.speaker_id}]"
            items.append(
                UnifiedTranscriptItem(
                    start=ct.start,
                    end=ct.end,
                    text=text,
                    speaker_id=ct.speaker_id,
                    is_overlapping=False,
                )
            )

        # 2. Add attributed overlapping segments
        for ov in attributed_overlaps:
            text = ov.text if ov.text else f"[Simultaneous speech collision: {', '.join(ov.candidate_speakers)}]"
            items.append(
                UnifiedTranscriptItem(
                    start=ov.start,
                    end=ov.end,
                    text=text,
                    speaker_id=ov.dominant_speaker,
                    is_overlapping=True,
                    dominant_speaker=ov.dominant_speaker,
                    secondary_speaker=ov.secondary_speaker,
                    confidence_distribution=ov.speaker_probabilities,
                )
            )

        # Sort chronologically by start timestamp
        items.sort(key=lambda x: x.start)
        return items


# Global singleton factory
_pipeline_instance: Optional[OverlappingAttributionPipeline] = None


def get_pipeline() -> OverlappingAttributionPipeline:
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = OverlappingAttributionPipeline()
    return _pipeline_instance
