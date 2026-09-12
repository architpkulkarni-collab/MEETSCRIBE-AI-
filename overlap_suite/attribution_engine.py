"""
Task 5: Probabilistic Attribution Engine for Overlapping Speech
===============================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution Suite

Patent Target: Multi-Modal Temporal Clash Detection & Probabilistic Voice Fingerprint Decomposition

Capabilities:
  - Takes collision intervals from Task 3 and enrolled speaker profiles from Task 4.
  - Decomposes simultaneous mixed-signal audio into soft probabilistic attributions (e.g. 72% Spk0, 28% Spk1).
  - Uses temperature-scaled acoustic softmax projection:
        P(S_k | mixed) = exp(sim(S_k) / tau) / sum_j exp(sim(S_j) / tau)
  - Produces word-level attributions for all words colliding during the speech clash.
  - Generates comprehensive attribution reports for downstream UI display and meeting summaries.
"""

import os
import sys
import math
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple

# Path bootstrap
_current_dir = os.path.dirname(os.path.abspath(__file__))
_parent_dir = os.path.dirname(_current_dir)
_root_dir = os.path.dirname(_parent_dir)
for p in [_root_dir, _parent_dir, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from voice_intelligence.config import ATTRIBUTION_TEMPERATURE, MIN_OVERLAP_DURATION_SEC  # type: ignore
    from voice_intelligence.overlap_suite.overlap_detector import OverlapSegment, CollisionSummary  # type: ignore
    from voice_intelligence.overlap_suite.voice_fingerprint import (  # type: ignore
        VoiceFingerprint,
        SpeakerProfileRegistry,
        VoiceFingerprintEngine,
        get_fingerprint_engine,
    )
    from voice_intelligence.whisper_transcriber import WordToken  # type: ignore
except ImportError:
    from config import ATTRIBUTION_TEMPERATURE, MIN_OVERLAP_DURATION_SEC  # type: ignore
    from overlap_suite.overlap_detector import OverlapSegment, CollisionSummary  # type: ignore
    from overlap_suite.voice_fingerprint import (  # type: ignore
        VoiceFingerprint,
        SpeakerProfileRegistry,
        VoiceFingerprintEngine,
        get_fingerprint_engine,
    )
    from whisper_transcriber import WordToken  # type: ignore

logger = logging.getLogger("voice_intelligence.attribution_engine")


@dataclass
class WordAttribution:
    """Attribution metadata for a single word spoken during an overlapping clash."""
    word: str
    start: float
    end: float
    attributed_speaker: str
    confidence: float
    speaker_probabilities: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "word": self.word,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "attributed_speaker": self.attributed_speaker,
            "confidence": round(self.confidence, 3),
            "speaker_probabilities": {
                k: round(v, 3) for k, v in self.speaker_probabilities.items()
            },
        }


@dataclass
class AttributedOverlapSegment:
    """Decomposed collision interval with probabilistic speaker attribution."""
    start: float
    end: float
    duration: float
    candidate_speakers: List[str]
    dominant_speaker: str
    secondary_speaker: Optional[str]
    speaker_probabilities: Dict[str, float]
    raw_similarities: Dict[str, float]
    word_attributions: List[WordAttribution] = field(default_factory=list)
    text: str = ""
    clash_severity: str = "moderate"  # 'severe' (near 50/50), 'moderate', or 'mild'

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": round(self.duration, 3),
            "candidate_speakers": self.candidate_speakers,
            "dominant_speaker": self.dominant_speaker,
            "secondary_speaker": self.secondary_speaker,
            "speaker_probabilities": {
                k: round(v, 3) for k, v in self.speaker_probabilities.items()
            },
            "raw_similarities": {
                k: round(v, 4) for k, v in self.raw_similarities.items()
            },
            "text": self.text,
            "clash_severity": self.clash_severity,
            "word_attributions": [w.to_dict() for w in self.word_attributions],
        }


@dataclass
class AttributionReport:
    """Master attribution report for an entire multi-speaker meeting."""
    total_audio_duration: float
    total_collisions: int
    total_words_in_overlap: int
    attributed_segments: List[AttributedOverlapSegment] = field(default_factory=list)
    speaker_dominance_summary: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_audio_duration": round(self.total_audio_duration, 2),
            "total_collisions": self.total_collisions,
            "total_words_in_overlap": self.total_words_in_overlap,
            "speaker_dominance_summary": {
                k: round(v, 3) for k, v in self.speaker_dominance_summary.items()
            },
            "attributed_segments": [s.to_dict() for s in self.attributed_segments],
        }


class AttributionEngine:
    """
    Decomposes overlapping speech collisions using temperature-scaled softmax over
    cosine similarity projections between mixed audio fingerprints and enrolled speaker profiles.
    """

    def __init__(
        self,
        temperature: float = ATTRIBUTION_TEMPERATURE,
        fingerprint_engine: Optional[VoiceFingerprintEngine] = None,
    ):
        self.temperature = max(0.01, float(temperature))
        self.fingerprint_engine = fingerprint_engine or get_fingerprint_engine()

    def attribute_collision(
        self,
        overlap_seg: OverlapSegment,
        audio_path: Optional[str],
        registry: SpeakerProfileRegistry,
    ) -> AttributedOverlapSegment:
        """
        Decomposes a single collision interval into probabilistic speaker attributions.
        """
        candidates = list(overlap_seg.speakers)
        if not candidates:
            candidates = registry.all_speakers()

        # If candidates are still missing, fallback gracefully
        if not candidates:
            candidates = ["SPEAKER_00", "SPEAKER_01"]

        # 1. Extract fingerprint of mixed collision audio
        mixed_vector: List[float] = []
        if audio_path and os.path.exists(audio_path):
            mixed_fp = self.fingerprint_engine.extract_embedding(
                audio_input=audio_path,
                speaker_id="MIXED_COLLISION",
                start_sec=overlap_seg.start,
                end_sec=overlap_seg.end,
            )
            mixed_vector = mixed_fp.vector
        elif overlap_seg.audio_slice_path and os.path.exists(overlap_seg.audio_slice_path):
            mixed_fp = self.fingerprint_engine.extract_embedding(
                audio_input=overlap_seg.audio_slice_path,
                speaker_id="MIXED_COLLISION",
            )
            mixed_vector = mixed_fp.vector

        # 2. Compute similarity against enrolled profiles
        similarities: Dict[str, float] = {}
        for spk in candidates:
            ref = registry.get(spk)
            if ref is not None and mixed_vector:
                sim = VoiceFingerprintEngine.compute_cosine_similarity(mixed_vector, ref.vector)
                similarities[spk] = sim
            else:
                # Balanced heuristic if profile missing
                similarities[spk] = 0.5

        # 3. Apply Temperature-Scaled Softmax: P(S_k) = exp(sim_k / T) / sum(exp(sim_j / T))
        probs = self._softmax_distribution(similarities, self.temperature)

        # 4. Determine dominant and secondary speakers
        sorted_speakers = sorted(probs.items(), key=lambda item: item[1], reverse=True)
        dominant_spk = sorted_speakers[0][0]
        secondary_spk = sorted_speakers[1][0] if len(sorted_speakers) > 1 else None

        # Determine clash severity
        top_prob = sorted_speakers[0][1]
        second_prob = sorted_speakers[1][1] if len(sorted_speakers) > 1 else 0.0
        diff = top_prob - second_prob
        if diff < 0.15:
            severity = "severe"  # Head-to-head clash (~50/50)
        elif diff < 0.40:
            severity = "moderate"
        else:
            severity = "mild"

        # 5. Word-level attribution for words within this collision
        word_attrs: List[WordAttribution] = []
        for w in overlap_seg.words:
            # Word-level attribution inheriting interval distribution
            word_attrs.append(
                WordAttribution(
                    word=w.word,
                    start=w.start,
                    end=w.end,
                    attributed_speaker=dominant_spk,
                    confidence=probs[dominant_spk],
                    speaker_probabilities=probs,
                )
            )

        return AttributedOverlapSegment(
            start=overlap_seg.start,
            end=overlap_seg.end,
            duration=overlap_seg.duration,
            candidate_speakers=candidates,
            dominant_speaker=dominant_spk,
            secondary_speaker=secondary_spk,
            speaker_probabilities=probs,
            raw_similarities=similarities,
            word_attributions=word_attrs,
            text=overlap_seg.text,
            clash_severity=severity,
        )

    def attribute_all(
        self,
        collision_summary: CollisionSummary,
        audio_path: Optional[str],
        registry: SpeakerProfileRegistry,
    ) -> AttributionReport:
        """
        Attributes all detected collision segments across the entire audio timeline.
        """
        attributed_segs: List[AttributedOverlapSegment] = []
        total_words = 0
        speaker_duration_accum: Dict[str, float] = {}

        for seg in collision_summary.overlap_segments:
            attr = self.attribute_collision(seg, audio_path, registry)
            attributed_segs.append(attr)
            total_words += len(attr.word_attributions)

            # Accumulate probabilistic duration
            for spk, prob in attr.speaker_probabilities.items():
                speaker_duration_accum[spk] = (
                    speaker_duration_accum.get(spk, 0.0) + (attr.duration * prob)
                )

        # Normalize dominance
        total_overlap_sec = sum(speaker_duration_accum.values())
        dominance_summary = {}
        if total_overlap_sec > 0:
            dominance_summary = {
                spk: round(dur / total_overlap_sec, 3)
                for spk, dur in speaker_duration_accum.items()
            }

        return AttributionReport(
            total_audio_duration=collision_summary.total_audio_duration,
            total_collisions=len(attributed_segs),
            total_words_in_overlap=total_words,
            attributed_segments=attributed_segs,
            speaker_dominance_summary=dominance_summary,
        )

    @staticmethod
    def _softmax_distribution(similarities: Dict[str, float], temperature: float) -> Dict[str, float]:
        """
        Computes numerically stable temperature-scaled softmax.
        """
        if not similarities:
            return {}

        max_sim = max(similarities.values())
        exps = {
            k: math.exp((v - max_sim) / temperature) for k, v in similarities.items()
        }
        total = sum(exps.values())
        if total <= 0:
            total = 1.0
        return {k: round(v / total, 4) for k, v in exps.items()}


# Global singleton factory
_attribution_engine: Optional[AttributionEngine] = None


def get_attribution_engine(temperature: float = ATTRIBUTION_TEMPERATURE) -> AttributionEngine:
    global _attribution_engine
    if _attribution_engine is None:
        _attribution_engine = AttributionEngine(temperature=temperature)
    return _attribution_engine
