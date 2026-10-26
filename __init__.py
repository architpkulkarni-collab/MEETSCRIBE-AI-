"""
Overlapping-Speech Attribution Suite
====================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Modules:
  - audio_utils: Sub-second WAV slicing and synthetic collision generators
  - overlap_detector: Task 3 - Temporal clash and collision detection
  - voice_fingerprint: Task 4 - SpeechBrain ECAPA-TDNN & acoustic fingerprinting
  - attribution_engine: Task 5 - Probabilistic soft decomposition and scoring
  - benchmark_eval: Task 6 - Empirical patent proof benchmark harness
  - pipeline: Full end-to-end multi-modal meeting attribution orchestrator
"""

from voice_intelligence.overlap_suite.audio_utils import (
    slice_wav_file,
    read_wav_samples,
    write_wav_samples,
    generate_multi_speaker_overlapping_wav,
)

from voice_intelligence.overlap_suite.overlap_detector import (
    OverlapSegment,
    CollisionSummary,
    OverlapDetector,
    get_overlap_detector,
)

from voice_intelligence.overlap_suite.voice_fingerprint import (
    VoiceFingerprint,
    SpeakerProfileRegistry,
    VoiceFingerprintEngine,
    get_fingerprint_engine,
)

from voice_intelligence.overlap_suite.attribution_engine import (
    WordAttribution,
    AttributedOverlapSegment,
    AttributionReport,
    AttributionEngine,
    get_attribution_engine,
)

from voice_intelligence.overlap_suite.benchmark_eval import (
    BenchmarkMetricSet,
    PatentBenchmarkHarness,
    get_benchmark_harness,
)

from voice_intelligence.overlap_suite.pipeline import (
    UnifiedTranscriptItem,
    PipelineExecutionResult,
    OverlappingAttributionPipeline,
    get_pipeline,
)

__all__ = [
    # Audio Utils
    "slice_wav_file",
    "read_wav_samples",
    "write_wav_samples",
    "generate_multi_speaker_overlapping_wav",
    # Task 3
    "OverlapSegment",
    "CollisionSummary",
    "OverlapDetector",
    "get_overlap_detector",
    # Task 4
    "VoiceFingerprint",
    "SpeakerProfileRegistry",
    "VoiceFingerprintEngine",
    "get_fingerprint_engine",
    # Task 5
    "WordAttribution",
    "AttributedOverlapSegment",
    "AttributionReport",
    "AttributionEngine",
    "get_attribution_engine",
    # Task 6
    "BenchmarkMetricSet",
    "PatentBenchmarkHarness",
    "get_benchmark_harness",
    # Pipeline
    "UnifiedTranscriptItem",
    "PipelineExecutionResult",
    "OverlappingAttributionPipeline",
    "get_pipeline",
]
