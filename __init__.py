"""
Voice & Speaker Intelligence Package
====================================
Feature #1: Overlapping-Speech Attribution
Owner: Vedant (Member A)

Modules:
  - whisper_transcriber: Task 1 - faster-whisper with word-level timestamps
  - diarization_engine:  Task 2 - pyannote speaker diarization
  - overlap_detector:    Task 3 - Temporal clash & collision detection
  - voice_fingerprint:   Task 4 - SpeechBrain ECAPA-TDNN voice embeddings on mixed audio
  - attribution_engine:  Task 5 - Overlapping attribution confidence (soft scoring)
  - benchmark_eval:      Task 6 - Empirical metrics and patent claim evidence logger
"""

__version__ = "1.0.0"
__author__ = "Vedant"

from voice_intelligence.whisper_transcriber import (  # type: ignore
    FasterWhisperTranscriber,
    WordToken,
    TranscriptSegment,
    TranscriptionResult,
    get_transcriber,
)

from voice_intelligence.diarization_engine import (  # type: ignore
    PyannoteDiarizer,
    SpeakerTurn,
    DiarizationResult,
    get_diarizer,
)

from voice_intelligence.overlap_detector import (  # type: ignore
    OverlapSegment,
    CollisionSummary,
    OverlapDetector,
    get_overlap_detector,
)

from voice_intelligence.voice_fingerprint import (  # type: ignore
    VoiceFingerprint,
    SpeakerProfileRegistry,
    VoiceFingerprintEngine,
    get_fingerprint_engine,
)

from voice_intelligence.attribution_engine import (  # type: ignore
    WordAttribution,
    AttributedOverlapSegment,
    AttributionReport,
    AttributionEngine,
    get_attribution_engine,
)

from voice_intelligence.benchmark_eval import (  # type: ignore
    BenchmarkMetricSet,
    PatentBenchmarkHarness,
    get_benchmark_harness,
)

from voice_intelligence.overlap_suite import (  # type: ignore
    UnifiedTranscriptItem,
    PipelineExecutionResult,
    OverlappingAttributionPipeline,
    get_pipeline,
)

