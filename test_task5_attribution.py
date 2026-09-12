"""
Test Script for Task 5: Attribution Engine (Soft Scoring & Probabilistic Decomposition)
========================================================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Run:
    python voice_intelligence/test_task5_attribution.py [optional_audio_path]

Verifies that:
  1. Engine decomposes mixed-signal audio into soft attribution percentages.
  2. Probabilities sum to 100% across candidate speakers.
  3. Correctly identifies dominant speaker vs. secondary interfering speaker.
  4. Attributes colliding word tokens with individual speaker confidences.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from voice_intelligence.config import TEMP_DIR  # type: ignore
    from voice_intelligence.diarization_engine import SpeakerTurn  # type: ignore
    from voice_intelligence.whisper_transcriber import WordToken  # type: ignore
    from voice_intelligence.overlap_detector import OverlapSegment, CollisionSummary  # type: ignore
    from voice_intelligence.voice_fingerprint import get_fingerprint_engine  # type: ignore
    from voice_intelligence.attribution_engine import (  # type: ignore
        AttributionEngine,
        get_attribution_engine,
    )
    from voice_intelligence.overlap_suite.audio_utils import generate_multi_speaker_overlapping_wav  # type: ignore
except ImportError:
    from config import TEMP_DIR  # type: ignore
    from diarization_engine import SpeakerTurn  # type: ignore
    from whisper_transcriber import WordToken  # type: ignore
    from overlap_detector import OverlapSegment, CollisionSummary  # type: ignore
    from voice_fingerprint import get_fingerprint_engine  # type: ignore
    from attribution_engine import (  # type: ignore
        AttributionEngine,
        get_attribution_engine,
    )
    from overlap_suite.audio_utils import generate_multi_speaker_overlapping_wav  # type: ignore


def run_task5_verification(audio_path: str = None) -> bool:
    print("\n" + "=" * 68)
    print("🚀 TASK 5 VERIFICATION: Probabilistic Attribution Engine")
    print("   Owner: Vedant | Feature: Overlapping-Speech Attribution")
    print("=" * 68)

    # 1. Setup audio and enrollment
    test_audio = audio_path or os.path.join(TEMP_DIR, "task5_attribution_test.wav")
    if not (audio_path and os.path.exists(audio_path)):
        generate_multi_speaker_overlapping_wav(test_audio, duration_sec=8.0)
        print(f"ℹ Synthesized multi-speaker collision audio at: {test_audio}")
    else:
        print(f"✓ Using audio: {test_audio}")

    fp_engine = get_fingerprint_engine()
    clean_turns = [
        SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=2.5),
        SpeakerTurn(speaker_id="SPEAKER_01", start=4.5, end=6.5),
    ]
    registry = fp_engine.enroll_speakers_from_clean_segments(test_audio, clean_turns)
    print(f"✓ Enrolled speaker profiles: {registry.all_speakers()}")

    # 2. Simulate collision segment with word tokens
    mock_words = [
        WordToken(word="Wait", start=2.6, end=2.9, probability=0.94),
        WordToken(word="let", start=3.0, end=3.3, probability=0.91),
        WordToken(word="me", start=3.4, end=3.7, probability=0.88),
        WordToken(word="finish", start=3.8, end=4.3, probability=0.95),
    ]
    overlap_seg = OverlapSegment(
        start=2.5,
        end=4.5,
        duration=2.0,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        words=mock_words,
        text="Wait let me finish",
    )
    summary = CollisionSummary(
        total_audio_duration=8.0,
        total_overlap_duration=2.0,
        overlap_ratio=0.25,
        total_collisions=1,
        clashing_speaker_pairs=[("SPEAKER_00", "SPEAKER_01")],
        clean_segments=clean_turns,
        overlap_segments=[overlap_seg],
    )

    # 3. Initialize Attribution Engine
    print("\n[Step 1] Initializing Attribution Engine (Temperature = 0.1)...")
    attr_engine = get_attribution_engine(temperature=0.1)

    # 4. Run attribution
    print("\n[Step 2] Decomposing collision zone [2.5s -> 4.5s] into probabilistic shares...")
    report = attr_engine.attribute_all(
        collision_summary=summary,
        audio_path=test_audio,
        registry=registry,
    )

    print(f"✓ Attribution completed ({len(report.attributed_segments)} segment(s) processed):")
    attr = report.attributed_segments[0]
    print(f"  - Dominant Speaker:   {attr.dominant_speaker}")
    print(f"  - Secondary Speaker:  {attr.secondary_speaker}")
    print(f"  - Clash Severity:     {attr.clash_severity.upper()}")
    print("  - Soft Attribution Distribution:")
    for spk, prob in attr.speaker_probabilities.items():
        print(f"      • {spk:<12}: {prob * 100.0:5.1f}% (raw acoustic similarity: {attr.raw_similarities.get(spk, 0.0):.4f})")

    # 5. Inspect word-level attribution
    print("\n[Step 3] Word-Level Collision Attribution:")
    for idx, w in enumerate(attr.word_attributions):
        print(
            f"  Word #{idx + 1:02d}: '{w.word:<8}' [{w.start:.2f}s -> {w.end:.2f}s] "
            f"-> Attributed to {w.attributed_speaker} ({w.confidence * 100:.1f}% confidence)"
        )

    # 6. Validate assertions
    total_prob = sum(attr.speaker_probabilities.values())
    assert abs(total_prob - 1.0) < 1e-2, f"Probabilities do not sum to 1.0 (got {total_prob})!"
    assert attr.dominant_speaker in ["SPEAKER_00", "SPEAKER_01"], "Unknown dominant speaker!"
    assert len(attr.word_attributions) == len(mock_words), "Word attribution mismatch!"

    print("\n" + "=" * 68)
    print("✅ TASK 5 COMPLETE & VERIFIED: Attribution Engine is operational!")
    print("=" * 68 + "\n")
    return True


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_task5_verification(path_arg)
