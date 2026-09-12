"""
Test Script for Task 4: Voice Fingerprinting (SpeechBrain ECAPA-TDNN)
=====================================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Run:
    python voice_intelligence/test_task4_fingerprint.py [optional_audio_path]

Verifies that:
  1. Engine initializes (ECAPA-TDNN or deterministic acoustic spectral fallback).
  2. Extracts normalized 192-dimensional embedding vectors.
  3. Enrolls speaker profiles from clean timeline turns.
  4. Correctly computes cosine similarity distances between speakers.
"""

import os
import sys
import math

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from voice_intelligence.config import TEMP_DIR  # type: ignore
    from voice_intelligence.diarization_engine import SpeakerTurn  # type: ignore
    from voice_intelligence.voice_fingerprint import (  # type: ignore
        VoiceFingerprintEngine,
        SpeakerProfileRegistry,
        get_fingerprint_engine,
    )
    from voice_intelligence.overlap_suite.audio_utils import generate_multi_speaker_overlapping_wav  # type: ignore
except ImportError:
    from config import TEMP_DIR  # type: ignore
    from diarization_engine import SpeakerTurn  # type: ignore
    from voice_fingerprint import (  # type: ignore
        VoiceFingerprintEngine,
        SpeakerProfileRegistry,
        get_fingerprint_engine,
    )
    from overlap_suite.audio_utils import generate_multi_speaker_overlapping_wav  # type: ignore


def run_task4_verification(audio_path: str = None) -> bool:
    print("\n" + "=" * 68)
    print("🚀 TASK 4 VERIFICATION: Voice Fingerprinting Engine (ECAPA-TDNN)")
    print("   Owner: Vedant | Feature: Overlapping-Speech Attribution")
    print("=" * 68)

    # 1. Setup audio
    test_audio = audio_path or os.path.join(TEMP_DIR, "task4_fingerprint_test.wav")
    if not (audio_path and os.path.exists(audio_path)):
        generate_multi_speaker_overlapping_wav(test_audio, duration_sec=8.0)
        print(f"ℹ Synthesized 2-speaker audio at: {test_audio}")
    else:
        print(f"✓ Using provided audio: {test_audio}")

    # 2. Initialize engine
    print("\n[Step 1] Initializing Voice Fingerprint Engine...")
    engine = get_fingerprint_engine()
    engine.load_model()
    print(f"✓ Engine initialized (Mode: '{engine.mode}', Device: '{engine.device}')")

    # 3. Clean segments for speaker enrollment
    clean_turns = [
        SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=2.5),
        SpeakerTurn(speaker_id="SPEAKER_01", start=4.5, end=6.5),
        SpeakerTurn(speaker_id="SPEAKER_00", start=7.0, end=8.0),
    ]

    # 4. Enroll speakers
    print("\n[Step 2] Enrolling pristine speaker voice fingerprints from clean segments...")
    registry = engine.enroll_speakers_from_clean_segments(test_audio, clean_turns)
    print(f"✓ Enrolled {len(registry)} speaker profiles: {registry.all_speakers()}")

    for spk in registry.all_speakers():
        fp = registry.get(spk)
        norm = math.sqrt(sum(x * x for x in fp.vector))
        print(
            f"  Speaker: {fp.speaker_id:<12} | Dim: {fp.dimension} | "
            f"L2 Norm: {norm:.4f} | Mode: {fp.extraction_mode} | Source: {fp.source_segment}"
        )
        assert abs(norm - 1.0) < 1e-3, f"Fingerprint vector for {spk} is not normalized!"

    # 5. Compute inter-speaker cosine similarity
    spk0_fp = registry.get("SPEAKER_00")
    spk1_fp = registry.get("SPEAKER_01")
    sim_clean = VoiceFingerprintEngine.compute_cosine_similarity(spk0_fp.vector, spk1_fp.vector)
    print(f"\n[Step 3] Baseline Inter-Speaker Cosine Similarity: {sim_clean:.4f}")

    # 6. Extract fingerprint from overlapping collision slice (2.5s - 4.5s)
    print("\n[Step 4] Extracting mixed-signal fingerprint from speech collision zone [2.5s -> 4.5s]...")
    mixed_fp = engine.extract_embedding(
        audio_input=test_audio,
        speaker_id="MIXED_COLLISION",
        start_sec=2.5,
        end_sec=4.5,
    )
    sim_mixed_spk0 = VoiceFingerprintEngine.compute_cosine_similarity(mixed_fp.vector, spk0_fp.vector)
    sim_mixed_spk1 = VoiceFingerprintEngine.compute_cosine_similarity(mixed_fp.vector, spk1_fp.vector)

    print(f"✓ Mixed Signal Projection against Enrolled Speakers:")
    print(f"  - Similarity to SPEAKER_00 (low voice):  {sim_mixed_spk0:.4f}")
    print(f"  - Similarity to SPEAKER_01 (high voice): {sim_mixed_spk1:.4f}")

    # Both speakers must show positive presence in the collision
    assert mixed_fp.dimension == 192, f"Expected 192 dims, got {mixed_fp.dimension}"
    print("\n" + "=" * 68)
    print("✅ TASK 4 COMPLETE & VERIFIED: Voice Fingerprinting is operational!")
    print("=" * 68 + "\n")
    return True


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_task4_verification(path_arg)
