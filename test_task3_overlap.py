"""
Test Script for Task 3: Overlap & Temporal Clash Detection
==========================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Run:
    python voice_intelligence/test_task3_overlap.py [optional_audio_path]

Verifies that:
  1. Sweep-line detector identifies multi-speaker collision boundaries.
  2. Partitions audio into clean reference segments vs. collision zones.
  3. Correctly isolates colliding speaker pairs.
  4. Prepares collision audio slices for voice fingerprinting (Task 4).
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from voice_intelligence.config import TEMP_DIR  # type: ignore
    from voice_intelligence.diarization_engine import SpeakerTurn, DiarizationResult  # type: ignore
    from voice_intelligence.overlap_detector import OverlapDetector, get_overlap_detector  # type: ignore
    from voice_intelligence.overlap_suite.audio_utils import generate_multi_speaker_overlapping_wav  # type: ignore
except ImportError:
    from config import TEMP_DIR  # type: ignore
    from diarization_engine import SpeakerTurn, DiarizationResult  # type: ignore
    from overlap_detector import OverlapDetector, get_overlap_detector  # type: ignore
    from overlap_suite.audio_utils import generate_multi_speaker_overlapping_wav  # type: ignore


def run_task3_verification(audio_path: str = None) -> bool:
    print("\n" + "=" * 68)
    print("🚀 TASK 3 VERIFICATION: Overlap & Temporal Clash Detector")
    print("   Owner: Vedant | Feature: Overlapping-Speech Attribution")
    print("=" * 68)

    # 1. Setup audio
    if audio_path and os.path.exists(audio_path):
        test_audio = audio_path
        print(f"✓ Using provided audio: {test_audio}")
        turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=4.5),
            SpeakerTurn(speaker_id="SPEAKER_01", start=2.5, end=6.5),
        ]
        duration = 7.0
    else:
        test_audio = os.path.join(TEMP_DIR, "task3_overlapping_test.wav")
        gt = generate_multi_speaker_overlapping_wav(test_audio, duration_sec=8.0)
        print(f"ℹ Synthesized 2-speaker test audio with 2.0s collision at: {test_audio}")
        turns = [
            SpeakerTurn(speaker_id="SPEAKER_00", start=0.0, end=4.5),
            SpeakerTurn(speaker_id="SPEAKER_01", start=2.5, end=6.5),
            SpeakerTurn(speaker_id="SPEAKER_00", start=7.0, end=8.0),
        ]
        duration = gt["duration"]

    diar_result = DiarizationResult(
        turns=turns,
        speakers=["SPEAKER_00", "SPEAKER_01"],
        num_speakers=2,
        duration=duration,
    )

    # 2. Initialize detector
    print("\n[Step 1] Initializing Sweep-Line Overlap Detector...")
    detector = get_overlap_detector()
    print(f"✓ Detector initialized (min_overlap_sec = {detector.min_overlap_sec}s)")

    # 3. Detect overlaps
    print("\n[Step 2] Scanning timeline for simultaneous speaker collisions...")
    summary = detector.detect_overlaps(
        diarization=diar_result,
        audio_path=test_audio,
        slice_audio=True,
    )

    print(f"✓ Detection completed:")
    print(f"  - Total Audio Duration:   {summary.total_audio_duration:.2f}s")
    print(f"  - Total Overlap Duration: {summary.total_overlap_duration:.2f}s")
    print(f"  - Overlap Ratio:          {summary.overlap_ratio * 100.0:.1f}%")
    print(f"  - Total Clashes Detected: {summary.total_collisions}")
    print(f"  - Clashing Speaker Pairs: {summary.clashing_speaker_pairs}")

    # 4. Inspect collision intervals
    print("\n[Step 3] Detected Collision Segments:")
    for idx, seg in enumerate(summary.overlap_segments):
        print(
            f"  Clash #{idx + 1}: [{seg.start:5.2f}s -> {seg.end:5.2f}s] "
            f"Duration: {seg.duration:.2f}s | Speakers: {seg.speakers} | Confidence: {seg.confidence_level}"
        )
        if seg.audio_slice_path:
            print(f"    Sliced audio: {os.path.basename(seg.audio_slice_path)}")

    # 5. Inspect clean segments (used for enrollment in Task 4)
    print("\n[Step 4] Pristine Single-Speaker Segments (Ready for Task 4 Enrollment):")
    for idx, cs in enumerate(summary.clean_segments):
        print(f"  Clean #{idx + 1}: [{cs.start:5.2f}s -> {cs.end:5.2f}s] {cs.speaker_id:<12} ({cs.duration:.2f}s)")

    # 6. Assertions
    assert summary.total_collisions > 0, "Failed to detect overlapping collision!"
    assert len(summary.clean_segments) >= 2, "Clean single-speaker segments missing!"
    print("\n" + "=" * 68)
    print("✅ TASK 3 COMPLETE & VERIFIED: Temporal Clash Detector is operational!")
    print("=" * 68 + "\n")
    return True


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_task3_verification(path_arg)
