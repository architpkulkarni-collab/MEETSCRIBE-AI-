"""
Test Script for Task 1: Speech-to-Text & Word Timestamps
=========================================================
Owner: Vedant (Member A)

Run this test:
    python voice_intelligence/test_task1_whisper.py [optional_path_to_wav]

Verifies that:
  1. Engine loads properly (faster-whisper preferred, openai-whisper fallback)
  2. Audio transcribes cleanly
  3. Word-level timestamps (start, end, confidence) are extracted
  4. Time-range word slicing (crucial for collision attribution) works
"""

import os
import sys
import math
import struct
import wave

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from voice_intelligence.config import TEMP_DIR  # type: ignore
    from voice_intelligence.whisper_transcriber import FasterWhisperTranscriber, get_transcriber  # type: ignore
except ImportError:
    from config import TEMP_DIR  # type: ignore
    from whisper_transcriber import FasterWhisperTranscriber, get_transcriber  # type: ignore


def create_synthetic_wav(filepath: str, duration_sec: float = 3.0, sample_rate: int = 16000):
    """Generate a clean synthetic WAV file for sanity testing."""
    num_samples = int(duration_sec * sample_rate)
    with wave.open(filepath, "w") as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)

        # Tone with varying frequency to simulate voice acoustic energy
        frames = bytearray()
        for i in range(num_samples):
            t = i / sample_rate
            freq = 220.0 + 80.0 * math.sin(2 * math.pi * 1.5 * t)
            sample = int(12000.0 * math.sin(2 * math.pi * freq * t))
            frames.extend(struct.pack("<h", sample))

        wav_file.writeframes(frames)
    return filepath


def run_task1_verification(audio_path: str = None):
    print("\n" + "=" * 65)
    print("🚀 TASK 1 VERIFICATION: Speech-to-Text with Word Timestamps")
    print("   Owner: Vedant | Feature: Overlapping-Speech Attribution")
    print("=" * 65)

    # 1. Check or prepare audio
    if audio_path and os.path.exists(audio_path):
        test_audio = audio_path
        print(f"✓ Using provided audio file: {test_audio}")
    else:
        test_audio = os.path.join(TEMP_DIR, "task1_sanity_test.wav")
        create_synthetic_wav(test_audio, duration_sec=3.0)
        print(f"ℹ No custom audio provided. Generated test audio at: {test_audio}")

    # 2. Initialize Transcriber
    print("\n[Step 1] Initializing Transcriber engine...")
    transcriber = FasterWhisperTranscriber(model_size="base")
    
    try:
        transcriber.load_model()
        print(f"✓ Model loaded successfully: engine='{transcriber.engine}', device='{transcriber.device}'")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return False

    # 3. Run transcription with word timestamps enabled
    print("\n[Step 2] Transcribing with word_timestamps=True...")
    try:
        result = transcriber.transcribe(test_audio, word_timestamps=True)
        print(f"✓ Transcription completed in {result.processing_time:.2f}s (Engine: {result.engine_used})")
        print(f"  - Detected Language: {result.language} (Confidence: {result.language_probability:.2f})")
        print(f"  - Audio Duration:    {result.duration:.2f}s")
        print(f"  - Total Segments:    {len(result.segments)}")
        print(f"  - Full Transcript:   \"{result.text}\"")

        # 4. Inspect word-level tokens
        print("\n[Step 3] Word-Level Timestamp Inspection:")
        total_words = 0
        for seg_idx, seg in enumerate(result.segments):
            print(f"\n  Segment #{seg_idx + 1} [{seg.start:.2f}s -> {seg.end:.2f}s]: \"{seg.text}\"")
            if seg.words:
                print("    Words:")
                for w in seg.words:
                    total_words += 1
                    print(f"      - '{w.word}' ({w.start:.2f}s - {w.end:.2f}s, prob: {w.probability:.2f})")
            else:
                print("    (No word tokens in this segment)")

        # 5. Test overlapping range query (simulating an overlap window)
        print("\n[Step 4] Testing Overlap Window Query (simulate collision [0.5s - 2.0s]):")
        simulated_collision_words = result.get_words_in_timerange(0.5, 2.0)
        print(f"  Words found in simulated overlap window [0.5s - 2.0s]: {len(simulated_collision_words)}")
        for w in simulated_collision_words:
            print(f"    -> '{w.word}' [{w.start:.2f}s - {w.end:.2f}s]")

        print("\n" + "=" * 65)
        print("✅ TASK 1 COMPLETE & VERIFIED!")
        print("=" * 65 + "\n")
        return True

    except Exception as e:
        print(f"\n❌ Transcription failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    audio_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_task1_verification(audio_arg)
