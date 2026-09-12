"""
Audio Utilities for Overlapping-Speech Attribution
==================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution Suite

Provides:
  - Sub-second precision WAV slicing (dependency-free using stdlib `wave`)
  - Audio sample loading and normalization to float32
  - Synthetic multi-speaker overlapping speech generator with ground-truth clash intervals
"""

import os
import math
import struct
import wave
from typing import Tuple, List, Dict, Any, Optional

try:
    import numpy as np  # type: ignore
except ImportError:
    np = None


def slice_wav_file(
    input_wav_path: str,
    start_sec: float,
    end_sec: float,
    output_wav_path: Optional[str] = None
) -> str:
    """
    Extract a sub-second time interval [start_sec, end_sec] from a WAV file.
    Does not require external FFmpeg or heavy dependencies; uses stdlib wave.
    
    Args:
        input_wav_path: Path to the source WAV file.
        start_sec: Beginning of slice in seconds.
        end_sec: End of slice in seconds.
        output_wav_path: Optional path to save slice. If None, auto-named in same dir.

    Returns:
        Absolute path to the sliced WAV file.
    """
    if not os.path.exists(input_wav_path):
        raise FileNotFoundError(f"Source audio file not found: {input_wav_path}")

    start_sec = max(0.0, float(start_sec))
    end_sec = max(start_sec, float(end_sec))

    if output_wav_path is None:
        base, ext = os.path.splitext(input_wav_path)
        output_wav_path = f"{base}_slice_{int(start_sec*1000)}ms_{int(end_sec*1000)}ms.wav"

    os.makedirs(os.path.dirname(os.path.abspath(output_wav_path)), exist_ok=True)

    with wave.open(input_wav_path, "rb") as src:
        channels = src.getnchannels()
        sampwidth = src.getsampwidth()
        framerate = src.getframerate()
        total_frames = src.getnframes()

        start_frame = min(int(start_sec * framerate), total_frames)
        end_frame = min(int(end_sec * framerate), total_frames)
        frames_to_read = max(0, end_frame - start_frame)

        src.setpos(start_frame)
        raw_frames = src.readframes(frames_to_read)

    with wave.open(output_wav_path, "wb") as dst:
        dst.setnchannels(channels)
        dst.setsampwidth(sampwidth)
        dst.setframerate(framerate)
        dst.writeframes(raw_frames)

    return output_wav_path


def read_wav_samples(wav_path: str) -> Tuple[Any, int]:
    """
    Read a WAV file and return a 1D float32 numpy array in range [-1.0, 1.0] and sample rate.
    If stereo, downmixes to mono. Fallbacks safely if numpy is unavailable.
    """
    with wave.open(wav_path, "rb") as w:
        channels = w.getnchannels()
        sampwidth = w.getsampwidth()
        framerate = w.getframerate()
        nframes = w.getnframes()
        raw_data = w.readframes(nframes)

    if sampwidth == 2:
        fmt = f"<{nframes * channels}h"
        int_data = struct.unpack(fmt, raw_data)
        max_val = 32768.0
    elif sampwidth == 1:
        fmt = f"<{nframes * channels}B"
        int_data = [x - 128 for x in struct.unpack(fmt, raw_data)]
        max_val = 128.0
    else:
        # 24-bit or 32-bit fallback: take 16-bit MSB
        int_data = []
        bytes_per_sample = sampwidth
        for i in range(0, len(raw_data), bytes_per_sample):
            chunk = raw_data[i : i + bytes_per_sample]
            if len(chunk) == bytes_per_sample:
                val = int.from_bytes(chunk, byteorder="little", signed=True)
                int_data.append(val)
        max_val = float(1 << (sampwidth * 8 - 1))

    if np is not None:
        arr = np.array(int_data, dtype=np.float32) / max_val
        if channels > 1:
            arr = arr.reshape(-1, channels).mean(axis=1)
        return arr, framerate
    else:
        # Pure python fallback list
        normalized = [x / max_val for x in int_data]
        if channels > 1:
            mono = [sum(normalized[i : i + channels]) / channels for i in range(0, len(normalized), channels)]
            return mono, framerate
        return normalized, framerate


def write_wav_samples(output_path: str, samples: Any, sample_rate: int = 16000) -> str:
    """Write 1D float samples [-1.0, 1.0] to a 16-bit mono WAV file."""
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with wave.open(output_path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)

        frames = bytearray()
        if np is not None and isinstance(samples, np.ndarray):
            clipped = np.clip(samples, -1.0, 1.0) * 32767.0
            int_samples = clipped.astype(np.int16)
            frames = int_samples.tobytes()
        else:
            for s in samples:
                val = max(-1.0, min(1.0, float(s)))
                ival = int(val * 32767.0)
                frames.extend(struct.pack("<h", ival))

        w.writeframes(frames)
    return output_path


def generate_multi_speaker_overlapping_wav(
    filepath: str,
    duration_sec: float = 8.0,
    sample_rate: int = 16000
) -> Dict[str, Any]:
    """
    Synthesize realistic 2-speaker multi-modal test audio with known collision intervals:
      - 0.0s - 2.5s: Speaker 0 alone (low resonant voice ~140Hz + harmonics)
      - 2.5s - 4.5s: CLASH / OVERLAP: Both Speaker 0 and Speaker 1 speak concurrently!
      - 4.5s - 6.5s: Speaker 1 alone (higher resonant voice ~260Hz + harmonics)
      - 6.5s - 7.0s: Silence pause
      - 7.0s - 8.0s: Speaker 0 alone (~140Hz)

    Returns:
        Dictionary with audio file path and ground-truth intervals for evaluation.
    """
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
    num_samples = int(duration_sec * sample_rate)

    # Audio synthesis with distinctive speaker formants/harmonics
    frames = bytearray()
    for i in range(num_samples):
        t = i / sample_rate

        # Speaker 0: Low voice ~140Hz with 2nd and 3rd harmonics
        spk0_active = (0.0 <= t < 4.5) or (7.0 <= t <= 8.0)
        s0 = 0.0
        if spk0_active:
            s0 = (
                0.55 * math.sin(2 * math.pi * 140.0 * t) +
                0.25 * math.sin(2 * math.pi * 280.0 * t) +
                0.15 * math.sin(2 * math.pi * 420.0 * t)
            )
            # Add subtle syllabic envelope modulation (4 syllables per second)
            s0 *= 0.7 + 0.3 * math.sin(2 * math.pi * 4.0 * t)

        # Speaker 1: Higher voice ~260Hz with 2nd and 3rd harmonics
        spk1_active = 2.5 <= t < 6.5
        s1 = 0.0
        if spk1_active:
            s1 = (
                0.55 * math.sin(2 * math.pi * 260.0 * t) +
                0.25 * math.sin(2 * math.pi * 520.0 * t) +
                0.15 * math.sin(2 * math.pi * 780.0 * t)
            )
            # Add subtle syllabic envelope modulation (5 syllables per second)
            s1 *= 0.7 + 0.3 * math.sin(2 * math.pi * 5.0 * t)

        # Mixed signal with soft limiting
        mixed = s0 + s1
        clamped = max(-0.95, min(0.95, mixed))
        sample_val = int(clamped * 32000.0)
        frames.extend(struct.pack("<h", sample_val))

    with wave.open(filepath, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(frames)

    ground_truth = {
        "audio_path": filepath,
        "duration": duration_sec,
        "sample_rate": sample_rate,
        "speakers": ["SPEAKER_00", "SPEAKER_01"],
        "clean_intervals": [
            {"start": 0.0, "end": 2.5, "speaker": "SPEAKER_00"},
            {"start": 4.5, "end": 6.5, "speaker": "SPEAKER_01"},
            {"start": 7.0, "end": 8.0, "speaker": "SPEAKER_00"},
        ],
        "overlap_intervals": [
            {
                "start": 2.5,
                "end": 4.5,
                "duration": 2.0,
                "speakers": ["SPEAKER_00", "SPEAKER_01"]
            }
        ]
    }
    return ground_truth
