"""
AirScribe AI Server — Speaker Diarization
============================================
Integrates pyannote.audio for detecting WHO is speaking WHEN.

Purpose:
    Given an audio file, produces a timeline of speaker segments:
    [(start_sec, end_sec, "SPEAKER_00"), (start_sec, end_sec, "SPEAKER_01"), ...]

    This tells us which time ranges belong to which speaker, but does NOT
    tell us the speaker's real name — that's the job of speaker_recognition.py.

How it connects:
    Called by pipeline.py after audio conversion but before Whisper transcription.
    The diarization output is used to assign speaker labels to Whisper segments.

How to configure:
    Set HF_TOKEN in .env with a HuggingFace token that has accepted the
    pyannote model license at https://huggingface.co/pyannote/speaker-diarization-3.1

How to debug:
    If pyannote is not installed or HF_TOKEN is missing, the module
    automatically falls back to simple_diarization_fallback() which uses
    silence gaps to alternate speakers (same as the original code).
"""

import time
import threading
import logging
from typing import Optional

from ai_server.config import HF_TOKEN

logger = logging.getLogger("airscribe.ai.diarization")

# ── Lazy-loaded pipeline ──
_diarization_pipeline = None
_diarization_lock = threading.Lock()
_diarization_mode = "not_loaded"  # "pyannote", "fallback", or "not_loaded"


def get_diarization_pipeline():
    """
    Lazy-load the pyannote speaker diarization pipeline.

    Falls back to a sentinel value "fallback" if:
        - pyannote.audio is not installed
        - HF_TOKEN is not set
        - The pipeline fails to load

    Returns:
        The pyannote Pipeline instance, or the string "fallback".
    """
    global _diarization_pipeline, _diarization_mode
    if _diarization_pipeline is None:
        with _diarization_lock:
            if _diarization_pipeline is None:
                try:
                    from pyannote.audio import Pipeline
                    if not HF_TOKEN:
                        logger.warning(
                            "HF_TOKEN not set — diarization will use silence-gap fallback. "
                            "Set HF_TOKEN in .env for pyannote speaker diarization."
                        )
                        _diarization_pipeline = "fallback"
                        _diarization_mode = "fallback"
                        return _diarization_pipeline

                    logger.info("Loading pyannote speaker diarization pipeline...")
                    start = time.time()

                    _diarization_pipeline = Pipeline.from_pretrained(
                        "pyannote/speaker-diarization-3.1",
                        use_auth_token=HF_TOKEN
                    )

                    # Move to GPU if available
                    try:
                        import torch
                        if torch.cuda.is_available():
                            _diarization_pipeline = _diarization_pipeline.to(
                                torch.device("cuda")
                            )
                            logger.info("Diarization pipeline moved to GPU")
                    except Exception:
                        logger.info("Diarization pipeline running on CPU")

                    elapsed = round(time.time() - start, 1)
                    logger.info(f"Diarization pipeline loaded in {elapsed}s")
                    _diarization_mode = "pyannote"

                except ImportError:
                    logger.warning(
                        "pyannote.audio not installed — using silence-gap fallback. "
                        "Install with: pip install pyannote.audio"
                    )
                    _diarization_pipeline = "fallback"
                    _diarization_mode = "fallback"
                except Exception as e:
                    logger.error(f"Diarization pipeline error: {e} — using fallback")
                    _diarization_pipeline = "fallback"
                    _diarization_mode = "fallback"

    return _diarization_pipeline


def get_mode() -> str:
    """Return the current diarization mode: 'pyannote', 'fallback', or 'not_loaded'."""
    return _diarization_mode


def is_loaded() -> bool:
    """Check if the diarization pipeline is loaded (either pyannote or fallback)."""
    return _diarization_pipeline is not None


def diarize_audio(
    audio_path: str,
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None
) -> list[dict]:
    """
    Perform speaker diarization on an audio file.

    Args:
        audio_path: Path to the audio file (WAV, 16kHz mono recommended).
        num_speakers: Exact number of speakers (if known).
        min_speakers: Minimum number of speakers (optional hint).
        max_speakers: Maximum number of speakers (optional hint).

    Returns:
        List of dicts, each containing:
            - "start": float (start time in seconds)
            - "end": float (end time in seconds)
            - "speaker": str (e.g., "SPEAKER_00", "SPEAKER_01")

        Returns an empty list if diarization fails completely.
    """
    pipeline = get_diarization_pipeline()

    if pipeline == "fallback" or pipeline is None:
        # Will be handled by the caller using Whisper segments + simple fallback
        logger.debug("Using fallback diarization (no pyannote)")
        return []

    try:
        logger.debug(f"Diarizing: {audio_path}")
        start = time.time()

        # Build kwargs for the pipeline
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = num_speakers
        elif min_speakers is not None or max_speakers is not None:
            if min_speakers is not None:
                kwargs["min_speakers"] = min_speakers
            if max_speakers is not None:
                kwargs["max_speakers"] = max_speakers

        diarization_result = pipeline(audio_path, **kwargs)

        # Extract segments
        segments = []
        for turn, _, speaker_label in diarization_result.itertracks(yield_label=True):
            segments.append({
                "start": round(turn.start, 3),
                "end": round(turn.end, 3),
                "speaker": speaker_label
            })

        elapsed = round(time.time() - start, 2)
        unique_speakers = set(s["speaker"] for s in segments)
        logger.info(
            f"Diarized {len(segments)} segments, "
            f"{len(unique_speakers)} speakers in {elapsed}s"
        )

        return segments

    except Exception as e:
        logger.error(f"Diarization error: {e}")
        return []


def simple_diarization_fallback(
    whisper_segments: list[dict],
    num_speakers: int = 2
) -> list[dict]:
    """
    Simple energy-based speaker diarization fallback.

    When pyannote is not available, uses silence gaps between Whisper
    segments to alternate speakers. This is the same algorithm that was
    in the original ai_meeting_server.py.

    Args:
        whisper_segments: List of Whisper segment dicts with 'start', 'end', 'text'.
        num_speakers: Number of speakers to alternate between.

    Returns:
        List of dicts with 'speaker', 'start', 'end', 'text' keys.
    """
    if not whisper_segments:
        return []

    result = []
    current_speaker = 0
    last_end_time = 0.0

    for seg in whisper_segments:
        start_time = seg.get("start", 0.0)
        end_time = seg.get("end", 0.0)
        text = seg.get("text", "").strip()

        if not text:
            continue

        # Detect speaker change based on silence gap
        gap = start_time - last_end_time
        if gap > 1.5:  # Gap > 1.5 seconds suggests speaker change
            current_speaker = (current_speaker + 1) % num_speakers

        result.append({
            "speaker": f"SPEAKER_{current_speaker:02d}",
            "start": start_time,
            "end": end_time,
            "text": text
        })
        last_end_time = end_time

    return result


def map_whisper_to_diarization(
    whisper_segments: list[dict],
    diarization_segments: list[dict]
) -> list[dict]:
    """
    Map each Whisper transcription segment to the diarization speaker
    that overlaps with it the most.

    Args:
        whisper_segments: Whisper output segments with 'start', 'end', 'text'.
        diarization_segments: Diarization output with 'start', 'end', 'speaker'.

    Returns:
        List of dicts with 'speaker', 'start', 'end', 'text' keys.
        Each Whisper segment gets assigned the speaker label with the
        maximum temporal overlap.
    """
    if not diarization_segments:
        return []

    result = []
    for seg in whisper_segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        text = seg.get("text", "").strip()

        if not text:
            continue

        # Find the diarization speaker with the most overlap
        best_speaker = None
        best_overlap = 0.0

        for d_seg in diarization_segments:
            overlap_start = max(seg_start, d_seg["start"])
            overlap_end = min(seg_end, d_seg["end"])
            overlap = max(0.0, overlap_end - overlap_start)

            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = d_seg["speaker"]

        if best_speaker is None:
            best_speaker = "SPEAKER_00"

        result.append({
            "speaker": best_speaker,
            "start": round(seg_start, 2),
            "end": round(seg_end, 2),
            "text": text
        })

    return result
