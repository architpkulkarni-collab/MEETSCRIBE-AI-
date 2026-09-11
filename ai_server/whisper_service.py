"""
AirScribe AI Server — Whisper Transcription Service
=====================================================
Handles loading and running OpenAI Whisper for speech-to-text.

Purpose:
    Provides a clean interface for transcribing audio files using
    Whisper. Supports lazy model loading with thread safety.

How it connects:
    Called by pipeline.py during the transcription phase.
    Also used standalone by ai_meeting_server.py for backward compat.

How to configure:
    Set WHISPER_MODEL in .env (tiny/base/small/medium/large).
    Default is 'base' — best balance for RTX 3050.

How to debug:
    Enable AI_DEBUG=True to see model loading and transcription timing.
    The transcribe_audio() function logs elapsed time for each call.
"""

import time
import threading
import logging
from typing import Optional

from ai_server.config import WHISPER_MODEL_SIZE

logger = logging.getLogger("airscribe.ai.whisper")

# ── Lazy-loaded model ──
_whisper_model = None
_whisper_lock = threading.Lock()


def get_whisper_model():
    """
    Lazy-load the Whisper model on first use.

    Uses a double-checked locking pattern to ensure thread safety
    without paying the lock cost on every call.

    Returns:
        The loaded Whisper model instance.

    Raises:
        ImportError: If openai-whisper is not installed.
        RuntimeError: If the model fails to load.
    """
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                try:
                    import whisper
                    logger.info(f"Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
                    start = time.time()
                    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
                    elapsed = round(time.time() - start, 1)
                    logger.info(f"Whisper model '{WHISPER_MODEL_SIZE}' loaded in {elapsed}s")
                except ImportError:
                    logger.error("openai-whisper not installed. Run: pip install openai-whisper")
                    raise
                except Exception as e:
                    logger.error(f"Failed to load Whisper model: {e}")
                    raise RuntimeError(f"Whisper model load failed: {e}") from e
    return _whisper_model


def is_loaded() -> bool:
    """Check if the Whisper model is currently loaded in memory."""
    return _whisper_model is not None


def transcribe_audio(
    audio_path: str,
    language: str = "en",
    word_timestamps: bool = True
) -> list[dict]:
    """
    Transcribe an audio file using Whisper.

    Args:
        audio_path: Path to the audio file (WAV preferred, 16kHz mono).
        language: Language code for transcription (default: "en").
        word_timestamps: Whether to include word-level timestamps.

    Returns:
        List of segment dicts, each containing:
            - "start": float (start time in seconds)
            - "end": float (end time in seconds)
            - "text": str (transcribed text)

        Returns an empty list if transcription fails or produces no output.
    """
    try:
        model = get_whisper_model()

        logger.debug(f"Transcribing: {audio_path}")
        start = time.time()

        result = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )

        segments = result.get("segments", [])
        elapsed = round(time.time() - start, 2)

        logger.info(
            f"Transcribed {len(segments)} segments in {elapsed}s "
            f"(file: {audio_path.split('/')[-1].split(chr(92))[-1]})"
        )

        return segments

    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return []
