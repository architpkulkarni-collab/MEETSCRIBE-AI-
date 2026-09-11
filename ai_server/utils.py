"""
AirScribe AI Server — Utilities
=================================
Shared helper functions used across all AI server modules.

Purpose:
    Audio format conversion, file validation, duration measurement,
    and logging setup. Avoids duplicating these helpers in every module.

How it connects:
    Imported by whisper_service, diarization, speaker_recognition,
    voice_enrollment, and pipeline modules.

How to debug:
    All functions log via the standard `logging` module at DEBUG level.
    Set AI_DEBUG=True or use --log-level=DEBUG to see verbose output.
"""

import os
import logging
import subprocess
import struct
import wave
from typing import Optional

from ai_server.config import UPLOAD_FOLDER, MAX_AUDIO_SIZE_MB

# ── Logging ──
logger = logging.getLogger("airscribe.ai")


def setup_logging(debug: bool = False):
    """Configure structured logging for the AI server."""
    level = logging.DEBUG if debug else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    logging.basicConfig(level=level, format=fmt)
    logger.setLevel(level)


# ══════════════════════════════════════════════════════
#  AUDIO CONVERSION
# ══════════════════════════════════════════════════════

def convert_to_wav(input_path: str) -> str:
    """
    Convert an audio file to 16kHz mono WAV format.

    Tries ffmpeg first (most reliable), then pydub as fallback.
    Returns the path to the converted WAV file, or the original
    path if conversion fails.

    Args:
        input_path: Path to the input audio file.

    Returns:
        Path to the WAV file (may be same as input if already WAV
        or conversion failed).
    """
    # If already a wav, check if it needs resampling
    if input_path.lower().endswith(".wav"):
        try:
            with wave.open(input_path, "rb") as wf:
                if wf.getframerate() == 16000 and wf.getnchannels() == 1:
                    return input_path  # Already in the right format
        except Exception:
            pass  # Corrupted wav, try re-encoding

    output_path = input_path.rsplit(".", 1)[0] + "_converted.wav"

    # Try ffmpeg first
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y", "-i", input_path,
                "-ar", "16000",   # 16kHz sample rate
                "-ac", "1",       # Mono
                "-f", "wav",
                output_path
            ],
            capture_output=True,
            timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path):
            logger.debug(f"Converted to WAV via ffmpeg: {output_path}")
            return output_path
        else:
            logger.warning(f"ffmpeg conversion failed: {result.stderr.decode()[:200]}")
    except FileNotFoundError:
        logger.warning("ffmpeg not found on PATH — trying pydub fallback")
    except subprocess.TimeoutExpired:
        logger.warning("ffmpeg conversion timed out")

    # Try pydub as fallback
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        if os.path.exists(output_path):
            logger.debug(f"Converted to WAV via pydub: {output_path}")
            return output_path
    except ImportError:
        logger.warning("pydub not installed — cannot convert audio")
    except Exception as e:
        logger.warning(f"pydub conversion error: {e}")

    # Return original if conversion fails
    logger.warning(f"Could not convert {input_path} to WAV — using original")
    return input_path


def get_audio_duration(audio_path: str) -> float:
    """
    Get the duration of an audio file in seconds.

    Tries multiple methods: wave module, ffprobe, pydub.

    Args:
        audio_path: Path to the audio file.

    Returns:
        Duration in seconds, or 0.0 if it cannot be determined.
    """
    # Method 1: wave module (only works for WAV files)
    if audio_path.lower().endswith(".wav"):
        try:
            with wave.open(audio_path, "rb") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                if rate > 0:
                    return frames / float(rate)
        except Exception:
            pass

    # Method 2: ffprobe
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                audio_path
            ],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            return float(result.stdout.strip())
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass

    # Method 3: pydub
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_path)
        return len(audio) / 1000.0
    except Exception:
        pass

    logger.warning(f"Could not determine duration of {audio_path}")
    return 0.0


# ══════════════════════════════════════════════════════
#  FILE VALIDATION
# ══════════════════════════════════════════════════════

def validate_audio_file(
    file_storage,
    max_size_mb: int = MAX_AUDIO_SIZE_MB,
    allowed_extensions: Optional[set] = None
) -> tuple[bool, str]:
    """
    Validate an uploaded audio file.

    Checks:
        - File is present and non-empty.
        - File size is within the limit.
        - File extension is allowed.

    Args:
        file_storage: Flask's FileStorage object (from request.files).
        max_size_mb: Maximum file size in megabytes.
        allowed_extensions: Set of allowed extensions (without dots).
            Defaults to {'wav', 'webm', 'ogg', 'opus', 'mp3', 'flac', 'm4a'}.

    Returns:
        Tuple of (is_valid, error_message). If valid, error_message is empty.
    """
    if allowed_extensions is None:
        allowed_extensions = {"wav", "webm", "ogg", "opus", "mp3", "flac", "m4a"}

    if not file_storage or not file_storage.filename:
        return False, "No audio file provided"

    # Check extension
    ext = file_storage.filename.rsplit(".", 1)[-1].lower() if "." in file_storage.filename else ""
    if ext not in allowed_extensions:
        return False, f"Invalid file type '.{ext}'. Allowed: {', '.join(sorted(allowed_extensions))}"

    # Check file size
    file_storage.seek(0, 2)  # Seek to end
    size_bytes = file_storage.tell()
    file_storage.seek(0)     # Reset to beginning

    if size_bytes == 0:
        return False, "Audio file is empty"

    max_bytes = max_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        size_mb = round(size_bytes / (1024 * 1024), 1)
        return False, f"File too large ({size_mb}MB). Maximum is {max_size_mb}MB"

    return True, ""


def cleanup_temp_files(*paths: str):
    """
    Remove temporary files, ignoring errors silently.

    Args:
        *paths: Variable number of file paths to delete.
    """
    for path in paths:
        if path and os.path.exists(path):
            try:
                os.remove(path)
                logger.debug(f"Cleaned up temp file: {path}")
            except Exception as e:
                logger.debug(f"Could not remove temp file {path}: {e}")
