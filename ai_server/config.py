"""
AirScribe AI Server — Configuration
=====================================
Centralized configuration for all AI server modules.
Reads from environment variables with sensible defaults.

How it connects:
    Every other module in ai_server/ imports from here instead of
    reading os.environ directly. This keeps config in one place.

How to configure:
    Set values in a .env file at the project root, or export them
    as environment variables before starting the server.

How to debug:
    Call `print_config()` at startup to dump all active settings.
"""

import os
from dotenv import load_dotenv

load_dotenv()


# ── Whisper ──
WHISPER_MODEL_SIZE: str = os.getenv("WHISPER_MODEL", "base")
# Options: tiny, base, small, medium, large
# base is the best balance for RTX 3050 (4GB VRAM)

# ── HuggingFace ──
HF_TOKEN: str = os.getenv("HF_TOKEN", "")
# Required for pyannote.audio models. Get a free token at https://huggingface.co/settings/tokens

# ── MongoDB ──
MONGO_URI: str = os.getenv("MONGO_URI", "")
DB_NAME: str = os.getenv("DB_NAME", "airscribe")

# ── Speaker Recognition ──
SPEAKER_CONFIDENCE_THRESHOLD: float = float(os.getenv("SPEAKER_CONFIDENCE_THRESHOLD", "0.70"))
# Cosine similarity threshold for matching a speaker to an enrolled profile.
# >= 0.70 → label with enrolled name
# < 0.70 → label as "Unknown Speaker"

# ── Voice Enrollment ──
MIN_ENROLLMENT_DURATION: int = int(os.getenv("MIN_ENROLLMENT_DURATION", "15"))
# Minimum audio duration (seconds) for a valid voice enrollment.
# SpeechBrain produces usable embeddings from ~10-15s; we default to 15.

MAX_AUDIO_SIZE_MB: int = int(os.getenv("MAX_AUDIO_SIZE_MB", "10"))
# Maximum upload size for audio files (in megabytes).

ENROLLMENT_SAMPLE_RATE: int = 16000
# Audio sample rate for processing (Whisper and SpeechBrain both expect 16kHz).

# ── Server ──
AI_HOST: str = os.getenv("AI_HOST", "0.0.0.0")
AI_PORT: int = int(os.getenv("AI_PORT", "5001"))
AI_DEBUG: bool = os.getenv("AI_DEBUG", "True").lower() == "true"

# ── Audio Processing ──
UPLOAD_FOLDER: str = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "temp_audio"
)
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Chunk Processing ──
CHUNK_INTERVAL_MS: int = 8000  # Matches frontend CHUNK_INTERVAL_MS

# ── Encrypted Note Store (Feature #6) ──
NOTE_ENCRYPTION_MASTER_KEY: str = os.getenv("NOTE_ENCRYPTION_MASTER_KEY", "")
NOTE_ENCRYPTION_SALT: str = os.getenv("NOTE_ENCRYPTION_SALT", "")
NOTE_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
NOTE_EMBEDDING_DIM: int = 384


def print_config():
    """Print all active configuration values. Useful for startup diagnostics."""
    print("=" * 55)
    print("  AirScribe AI Server — Configuration")
    print("=" * 55)
    print(f"  Whisper model:        {WHISPER_MODEL_SIZE}")
    print(f"  HF_TOKEN set:         {'Yes' if HF_TOKEN else 'No (diarization will use fallback)'}")
    print(f"  MongoDB:              {DB_NAME} ({'configured' if MONGO_URI else 'NOT SET'})")
    print(f"  Confidence threshold: {SPEAKER_CONFIDENCE_THRESHOLD}")
    print(f"  Min enrollment:       {MIN_ENROLLMENT_DURATION}s")
    print(f"  Max audio size:       {MAX_AUDIO_SIZE_MB}MB")
    print(f"  Upload folder:        {UPLOAD_FOLDER}")
    print(f"  Server:               {AI_HOST}:{AI_PORT} (debug={AI_DEBUG})")
    print(f"  Note encryption:      {'CONFIGURED' if NOTE_ENCRYPTION_MASTER_KEY else 'NOT SET (notes will be unencrypted)'}")
    print(f"  Embedding model:      {NOTE_EMBEDDING_MODEL} ({NOTE_EMBEDDING_DIM}-dim)")
    print("=" * 55)
