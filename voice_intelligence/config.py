"""
Configuration for Voice & Speaker Intelligence
==============================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution
"""

import os
import sys
import importlib

# Ensure project root and package directory are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
for p in [_project_root, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

# ── Base Paths ──
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(BASE_DIR, "temp")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BENCHMARK_DIR = os.path.join(BASE_DIR, "benchmarks")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(BENCHMARK_DIR, exist_ok=True)

# ── Hardware Device Detection ──
def get_device() -> str:
    """Detect if CUDA GPU is available, fallback to CPU."""
    try:
        torch = importlib.import_module("torch")
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

DEVICE = get_device()

# ── Task 1: faster-whisper Settings ──
# Available sizes: 'tiny', 'base', 'small', 'medium', 'large-v3'
# 'base' or 'small' gives excellent accuracy and speed
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")
WHISPER_DEVICE = DEVICE
# Use float16 or int8_float16 on GPU, int8 on CPU for maximum efficiency
WHISPER_COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"
WHISPER_BEAM_SIZE = 5
WHISPER_VAD_FILTER = True  # Silero VAD to suppress hallucinated silence
WHISPER_WORD_TIMESTAMPS = True  # Essential for overlapping speech word isolation

# ── Task 2: pyannote Diarization Settings ──
HF_TOKEN = os.getenv("HF_TOKEN", "")
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# ── Task 3 & 4: SpeechBrain Voice Fingerprinting ──
SPEECHBRAIN_MODEL_SOURCE = "speechbrain/spkrec-ecapa-voxceleb"
SPEECHBRAIN_SAVEDIR = os.path.join(MODELS_DIR, "spkrec-ecapa-voxceleb")
SPEECHBRAIN_SIMILARITY_THRESHOLD = float(os.getenv("SPEAKER_CONFIDENCE_THRESHOLD", "0.65"))

# ── Task 5: Overlapping Attribution Tuning ──
# Minimum overlap duration (seconds) to trigger clash analysis
MIN_OVERLAP_DURATION_SEC = 0.25
# Softmax temperature for converting similarity scores to percentage confidence
ATTRIBUTION_TEMPERATURE = 0.1
