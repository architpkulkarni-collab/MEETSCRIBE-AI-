"""
AirScribe AI Server — Speaker Recognition
=============================================
Integrates SpeechBrain for identifying WHO a speaker is by matching
their voice against enrolled voice profiles.

Purpose:
    - Extract speaker embeddings from audio (192-dimensional ECAPA-TDNN vectors).
    - Compare embeddings using cosine similarity.
    - Match unknown speakers against enrolled profiles to produce real names.

How it connects:
    Called by pipeline.py after diarization segments are produced.
    For each speaker segment, we extract an embedding and compare it
    against all enrolled profiles from MongoDB.

    Also called by voice_enrollment.py when a user records their voice
    to generate their enrollment embedding.

How to configure:
    Set SPEAKER_CONFIDENCE_THRESHOLD in .env (default 0.70).
    Higher values = stricter matching (fewer false positives).
    Lower values = more lenient matching (fewer "Unknown Speaker" labels).

How to debug:
    Enable AI_DEBUG=True to see similarity scores for every match attempt.
    The identify_speaker() function logs which profile scored highest.
"""

import time
import threading
import logging
from typing import Optional
import numpy as np

from ai_server.config import SPEAKER_CONFIDENCE_THRESHOLD

logger = logging.getLogger("airscribe.ai.recognition")

# ── Lazy-loaded SpeechBrain encoder ──
_speaker_encoder = None
_encoder_lock = threading.Lock()


def get_speaker_encoder():
    """
    Lazy-load the SpeechBrain ECAPA-TDNN speaker encoder.

    Uses the pretrained 'spkrec-ecapa-voxceleb' model which produces
    192-dimensional embeddings. This model is ~100MB and downloaded
    on first use.

    Returns:
        SpeechBrain EncoderClassifier instance.

    Raises:
        ImportError: If speechbrain is not installed.
        RuntimeError: If the model fails to load.
    """
    global _speaker_encoder
    if _speaker_encoder is None:
        with _encoder_lock:
            if _speaker_encoder is None:
                try:
                    from speechbrain.inference.speaker import EncoderClassifier
                    logger.info("Loading SpeechBrain ECAPA-TDNN speaker encoder...")
                    start = time.time()

                    _speaker_encoder = EncoderClassifier.from_hparams(
                        source="speechbrain/spkrec-ecapa-voxceleb",
                        savedir="pretrained_models/spkrec-ecapa-voxceleb",
                        run_opts={"device": _get_device()}
                    )

                    elapsed = round(time.time() - start, 1)
                    logger.info(f"SpeechBrain encoder loaded in {elapsed}s")

                except ImportError:
                    logger.error(
                        "SpeechBrain not installed. "
                        "Install with: pip install speechbrain"
                    )
                    raise
                except Exception as e:
                    logger.error(f"SpeechBrain encoder load failed: {e}")
                    raise RuntimeError(f"SpeechBrain load failed: {e}") from e
    return _speaker_encoder


def _get_device() -> str:
    """Determine the best available device (CUDA or CPU)."""
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("SpeechBrain will use GPU (CUDA)")
            return "cuda"
    except ImportError:
        pass
    logger.info("SpeechBrain will use CPU")
    return "cpu"


def is_loaded() -> bool:
    """Check if the speaker encoder is currently loaded in memory."""
    return _speaker_encoder is not None


def extract_embedding(audio_path: str) -> Optional[np.ndarray]:
    """
    Extract a speaker embedding from an audio file.

    Args:
        audio_path: Path to the audio file (WAV, 16kHz mono preferred).

    Returns:
        numpy array of shape (192,) containing the speaker embedding,
        or None if extraction fails.
    """
    try:
        encoder = get_speaker_encoder()

        logger.debug(f"Extracting embedding from: {audio_path}")
        start = time.time()

        # SpeechBrain expects a signal tensor; it can load from a file path
        embedding = encoder.encode_batch(
            encoder.load_audio(audio_path).unsqueeze(0)
        )

        # Convert from torch tensor to numpy array
        embedding_np = embedding.squeeze().cpu().detach().numpy()

        elapsed = round(time.time() - start, 3)
        logger.debug(
            f"Embedding extracted in {elapsed}s "
            f"(shape: {embedding_np.shape}, norm: {np.linalg.norm(embedding_np):.4f})"
        )

        return embedding_np

    except Exception as e:
        logger.error(f"Embedding extraction failed: {e}")
        return None


def cosine_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    """
    Compute cosine similarity between two embeddings.

    Args:
        embedding_a: First embedding vector.
        embedding_b: Second embedding vector.

    Returns:
        Cosine similarity score in range [-1, 1].
        1.0 = identical, 0.0 = orthogonal, -1.0 = opposite.
    """
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(embedding_a, embedding_b) / (norm_a * norm_b))


def identify_speaker(
    audio_path: str,
    enrolled_profiles: list[dict],
    threshold: float = SPEAKER_CONFIDENCE_THRESHOLD
) -> tuple[str, float]:
    """
    Identify a speaker by matching their voice against enrolled profiles.

    Args:
        audio_path: Path to the audio segment to identify.
        enrolled_profiles: List of profile dicts, each containing:
            - "userName": str (display name)
            - "voiceEmbedding": list[float] or numpy array (192-dim)
        threshold: Minimum cosine similarity to accept a match.

    Returns:
        Tuple of (speaker_name, confidence_score).
        If no match exceeds the threshold, returns ("Unknown Speaker", 0.0).
    """
    if not enrolled_profiles:
        return "Unknown Speaker", 0.0

    # Extract embedding for the unknown speaker
    unknown_embedding = extract_embedding(audio_path)
    if unknown_embedding is None:
        return "Unknown Speaker", 0.0

    return identify_speaker_from_embedding(unknown_embedding, enrolled_profiles, threshold)


def identify_speaker_from_embedding(
    unknown_embedding: np.ndarray,
    enrolled_profiles: list[dict],
    threshold: float = SPEAKER_CONFIDENCE_THRESHOLD
) -> tuple[str, float]:
    """
    Identify a speaker from a pre-extracted embedding.

    This is faster than identify_speaker() when you already have the
    embedding and want to match against multiple profiles.

    Args:
        unknown_embedding: The 192-dim embedding to identify.
        enrolled_profiles: List of profile dicts with 'userName' and 'voiceEmbedding'.
        threshold: Minimum cosine similarity to accept a match.

    Returns:
        Tuple of (speaker_name, confidence_score).
    """
    if not enrolled_profiles:
        return "Unknown Speaker", 0.0

    best_name = "Unknown Speaker"
    best_score = 0.0

    for profile in enrolled_profiles:
        profile_embedding = profile.get("voiceEmbedding")
        if profile_embedding is None:
            continue

        # Convert list to numpy if needed
        if isinstance(profile_embedding, list):
            profile_embedding = np.array(profile_embedding, dtype=np.float32)

        score = cosine_similarity(unknown_embedding, profile_embedding)

        logger.debug(
            f"  Match vs '{profile.get('userName', '?')}': "
            f"similarity={score:.4f} (threshold={threshold})"
        )

        if score > best_score:
            best_score = score
            if score >= threshold:
                best_name = profile.get("userName", "Unknown Speaker")

    logger.info(
        f"Speaker identified: '{best_name}' "
        f"(confidence={best_score:.4f}, threshold={threshold})"
    )

    return best_name, round(best_score, 4)


def extract_segment_audio(
    audio_path: str,
    start_sec: float,
    end_sec: float,
    output_path: str
) -> Optional[str]:
    """
    Extract a time segment from an audio file.

    Args:
        audio_path: Path to the source audio file.
        start_sec: Start time in seconds.
        end_sec: End time in seconds.
        output_path: Path for the extracted segment.

    Returns:
        Path to the extracted segment, or None if extraction fails.
    """
    import subprocess

    try:
        duration = end_sec - start_sec
        if duration <= 0:
            return None

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", audio_path,
                "-ss", str(start_sec),
                "-t", str(duration),
                "-ar", "16000",
                "-ac", "1",
                "-f", "wav",
                output_path
            ],
            capture_output=True,
            timeout=15
        )

        if result.returncode == 0:
            return output_path
        else:
            logger.warning(f"Segment extraction failed: {result.stderr.decode()[:100]}")
            return None

    except Exception as e:
        logger.error(f"Segment extraction error: {e}")
        return None
