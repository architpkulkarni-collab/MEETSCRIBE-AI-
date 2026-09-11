"""
AirScribe — Note Text Embeddings Service
============================================
Generates semantic embeddings for note content using sentence-transformers.

Purpose:
    Enables semantic search over encrypted notes. Since encrypted ciphertext
    cannot be searched with regex, we compute a semantic embedding vector
    BEFORE encryption and store it alongside the ciphertext. Search queries
    are then matched against these embeddings using cosine similarity.

Model:
    sentence-transformers/all-MiniLM-L6-v2
    - 384 dimensions
    - ~90MB download (auto-cached after first use)
    - Runs locally on CPU (no API keys needed)
    - Apache 2.0 license

Security notes:
    - Embeddings encode semantic MEANING, not exact text
    - They are a lossy representation — the original text cannot be
      perfectly reconstructed from an embedding
    - However, embeddings DO leak topical information: an attacker
      with access to embeddings can determine what topics notes
      are about (see implementation_plan.md Section I for full analysis)
    - Embeddings are stored UNENCRYPTED to enable search without
      decrypting the entire corpus

How it connects:
    Called by secure_notes.py during note creation/update to generate
    the embedding vector, and during search to compute query embeddings
    and rank notes by similarity.
"""

import logging
import threading
import time
from typing import Optional

import numpy as np

logger = logging.getLogger("airscribe.note_embeddings")

# ── Lazy-loaded model ──
_model = None
_model_lock = threading.Lock()

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


def get_embedding_model():
    """
    Lazy-load the sentence-transformers embedding model.

    Returns:
        A SentenceTransformer model instance.

    Raises:
        ImportError: If sentence-transformers is not installed.
        RuntimeError: If the model fails to load.
    """
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from sentence_transformers import SentenceTransformer
                    logger.info(f"Loading embedding model: {MODEL_NAME}...")
                    start = time.time()

                    _model = SentenceTransformer(
                        "all-MiniLM-L6-v2",
                        device="cpu"  # CPU is fast enough for single-note embedding
                    )

                    elapsed = round(time.time() - start, 1)
                    logger.info(f"Embedding model loaded in {elapsed}s")

                except ImportError:
                    logger.error(
                        "sentence-transformers not installed. "
                        "Install with: pip install sentence-transformers"
                    )
                    raise
                except Exception as e:
                    logger.error(f"Embedding model load failed: {e}")
                    raise RuntimeError(f"Embedding model load failed: {e}") from e
    return _model


def generate_embedding(text: str) -> Optional[list]:
    """
    Generate a 384-dimensional semantic embedding for a text string.

    Args:
        text: The plaintext note content to embed. Can be any length;
              the model handles truncation internally (max ~256 word pieces).

    Returns:
        A list of 384 floats representing the semantic embedding,
        or None if embedding generation fails.
    """
    if not text or not text.strip():
        # Return a zero vector for empty text (preserves schema consistency)
        return [0.0] * EMBEDDING_DIM

    try:
        model = get_embedding_model()
        # Encode returns a numpy array of shape (384,)
        embedding = model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,  # L2-normalize for cosine similarity
            show_progress_bar=False
        )

        return embedding.tolist()

    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return None


def generate_embeddings_batch(texts: list[str]) -> Optional[list[list]]:
    """
    Generate embeddings for a batch of texts (more efficient than one-by-one).

    Args:
        texts: List of plaintext strings.

    Returns:
        List of embedding lists (each 384 floats), or None on failure.
    """
    if not texts:
        return []

    try:
        model = get_embedding_model()
        # Replace empty strings with a placeholder to avoid errors
        processed = [t if t and t.strip() else " " for t in texts]

        embeddings = model.encode(
            processed,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=len(processed) > 50,
            batch_size=32
        )

        result = []
        for i, text in enumerate(texts):
            if not text or not text.strip():
                result.append([0.0] * EMBEDDING_DIM)
            else:
                result.append(embeddings[i].tolist())

        return result

    except Exception as e:
        logger.error(f"Batch embedding generation failed: {e}")
        return None


def cosine_similarity(vec_a: list, vec_b: list) -> float:
    """
    Compute cosine similarity between two embedding vectors.

    Args:
        vec_a: First embedding (list of floats).
        vec_b: Second embedding (list of floats).

    Returns:
        Cosine similarity score in range [-1, 1].
        1.0 = identical meaning, 0.0 = unrelated, -1.0 = opposite.
    """
    a = np.array(vec_a, dtype=np.float32)
    b = np.array(vec_b, dtype=np.float32)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))


def semantic_search(
    query_embedding: list,
    note_embeddings: list[dict],
    top_k: int = 10,
    threshold: float = 0.25
) -> list[dict]:
    """
    Find the most semantically similar notes to a query.

    Args:
        query_embedding: The embedding vector for the search query (384-dim).
        note_embeddings: List of dicts, each with at least:
            - "_id": note ID
            - "embedding": list of floats (384-dim)
        top_k: Maximum number of results to return.
        threshold: Minimum cosine similarity score to include in results.

    Returns:
        List of dicts sorted by descending similarity:
        [
            {"_id": "...", "score": 0.87, ...},
            {"_id": "...", "score": 0.72, ...},
        ]
    """
    if not query_embedding or not note_embeddings:
        return []

    query_vec = np.array(query_embedding, dtype=np.float32)
    query_norm = np.linalg.norm(query_vec)
    if query_norm == 0:
        return []

    results = []
    for note in note_embeddings:
        emb = note.get("embedding")
        if not emb:
            continue

        note_vec = np.array(emb, dtype=np.float32)
        note_norm = np.linalg.norm(note_vec)
        if note_norm == 0:
            continue

        score = float(np.dot(query_vec, note_vec) / (query_norm * note_norm))

        if score >= threshold:
            results.append({
                **{k: v for k, v in note.items() if k != "embedding"},
                "score": round(score, 4)
            })

    # Sort by score descending
    results.sort(key=lambda x: x["score"], reverse=True)

    return results[:top_k]
