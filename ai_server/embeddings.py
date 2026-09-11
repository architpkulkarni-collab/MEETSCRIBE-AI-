"""
AirScribe AI Server — Embeddings Storage
==========================================
MongoDB integration for voice profile embeddings.

Purpose:
    CRUD operations for the voice_profiles collection in MongoDB.
    Handles storage and retrieval of speaker embeddings, ensuring
    they are never exposed to the frontend in raw form.

How it connects:
    Used by voice_enrollment.py (to store/delete profiles) and
    pipeline.py (to retrieve profiles for speaker matching).

How to configure:
    Uses MONGO_URI and DB_NAME from config.py.

How to debug:
    All operations log via the 'airscribe.ai.embeddings' logger.
    Check MongoDB Atlas dashboard to verify documents in voice_profiles.
"""

import logging
from datetime import datetime
from typing import Optional
import numpy as np

from pymongo import MongoClient
from bson.objectid import ObjectId

from ai_server.config import MONGO_URI, DB_NAME

logger = logging.getLogger("airscribe.ai.embeddings")

# ── MongoDB Connection ──
_client = None
_db = None
_voice_profiles_col = None


def _get_collection():
    """
    Get the voice_profiles MongoDB collection.
    Lazy-initializes the connection on first use.
    """
    global _client, _db, _voice_profiles_col
    if _voice_profiles_col is None:
        _client = MongoClient(MONGO_URI)
        _db = _client[DB_NAME]
        _voice_profiles_col = _db["voice_profiles"]
        logger.info(f"Connected to MongoDB: {DB_NAME}.voice_profiles")
    return _voice_profiles_col


def store_embedding(
    user_id: str,
    user_name: str,
    embedding_vector: np.ndarray,
    sample_duration: float = 0.0,
    sample_rate: int = 16000
) -> Optional[str]:
    """
    Store a voice embedding in MongoDB.

    If a profile already exists for this user_id, it is updated (replaced).
    Each user gets exactly one voice profile.

    Args:
        user_id: The user's unique ID (from the auth system).
        user_name: Display name for transcripts (e.g., "Archit").
        embedding_vector: The 192-dim numpy array from SpeechBrain.
        sample_duration: Duration of the enrollment audio in seconds.
        sample_rate: Sample rate of the enrollment audio.

    Returns:
        The profile document's _id as a string, or None on failure.
    """
    col = _get_collection()
    now = datetime.utcnow().isoformat() + "Z"

    # Convert numpy array to a plain Python list for BSON storage
    embedding_list = embedding_vector.tolist()

    profile = {
        "userId": user_id,
        "userName": user_name,
        "voiceEmbedding": embedding_list,
        "embeddingModel": "speechbrain-ecapa-voxceleb",
        "embeddingDimension": len(embedding_list),
        "sampleDuration": round(sample_duration, 2),
        "sampleRate": sample_rate,
        "updatedAt": now
    }

    try:
        # Upsert: update if exists, insert if not
        result = col.update_one(
            {"userId": user_id},
            {
                "$set": profile,
                "$setOnInsert": {"createdAt": now}
            },
            upsert=True
        )

        # Get the document ID
        if result.upserted_id:
            doc_id = str(result.upserted_id)
            logger.info(f"Stored new voice profile for '{user_name}' (id: {doc_id})")
        else:
            existing = col.find_one({"userId": user_id})
            doc_id = str(existing["_id"]) if existing else None
            logger.info(f"Updated voice profile for '{user_name}'")

        return doc_id

    except Exception as e:
        logger.error(f"Failed to store embedding: {e}")
        return None


def get_embedding(profile_id: str) -> Optional[dict]:
    """
    Retrieve a single voice profile by its _id.

    Args:
        profile_id: The MongoDB _id as a string.

    Returns:
        Profile dict with 'userId', 'userName', 'voiceEmbedding', etc.
        Returns None if not found.
    """
    col = _get_collection()
    try:
        doc = col.find_one({"_id": ObjectId(profile_id)})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve profile {profile_id}: {e}")
        return None


def get_profile_by_user(user_id: str) -> Optional[dict]:
    """
    Retrieve a voice profile by user ID.

    Args:
        user_id: The user's unique ID.

    Returns:
        Profile dict, or None if no profile exists for this user.
    """
    col = _get_collection()
    try:
        doc = col.find_one({"userId": user_id})
        if doc:
            doc["_id"] = str(doc["_id"])
            return doc
        return None
    except Exception as e:
        logger.error(f"Failed to retrieve profile for user {user_id}: {e}")
        return None


def get_all_embeddings(user_id: Optional[str] = None) -> list[dict]:
    """
    Retrieve all voice profiles, optionally filtered by user.

    This is used during meeting processing to get all enrolled
    profiles for speaker matching.

    Args:
        user_id: If provided, only return profiles for this user.
            If None, returns ALL profiles (for matching against any enrolled speaker).

    Returns:
        List of profile dicts with 'userId', 'userName', 'voiceEmbedding'.
    """
    col = _get_collection()
    try:
        query = {"userId": user_id} if user_id else {}
        docs = list(col.find(query))
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        logger.debug(f"Retrieved {len(docs)} voice profiles")
        return docs
    except Exception as e:
        logger.error(f"Failed to retrieve profiles: {e}")
        return []


def get_all_profiles_metadata(user_id: Optional[str] = None) -> list[dict]:
    """
    Retrieve all voice profiles WITHOUT the embedding vectors.

    This is the safe version for API responses — never exposes
    raw embedding data to the frontend.

    Args:
        user_id: If provided, only return profiles for this user.

    Returns:
        List of profile metadata dicts (no voiceEmbedding field).
    """
    col = _get_collection()
    try:
        query = {"userId": user_id} if user_id else {}
        # Projection: exclude the embedding field
        docs = list(col.find(query, {"voiceEmbedding": 0}))
        for doc in docs:
            doc["_id"] = str(doc["_id"])
        return docs
    except Exception as e:
        logger.error(f"Failed to retrieve profile metadata: {e}")
        return []


def delete_embedding(profile_id: str) -> bool:
    """
    Delete a voice profile by its _id.

    Args:
        profile_id: The MongoDB _id as a string.

    Returns:
        True if deleted, False if not found or error.
    """
    col = _get_collection()
    try:
        result = col.delete_one({"_id": ObjectId(profile_id)})
        if result.deleted_count > 0:
            logger.info(f"Deleted voice profile: {profile_id}")
            return True
        logger.warning(f"Voice profile not found for deletion: {profile_id}")
        return False
    except Exception as e:
        logger.error(f"Failed to delete profile {profile_id}: {e}")
        return False


def delete_user_profile(user_id: str) -> bool:
    """
    Delete a voice profile by user ID.

    Args:
        user_id: The user's unique ID.

    Returns:
        True if deleted, False if not found or error.
    """
    col = _get_collection()
    try:
        result = col.delete_one({"userId": user_id})
        if result.deleted_count > 0:
            logger.info(f"Deleted voice profile for user: {user_id}")
            return True
        return False
    except Exception as e:
        logger.error(f"Failed to delete profile for user {user_id}: {e}")
        return False


def count_profiles(user_id: Optional[str] = None) -> int:
    """Count the number of enrolled voice profiles."""
    col = _get_collection()
    try:
        query = {"userId": user_id} if user_id else {}
        return col.count_documents(query)
    except Exception:
        return 0
