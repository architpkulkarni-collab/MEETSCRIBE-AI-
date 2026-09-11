"""
AirScribe AI Server — Voice Enrollment
=========================================
Flask blueprint providing API routes for voice profile management.

Purpose:
    Allows users to enroll their voice by recording ~30 seconds of speech.
    The audio is processed to extract a speaker embedding, which is stored
    in MongoDB for later speaker recognition during meetings.

How it connects:
    Registered as a Flask blueprint in ai_server/app.py.
    The frontend (voice_profile.js) calls these routes.
    Embeddings are stored via ai_server/embeddings.py.

How to configure:
    MIN_ENROLLMENT_DURATION and MAX_AUDIO_SIZE_MB in .env control
    validation thresholds.

How to debug:
    All routes log via 'airscribe.ai.enrollment'.
    POST /ai/voice/enroll returns detailed error messages on validation failure.
"""

import os
import time
import logging

from flask import Blueprint, request, jsonify

from ai_server.config import (
    UPLOAD_FOLDER,
    MIN_ENROLLMENT_DURATION,
    MAX_AUDIO_SIZE_MB
)
from ai_server.utils import (
    convert_to_wav,
    get_audio_duration,
    validate_audio_file,
    cleanup_temp_files
)
from ai_server.speaker_recognition import extract_embedding
from ai_server import embeddings

logger = logging.getLogger("airscribe.ai.enrollment")

# ── Flask Blueprint ──
voice_enrollment_bp = Blueprint("voice_enrollment", __name__)


@voice_enrollment_bp.route("/ai/voice/enroll", methods=["POST"])
def enroll_voice():
    """
    Enroll a user's voice profile.

    Accepts an audio recording (30s recommended), extracts a speaker
    embedding using SpeechBrain, and stores it in MongoDB.

    Request: multipart/form-data
        - audio: Audio file (wav/webm/ogg, max 10MB)
        - userId: User's unique ID from the auth system
        - userName: Display name for transcripts

    Returns:
        201: { success: true, profileId, userName, duration }
        400: { error: "validation message" }
        500: { error: "processing error" }
    """
    # ── Validate required fields ──
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    user_id = request.form.get("userId", "").strip()
    user_name = request.form.get("userName", "").strip()

    if not user_id:
        return jsonify({"error": "userId is required"}), 400
    if not user_name:
        return jsonify({"error": "userName is required"}), 400

    audio_file = request.files["audio"]

    # ── Validate audio file ──
    is_valid, error_msg = validate_audio_file(audio_file, MAX_AUDIO_SIZE_MB)
    if not is_valid:
        return jsonify({"error": error_msg}), 400

    # ── Save to temp file ──
    ext = audio_file.filename.rsplit(".", 1)[-1] if "." in (audio_file.filename or "") else "webm"
    temp_filename = f"enroll_{user_id}_{int(time.time())}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    wav_path = None

    try:
        audio_file.save(temp_path)
        logger.info(f"Enrollment audio saved: {temp_path}")

        # ── Convert to WAV ──
        wav_path = convert_to_wav(temp_path)

        # ── Check duration ──
        duration = get_audio_duration(wav_path)
        if duration < MIN_ENROLLMENT_DURATION:
            return jsonify({
                "error": f"Audio too short ({duration:.1f}s). "
                         f"Minimum is {MIN_ENROLLMENT_DURATION}s. "
                         f"Please record for at least 30 seconds."
            }), 400

        logger.info(f"Enrollment audio duration: {duration:.1f}s")

        # ── Extract embedding ──
        embedding_vector = extract_embedding(wav_path)
        if embedding_vector is None:
            return jsonify({
                "error": "Failed to extract voice embedding. "
                         "Please try recording again in a quiet environment."
            }), 500

        # ── Store in MongoDB ──
        profile_id = embeddings.store_embedding(
            user_id=user_id,
            user_name=user_name,
            embedding_vector=embedding_vector,
            sample_duration=duration
        )

        if not profile_id:
            return jsonify({"error": "Failed to save voice profile"}), 500

        logger.info(
            f"Voice enrollment complete: '{user_name}' "
            f"(id: {profile_id}, duration: {duration:.1f}s)"
        )

        return jsonify({
            "success": True,
            "profileId": profile_id,
            "userName": user_name,
            "duration": round(duration, 1),
            "message": f"Voice profile for '{user_name}' enrolled successfully"
        }), 201

    except Exception as e:
        logger.error(f"Enrollment error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Enrollment failed: {str(e)}"}), 500

    finally:
        # Clean up temp files — never store raw recordings
        paths_to_clean = [temp_path]
        if wav_path and wav_path != temp_path:
            paths_to_clean.append(wav_path)
        cleanup_temp_files(*paths_to_clean)


@voice_enrollment_bp.route("/ai/voice/profiles", methods=["GET"])
def list_profiles():
    """
    List all enrolled voice profiles.

    Never exposes raw embedding vectors — returns metadata only.

    Query params:
        - userId: (optional) Filter by user ID

    Returns:
        200: [ { _id, userId, userName, embeddingModel, sampleDuration, createdAt, updatedAt } ]
    """
    user_id = request.args.get("userId")
    profiles = embeddings.get_all_profiles_metadata(user_id)
    return jsonify(profiles)


@voice_enrollment_bp.route("/ai/voice/profile/<profile_id>", methods=["GET"])
def get_profile(profile_id):
    """
    Get a single voice profile's metadata (without embedding).

    Returns:
        200: { _id, userId, userName, ... }
        404: { error: "Profile not found" }
    """
    profile = embeddings.get_embedding(profile_id)
    if not profile:
        return jsonify({"error": "Profile not found"}), 404

    # Remove the embedding before returning
    profile.pop("voiceEmbedding", None)
    return jsonify(profile)


@voice_enrollment_bp.route("/ai/voice/profile/<profile_id>", methods=["DELETE"])
def delete_profile(profile_id):
    """
    Delete a voice profile.

    Returns:
        200: { success: true, message: "..." }
        404: { error: "Profile not found" }
    """
    success = embeddings.delete_embedding(profile_id)
    if success:
        return jsonify({
            "success": True,
            "message": "Voice profile deleted"
        })
    return jsonify({"error": "Profile not found"}), 404


@voice_enrollment_bp.route("/ai/voice/test-mic", methods=["POST"])
def test_microphone():
    """
    Quick microphone test endpoint.

    Receives a short audio clip and returns quality metrics:
    - Duration
    - Estimated volume level
    - Whether audio is usable

    Request: multipart/form-data
        - audio: Short audio clip (1-5 seconds)

    Returns:
        200: { success: true, duration, volumeLevel, isUsable }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    temp_path = os.path.join(UPLOAD_FOLDER, f"mictest_{int(time.time())}.webm")
    wav_path = None

    try:
        audio_file.save(temp_path)
        wav_path = convert_to_wav(temp_path)
        duration = get_audio_duration(wav_path)

        # Estimate volume level from WAV data
        volume_level = _estimate_volume(wav_path)
        is_usable = duration > 0.5 and volume_level > 0.01

        return jsonify({
            "success": True,
            "duration": round(duration, 2),
            "volumeLevel": round(volume_level, 4),
            "isUsable": is_usable,
            "message": "Microphone working" if is_usable else "Audio too quiet or too short"
        })

    except Exception as e:
        logger.error(f"Mic test error: {e}")
        return jsonify({"error": str(e)}), 500

    finally:
        paths = [temp_path]
        if wav_path and wav_path != temp_path:
            paths.append(wav_path)
        cleanup_temp_files(*paths)


def _estimate_volume(wav_path: str) -> float:
    """
    Estimate the RMS volume level of a WAV file.

    Returns a float between 0.0 (silence) and 1.0 (maximum volume).
    """
    try:
        import wave
        import struct

        with wave.open(wav_path, "rb") as wf:
            n_frames = min(wf.getnframes(), 16000 * 5)  # Max 5 seconds
            if n_frames == 0:
                return 0.0

            frames = wf.readframes(n_frames)
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()

            if sample_width == 2:
                fmt = f"<{n_frames * n_channels}h"
                samples = struct.unpack(fmt, frames)
            else:
                return 0.0

            # Compute RMS
            sum_sq = sum(s * s for s in samples)
            rms = (sum_sq / len(samples)) ** 0.5
            # Normalize to 0-1 range (16-bit audio max is 32767)
            return min(rms / 32767.0, 1.0)

    except Exception:
        return 0.0
