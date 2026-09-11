"""
AirScribe AI Server — Modular Entry Point
============================================
New Flask app that registers all AI service blueprints.

Purpose:
    This is the NEW entry point for the AI server. It imports all the
    modular components (whisper, diarization, recognition, enrollment)
    and registers them as Flask blueprints on a single server.

    The original ai_meeting_server.py continues to work as before —
    this new app.py is an alternative entry point that adds voice
    enrollment and enhanced speaker recognition routes.

How it connects:
    Run this instead of (or alongside) ai_meeting_server.py:
        python -m ai_server.app

    It registers:
        - Voice enrollment routes (/ai/voice/*)
        - Health check (/ai/health)
        - All existing meeting routes (imported from ai_meeting_server.py)

How to configure:
    Same as ai_meeting_server.py — uses config.py for all settings.

How to debug:
    Run with AI_DEBUG=True for verbose logging.
    Check /ai/health for model status.
"""

import os
import sys
import logging

from flask import Flask, jsonify
from flask_cors import CORS

# Add parent directory to path so we can import ai_meeting_server
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ai_server.config import AI_HOST, AI_PORT, AI_DEBUG, print_config
from ai_server.utils import setup_logging
from ai_server.voice_enrollment import voice_enrollment_bp

# ── App Setup ──
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/ai/*": {"origins": "*"}})

# ── Register Blueprints ──
app.register_blueprint(voice_enrollment_bp)


# ── Enhanced Health Check ──
@app.route("/ai/health", methods=["GET"])
def health():
    """
    Enhanced health check that reports status of all AI modules.

    Returns model loading status, diarization mode, recognition
    capability, and enrolled profile count.
    """
    from ai_server.whisper_service import is_loaded as whisper_loaded
    from ai_server.diarization import is_loaded as diarization_loaded, get_mode
    from ai_server.speaker_recognition import is_loaded as recognition_loaded
    from ai_server.embeddings import count_profiles
    from ai_server.config import WHISPER_MODEL_SIZE

    return jsonify({
        "status": "ok",
        "service": "AirScribe AI Meeting Server (Modular)",
        "modules": {
            "whisper": {
                "loaded": whisper_loaded(),
                "model": WHISPER_MODEL_SIZE
            },
            "diarization": {
                "loaded": diarization_loaded(),
                "mode": get_mode()
            },
            "recognition": {
                "loaded": recognition_loaded(),
                "model": "speechbrain-ecapa-voxceleb"
            },
            "enrollment": {
                "enrolledProfiles": count_profiles()
            }
        }
    })


# ── Summary Route ──
@app.route("/ai/summary", methods=["POST"])
def generate_meeting_summary():
    """
    Generate a meeting summary from transcript segments.

    Request body:
    {
        "segments": [
            {"speaker": "Archit", "text": "Hello"},
            {"speaker": "Rahul", "text": "Hi"}
        ]
    }

    Returns:
    {
        "overview": "...",
        "keyPoints": [...],
        "actionItems": [...],
        "decisions": [...],
        "speakerStats": {...}
    }
    """
    from flask import request
    from ai_server.summary import generate_summary

    data = request.json or {}
    segments = data.get("segments", [])

    if not segments:
        return jsonify({"error": "No segments provided"}), 400

    summary = generate_summary(segments)
    return jsonify(summary)


# ── Import and mount the original meeting routes ──
def _mount_original_routes():
    """
    Import the original ai_meeting_server.py and mount its routes
    on this app, so we have a single unified server.

    This is done carefully to avoid duplicate route registration.
    """
    try:
        import ai_meeting_server as original

        # Copy routes from the original app that we don't already have
        existing_rules = {rule.rule for rule in app.url_map.iter_rules()}

        for rule in original.app.url_map.iter_rules():
            if rule.rule not in existing_rules and rule.endpoint != "static":
                # Get the view function
                view_func = original.app.view_functions.get(rule.endpoint)
                if view_func:
                    app.add_url_rule(
                        rule.rule,
                        endpoint=f"original_{rule.endpoint}",
                        view_func=view_func,
                        methods=rule.methods - {"OPTIONS", "HEAD"}
                    )

        # Share the sessions store
        app.config["_original_sessions"] = original.sessions
        app.config["_original_sessions_lock"] = original.sessions_lock

        logger = logging.getLogger("airscribe.ai.app")
        logger.info("Original ai_meeting_server routes mounted successfully")
        return True

    except ImportError:
        logger = logging.getLogger("airscribe.ai.app")
        logger.warning(
            "ai_meeting_server.py not found — only voice enrollment routes available. "
            "For full meeting transcription, ensure ai_meeting_server.py is in the parent directory."
        )
        return False
    except Exception as e:
        logger = logging.getLogger("airscribe.ai.app")
        logger.error(f"Failed to mount original routes: {e}")
        return False


# ── Run ──
if __name__ == "__main__":
    setup_logging(AI_DEBUG)
    print_config()

    # Try to mount original meeting routes
    _mount_original_routes()

    print(f"\n{'=' * 55}")
    print(f"  AirScribe AI Meeting Server (Modular)")
    print(f"  Running at http://localhost:{AI_PORT}")
    print(f"  Routes: /ai/health, /ai/voice/*, /ai/session/*, /ai/transcribe")
    print(f"{'=' * 55}\n")

    app.run(host=AI_HOST, port=AI_PORT, debug=AI_DEBUG)
