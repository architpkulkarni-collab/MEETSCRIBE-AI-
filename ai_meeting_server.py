"""
AirScribe AI Meeting Server
============================
Flask backend for AI-powered multi-speaker meeting transcription.

Features:
  - Receives audio chunks from the frontend (MediaRecorder API)
  - Transcribes speech using OpenAI Whisper (local model)
  - Performs speaker diarization to detect multiple speakers
  - Maintains speaker consistency across audio chunks
  - Preserves manual speaker name assignments (never overrides them)
  - Persistent voice registration and speaker recognition for Person 1, Person 2, etc.

Runs on port 5001 to avoid conflict with the main AirScribe backend (port 5000).
"""

import os
import io
import json
import wave
import time
import uuid
import tempfile
import threading
from datetime import datetime
from collections import defaultdict

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

load_dotenv()

# ── App Setup ──
app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/ai/*": {"origins": "*"}})

# ── Register voice enrollment routes from ai_server package ──
try:
    from ai_server.voice_enrollment import voice_enrollment_bp
    app.register_blueprint(voice_enrollment_bp)
    print("✓ Voice enrollment routes registered (/ai/voice/*)")
except ImportError:
    print("ℹ  ai_server package not found — voice enrollment disabled")
    print("   (This is fine if you haven't installed the speaker recognition dependencies)")


# ── Configuration ──
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL", "base")  # tiny, base, small, medium, large
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "temp_audio")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Lazy-loaded AI models ──
_whisper_model = None
_whisper_lock = threading.Lock()
_diarization_pipeline = None
_diarization_lock = threading.Lock()


def get_whisper_model():
    """Lazy-load Whisper model on first use."""
    global _whisper_model
    if _whisper_model is None:
        with _whisper_lock:
            if _whisper_model is None:
                try:
                    import whisper
                    print(f"⏳ Loading Whisper model '{WHISPER_MODEL_SIZE}'...")
                    _whisper_model = whisper.load_model(WHISPER_MODEL_SIZE)
                    print(f"✅ Whisper model '{WHISPER_MODEL_SIZE}' loaded successfully")
                except ImportError:
                    print("⚠️  Whisper not installed. Install with: pip install openai-whisper")
                    raise
                except Exception as e:
                    print(f"⚠️  Whisper load error: {e}")
                    raise
    return _whisper_model


def get_diarization_pipeline():
    """Lazy-load speaker diarization pipeline on first use."""
    global _diarization_pipeline
    if _diarization_pipeline is None:
        with _diarization_lock:
            if _diarization_pipeline is None:
                try:
                    from pyannote.audio import Pipeline
                    hf_token = os.getenv("HF_TOKEN", "")
                    if hf_token:
                        print("⏳ Loading speaker diarization pipeline...")
                        _diarization_pipeline = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=hf_token
                        )
                        print("✅ Diarization pipeline loaded")
                    else:
                        print("⚠️  HF_TOKEN not set — using simple diarization fallback")
                        _diarization_pipeline = "fallback"
                except ImportError:
                    print("⚠️  pyannote.audio not installed — using simple diarization fallback")
                    _diarization_pipeline = "fallback"
                except Exception as e:
                    print(f"⚠️  Diarization pipeline error: {e} — using fallback")
                    _diarization_pipeline = "fallback"
    return _diarization_pipeline


# ══════════════════════════════════════════════════════
#  SESSION MANAGEMENT
#  Tracks speaker consistency across chunks per session
# ══════════════════════════════════════════════════════

class MeetingSession:
    """Stores state for an active meeting session."""

    def __init__(self, session_id, speaker_count=2):
        self.session_id = session_id
        self.speaker_count = speaker_count
        self.created_at = datetime.utcnow()
        self.chunks_processed = 0
        self.total_duration = 0.0

        # Speaker tracking
        # Maps diarization speaker labels (e.g. "SPEAKER_00") → our consistent index (0, 1, 2...)
        self.diarization_to_index = {}
        self.next_speaker_index = 0

        # Manual name overrides: index → custom name (set by user)
        # These are NEVER overwritten by AI detection
        self.manual_names = {}

        # Default labels: index → "Speaker 1", "Speaker 2", etc.
        self.auto_labels = {}

        # Full transcript history
        self.transcript = []

        # Voice recognition: enrolled profiles for this session
        self.enrolled_profiles = []

    def get_or_assign_speaker_index(self, diarization_label):
        """
        Map a diarization label to a consistent speaker index.
        If this label was seen before, return the same index.
        If new, assign the next available index.
        """
        if diarization_label in self.diarization_to_index:
            return self.diarization_to_index[diarization_label]

        idx = self.next_speaker_index
        self.diarization_to_index[diarization_label] = idx
        self.auto_labels[idx] = f"Speaker {idx + 1}"
        self.next_speaker_index += 1
        return idx

    def get_speaker_name(self, index):
        """
        Get the display name for a speaker index.
        Manual names always take priority over auto-generated labels.
        """
        if index in self.manual_names:
            return self.manual_names[index]
        return self.auto_labels.get(index, f"Speaker {index + 1}")

    def set_manual_name(self, index, name):
        """
        Set a manual name for a speaker. This will never be overwritten by AI.
        """
        if name and name.strip():
            self.manual_names[index] = name.strip()

    def to_dict(self):
        """Serialize session state for API responses."""
        speakers = {}
        for idx in range(self.next_speaker_index):
            speakers[str(idx)] = {
                "index": idx,
                "label": self.get_speaker_name(idx),
                "isManual": idx in self.manual_names
            }
        return {
            "sessionId": self.session_id,
            "speakerCount": self.next_speaker_index,
            "speakers": speakers,
            "chunksProcessed": self.chunks_processed,
            "totalDuration": round(self.total_duration, 1),
            "createdAt": self.created_at.isoformat() + "Z"
        }


# Active sessions store
sessions = {}
sessions_lock = threading.Lock()


def get_session(session_id):
    """Get or create a meeting session."""
    with sessions_lock:
        if session_id not in sessions:
            sessions[session_id] = MeetingSession(session_id)
        return sessions[session_id]


# ══════════════════════════════════════════════════════
#  AUDIO PROCESSING
# ══════════════════════════════════════════════════════

def simple_diarization(segments, num_speakers=2):
    """
    Simple energy-based speaker diarization fallback.
    When pyannote is not available, uses silence gaps to alternate speakers.
    This is a best-effort approach that works reasonably for 2-speaker meetings.
    """
    if not segments:
        return []

    result = []
    current_speaker = 0
    last_end_time = 0.0

    for seg in segments:
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


def transcribe_and_diarize(audio_path, session):
    """
    Process an audio file: transcribe with Whisper, then diarize speakers.

    Returns a list of segments:
    [
        {"speaker": "Speaker 1", "speakerIndex": 0, "text": "Hello everyone", "start": 0.0, "end": 2.1},
        {"speaker": "Speaker 2", "speakerIndex": 1, "text": "Hi, let's begin", "start": 2.5, "end": 4.0},
    ]
    """
    results = []

    try:
        # ── Enhanced Pipeline Bridge ──
        # If the new ai_server.pipeline module is installed, use it for
        # speaker recognition. Otherwise, fall through to the existing logic.
        if session.enrolled_profiles:
            try:
                from ai_server.pipeline import process_audio_chunk
                results = process_audio_chunk(
                    audio_path, session, session.enrolled_profiles
                )
                if results:
                    return results
            except ImportError:
                pass  # ai_server not installed — use existing logic
            except Exception as e:
                print(f"⚠️  Enhanced pipeline error: {e} — falling back")

        # Step 1: Transcribe with Whisper
        model = get_whisper_model()
        whisper_result = model.transcribe(
            audio_path,
            language="en",
            word_timestamps=True,
            verbose=False
        )

        raw_segments = whisper_result.get("segments", [])
        if not raw_segments:
            return results

        # ── Step 1.5: Match against registered voice profiles from saved data ──
        saved_reg = None
        try:
            from voice_intelligence.overlap_suite.voice_fingerprint import get_saved_registry
            saved_reg = get_saved_registry()
        except Exception:
            pass

        if saved_reg is not None and len(saved_reg) > 0:
            logger_msg = f"Matching against {len(saved_reg)} saved voice profiles ({saved_reg.all_speakers()})"
            print(f"ℹ️  {logger_msg}")
            for seg in raw_segments:
                seg_start = seg["start"]
                seg_end = seg["end"]
                seg_text = seg["text"].strip()
                if not seg_text:
                    continue

                best_speaker, confidence, _ = saved_reg.recognize(
                    audio_input=audio_path,
                    start_sec=seg_start,
                    end_sec=seg_end,
                    threshold=0.38
                )

                if best_speaker != "UNKNOWN":
                    speaker_idx = session.get_or_assign_speaker_index(best_speaker)
                    if speaker_idx not in session.manual_names:
                        session.auto_labels[speaker_idx] = best_speaker
                    speaker_name = session.get_speaker_name(speaker_idx)
                else:
                    speaker_idx = session.get_or_assign_speaker_index("SPEAKER_00")
                    speaker_name = session.get_speaker_name(speaker_idx)

                results.append({
                    "speaker": speaker_name,
                    "speakerIndex": speaker_idx,
                    "text": seg_text,
                    "start": round(seg_start, 2),
                    "end": round(seg_end, 2),
                    "confidence": round(confidence, 3)
                })

            if results:
                return results

        # Step 2: Speaker diarization (if no saved profiles matched)
        pipeline = get_diarization_pipeline()

        if pipeline != "fallback" and pipeline is not None:
            # Use pyannote for real diarization
            try:
                diarization = pipeline(audio_path, num_speakers=session.speaker_count)

                # Map each Whisper segment to the most overlapping speaker
                for seg in raw_segments:
                    seg_start = seg["start"]
                    seg_end = seg["end"]
                    seg_text = seg["text"].strip()
                    if not seg_text:
                        continue

                    # Find which diarization speaker overlaps most with this segment
                    best_speaker = None
                    best_overlap = 0.0

                    for turn, _, speaker_label in diarization.itertracks(yield_label=True):
                        overlap_start = max(seg_start, turn.start)
                        overlap_end = min(seg_end, turn.end)
                        overlap = max(0, overlap_end - overlap_start)
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_speaker = speaker_label

                    if best_speaker is None:
                        best_speaker = "SPEAKER_00"

                    speaker_idx = session.get_or_assign_speaker_index(best_speaker)
                    speaker_name = session.get_speaker_name(speaker_idx)

                    results.append({
                        "speaker": speaker_name,
                        "speakerIndex": speaker_idx,
                        "text": seg_text,
                        "start": round(seg_start, 2),
                        "end": round(seg_end, 2)
                    })

            except Exception as e:
                print(f"⚠️  Diarization error: {e} — falling back to simple method")
                # Fall through to simple diarization
                diarized = simple_diarization(raw_segments, session.speaker_count)
                for seg in diarized:
                    speaker_idx = session.get_or_assign_speaker_index(seg["speaker"])
                    speaker_name = session.get_speaker_name(speaker_idx)
                    results.append({
                        "speaker": speaker_name,
                        "speakerIndex": speaker_idx,
                        "text": seg["text"],
                        "start": seg["start"],
                        "end": seg["end"]
                    })
        else:
            # Use simple fallback diarization
            diarized = simple_diarization(raw_segments, session.speaker_count)
            for seg in diarized:
                speaker_idx = session.get_or_assign_speaker_index(seg["speaker"])
                speaker_name = session.get_speaker_name(speaker_idx)
                results.append({
                    "speaker": speaker_name,
                    "speakerIndex": speaker_idx,
                    "text": seg["text"],
                    "start": seg["start"],
                    "end": seg["end"]
                })

    except Exception as e:
        print(f"❌ Transcription error: {e}")
        import traceback
        traceback.print_exc()

    return results


def merge_consecutive_speakers(segments):
    """
    Merge consecutive segments that have the same speaker.
    e.g., [Alice: "Hello", Alice: "world"] → [Alice: "Hello world"]
    """
    if not segments:
        return []

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speakerIndex"] == prev["speakerIndex"]:
            prev["text"] += " " + seg["text"]
            prev["end"] = seg["end"]
        else:
            merged.append(seg.copy())

    return merged


# ══════════════════════════════════════════════════════
#  API ROUTES
# ══════════════════════════════════════════════════════

@app.route("/ai/health", methods=["GET"])
def health():
    """Health check for the AI meeting server."""
    whisper_loaded = _whisper_model is not None
    diarization_loaded = _diarization_pipeline is not None

    return jsonify({
        "status": "ok",
        "service": "AirScribe AI Meeting Server",
        "whisperModel": WHISPER_MODEL_SIZE,
        "whisperLoaded": whisper_loaded,
        "diarizationLoaded": diarization_loaded,
        "diarizationMode": "pyannote" if (diarization_loaded and _diarization_pipeline != "fallback") else "fallback",
        "activeSessions": len(sessions)
    })


@app.route("/ai/session/start", methods=["POST"])
def start_session():
    """
    Start a new meeting session.

    Request body:
    {
        "speakerCount": 2,
        "speakerNames": ["Alice", "Bob"]  // optional — manual names
    }

    Returns: { sessionId, speakers, ... }
    """
    data = request.json or {}
    speaker_count = data.get("speakerCount", 2)
    speaker_names = data.get("speakerNames", [])

    session_id = str(uuid.uuid4())[:12]
    session = MeetingSession(session_id, speaker_count=speaker_count)

    # Apply initial manual names from the setup modal
    for i, name in enumerate(speaker_names):
        if name and name.strip() and not name.strip().startswith("Speaker "):
            session.set_manual_name(i, name.strip())
        # Pre-create auto labels for expected speakers
        session.auto_labels[i] = f"Speaker {i + 1}"
        if i >= session.next_speaker_index:
            session.next_speaker_index = i + 1

    # ── Voice Recognition: Load enrolled profiles if requested ──
    if data.get("useVoiceRecognition", False):
        try:
            from ai_server.embeddings import get_all_embeddings
            session.enrolled_profiles = get_all_embeddings()
            print(f"  Loaded {len(session.enrolled_profiles)} voice profiles for recognition")
        except ImportError:
            print("  ai_server.embeddings not available — voice recognition disabled")
        except Exception as e:
            print(f"  Could not load voice profiles: {e}")

    with sessions_lock:
        sessions[session_id] = session

    return jsonify(session.to_dict()), 201


@app.route("/ai/session/<session_id>/rename-speaker", methods=["POST"])
def rename_speaker(session_id):
    """
    Rename a speaker in an active session.
    Preserves manual names so AI diarization never overwrites them.

    Request body:
    {
        "speakerIndex": 0,
        "name": "Alice"
    }
    """
    data = request.json or {}
    speaker_index = int(data.get("speakerIndex", 0))
    name = data.get("name", "").strip()

    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        session = sessions[session_id]
        session.set_manual_name(speaker_index, name)

        return jsonify({
            "message": f"Speaker {speaker_index} renamed to '{name}'",
            "speakers": session.to_dict()["speakers"]
        })


@app.route("/ai/session/<session_id>/speakers", methods=["GET"])
def get_speakers(session_id):
    """Get the current speaker mapping for a session."""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        session = sessions[session_id]
        return jsonify({
            "speakers": [
                {
                    "index": idx,
                    "name": session.get_speaker_name(idx),
                    "isManual": idx in session.manual_names
                }
                for idx in range(session.next_speaker_index)
            ]
        })


@app.route("/ai/transcribe", methods=["POST"])
def transcribe_chunk():
    """
    Receive an audio chunk, transcribe it with Whisper, and diarize speakers.

    Request: multipart/form-data
        - audio: audio file (webm/wav/ogg)
        - sessionId: meeting session ID
        - chunkIndex: sequential chunk number

    Returns:
    {
        "segments": [
            {"speaker": "Alice", "speakerIndex": 0, "text": "Hello everyone", "start": 0.0, "end": 2.1},
            {"speaker": "Speaker 2", "speakerIndex": 1, "text": "Hi there", "start": 2.5, "end": 4.0}
        ],
        "session": { ... session state ... }
    }
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    session_id = request.form.get("sessionId", "default")
    chunk_index = int(request.form.get("chunkIndex", 0))

    session = get_session(session_id)

    # Save the audio chunk to a temp file
    ext = audio_file.filename.rsplit(".", 1)[-1] if "." in (audio_file.filename or "") else "webm"
    temp_filename = f"chunk_{session_id}_{chunk_index}_{int(time.time())}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)

    try:
        audio_file.save(temp_path)

        # Convert webm to wav if needed (Whisper works best with wav)
        wav_path = temp_path
        if ext in ("webm", "ogg", "opus"):
            wav_path = convert_to_wav(temp_path)

        # Transcribe and diarize
        segments = transcribe_and_diarize(wav_path, session)

        # Merge consecutive same-speaker segments
        merged = merge_consecutive_speakers(segments)

        # Update session state
        session.chunks_processed += 1
        session.transcript.extend(merged)

        # Add to session transcript
        for seg in merged:
            seg["chunkIndex"] = chunk_index

        return jsonify({
            "segments": merged,
            "session": session.to_dict(),
            "chunkIndex": chunk_index
        })

    except Exception as e:
        print(f"❌ Transcription error for chunk {chunk_index}: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            "error": str(e),
            "segments": [],
            "session": session.to_dict()
        }), 500

    finally:
        # Cleanup temp files
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@app.route("/ai/session/<session_id>/transcript", methods=["GET"])
def get_transcript(session_id):
    """Get the full transcript for a session."""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        session = sessions[session_id]
        return jsonify({
            "transcript": session.transcript,
            "session": session.to_dict()
        })


@app.route("/ai/session/<session_id>/stop", methods=["POST"])
def stop_session(session_id):
    """
    Stop a meeting session. Returns the final transcript.
    Session data is kept in memory for retrieval but marked as ended.
    """
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        session = sessions[session_id]
        return jsonify({
            "message": "Session stopped",
            "transcript": session.transcript,
            "session": session.to_dict()
        })


# ══════════════════════════════════════════════════════
#  SAVED VOICE PROFILES (PERSON 1 & PERSON 2 REGISTRATION)
# ══════════════════════════════════════════════════════

@app.route("/ai/voice/register", methods=["POST"])
def register_voice_profile():
    """
    Register a person's voice (e.g. 'Person 1', 'Person 2') and save to persistent storage.
    Accepts:
        - audio: Audio file (wav, webm, ogg, mp3)
        - speakerName / userName: Display name for the person (e.g. "Person 1", "Person 2")
    """
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    speaker_name = (
        request.form.get("speakerName") or
        request.form.get("userName") or
        request.form.get("name") or
        "Person 1"
    ).strip()

    if not speaker_name:
        return jsonify({"error": "speakerName is required"}), 400

    ext = audio_file.filename.rsplit(".", 1)[-1] if "." in (audio_file.filename or "") else "webm"
    temp_filename = f"reg_{speaker_name.replace(' ', '_')}_{int(time.time())}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    wav_path = None

    try:
        audio_file.save(temp_path)
        wav_path = convert_to_wav(temp_path)

        from voice_intelligence.overlap_suite.voice_fingerprint import get_saved_registry
        reg = get_saved_registry()
        fp = reg.register_speaker(speaker_name, wav_path, save=True)

        # Also store to MongoDB if available
        try:
            from ai_server import embeddings
            embeddings.store_embedding(
                user_id=speaker_name.lower().replace(" ", "_"),
                user_name=speaker_name,
                embedding_vector=fp.vector,
                sample_duration=fp.duration_sec
            )
        except Exception:
            pass

        return jsonify({
            "success": True,
            "speakerName": speaker_name,
            "duration": round(fp.duration_sec, 2),
            "dimension": fp.dimension,
            "extractionMode": fp.extraction_mode,
            "message": f"Voice registered successfully for '{speaker_name}' and saved to persistent data."
        }), 201

    except Exception as e:
        print(f"❌ Error registering voice for {speaker_name}: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if wav_path and wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@app.route("/ai/voice/saved-profiles", methods=["GET"])
def list_saved_voice_profiles():
    """List all registered voice profiles from saved data."""
    try:
        from voice_intelligence.overlap_suite.voice_fingerprint import get_saved_registry
        reg = get_saved_registry()
        profiles = []
        for spk_id, fp in reg.all_profiles().items():
            p_dict = fp.to_dict()
            p_dict.pop("vector", None)  # Don't send huge raw vector array
            profiles.append(p_dict)
        return jsonify({
            "count": len(profiles),
            "profiles": profiles
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ai/voice/recognize", methods=["POST"])
def recognize_audio_voice():
    """Identify which registered person is speaking in an audio file."""
    if "audio" not in request.files:
        return jsonify({"error": "No audio file provided"}), 400

    audio_file = request.files["audio"]
    ext = audio_file.filename.rsplit(".", 1)[-1] if "." in (audio_file.filename or "") else "webm"
    temp_filename = f"rec_{int(time.time())}.{ext}"
    temp_path = os.path.join(UPLOAD_FOLDER, temp_filename)
    wav_path = None

    try:
        audio_file.save(temp_path)
        wav_path = convert_to_wav(temp_path)

        from voice_intelligence.overlap_suite.voice_fingerprint import get_saved_registry
        reg = get_saved_registry()
        best_spk, conf, all_scores = reg.recognize(wav_path, threshold=0.38)

        return jsonify({
            "speaker": best_spk,
            "confidence": conf,
            "scores": all_scores
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            if wav_path and wav_path != temp_path and os.path.exists(wav_path):
                os.remove(wav_path)
        except Exception:
            pass


@app.route("/ai/voice/saved-profile/<speaker_name>", methods=["DELETE"])
def delete_saved_voice_profile(speaker_name):
    """Delete a registered voice profile from saved data."""
    try:
        from voice_intelligence.overlap_suite.voice_fingerprint import get_saved_registry
        reg = get_saved_registry()
        deleted = reg.delete(speaker_name, save=True)
        if deleted:
            return jsonify({"success": True, "message": f"Profile '{speaker_name}' deleted."}), 200
        else:
            return jsonify({"error": f"Profile '{speaker_name}' not found."}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── Audio Conversion Helper ──

def convert_to_wav(input_path):
    """
    Convert audio file to WAV format using ffmpeg or pydub.
    Falls back to returning the original path if conversion fails.
    """
    output_path = input_path.rsplit(".", 1)[0] + ".wav"

    # Try ffmpeg first (most reliable)
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1", "-f", "wav", output_path],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(output_path):
            return output_path
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Try pydub as fallback
    try:
        from pydub import AudioSegment
        audio = AudioSegment.from_file(input_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(output_path, format="wav")
        if os.path.exists(output_path):
            return output_path
    except ImportError:
        pass
    except Exception as e:
        print(f"⚠️  pydub conversion error: {e}")

    # Return original if conversion fails
    print(f"⚠️  Could not convert {input_path} to WAV — using original")
    return input_path


# ── Serve Frontend (for standalone testing) ──

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "frontend")


@app.route("/")
def serve_test_page():
    """Serve the standalone test page."""
    return send_from_directory(FRONTEND_DIR, "ai_meeting_test.html")


@app.route("/<path:filename>")
def serve_static(filename):
    return send_from_directory(FRONTEND_DIR, filename)


# ── Run ──
if __name__ == "__main__":
    host = os.getenv("AI_HOST", "0.0.0.0")
    port = int(os.getenv("AI_PORT", 5001))
    debug = os.getenv("AI_DEBUG", "True").lower() == "true"

    print(f"\n{'=' * 55}")
    print(f"  * AirScribe AI Meeting Server")
    print(f"    Running at http://localhost:{port}")
    print(f"    Whisper model: {WHISPER_MODEL_SIZE}")
    print(f"    Audio temp folder: {UPLOAD_FOLDER}")
    has_enrollment = 'voice_enrollment' in [bp.name for bp in app.iter_blueprints()]
    print(f"    Voice enrollment: {'enabled' if has_enrollment else 'disabled'}")
    print(f"    Routes: /ai/health, /ai/session/*, /ai/transcribe")
    print(f"            /ai/voice/register, /ai/voice/saved-profiles, /ai/voice/recognize")
    if has_enrollment:
        print(f"            /ai/voice/enroll, /ai/voice/profiles, /ai/voice/profile/*")
    print(f"{'=' * 55}\n")

    app.run(host=host, port=port, debug=debug)
