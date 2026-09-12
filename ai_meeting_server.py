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

# ── Feature #3: Multi-Modal Timeline Fusion ──
try:
    from timeline_fusion import (
        MeetingTimeline, create_event,
        WhisperTimestampNormalizer,
        correct_ocr_timestamp, correct_whisper_chunk_timestamp,
    )
    TIMELINE_AVAILABLE = True
except ImportError:
    TIMELINE_AVAILABLE = False
    print("\u26a0\ufe0f  timeline_fusion not available \u2014 timeline features disabled")

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

        # ── Feature #3: Multi-Modal Timeline ──
        self.meeting_start_ms = time.time() * 1000
        if TIMELINE_AVAILABLE:
            self.timeline = MeetingTimeline(meeting_start_ms=self.meeting_start_ms)
        else:
            self.timeline = None

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
                # If pipeline returned empty, fall through to existing logic
            except ImportError:
                pass  # ai_server not installed — use existing logic
            except Exception as e:
                print(f"\u26a0\ufe0f  Enhanced pipeline error: {e} \u2014 falling back")

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

        # Step 2: Speaker diarization
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
    Merge consecutive segments from the same speaker into single blocks.
    Preserves message order and avoids mixing lines between speakers.
    """
    if not segments:
        return segments

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speakerIndex"] == prev["speakerIndex"]:
            # Same speaker — merge text
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
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

    print(f"* New meeting session: {session_id} ({speaker_count} speakers)")
    return jsonify(session.to_dict()), 201


@app.route("/ai/session/<session_id>", methods=["GET"])
def get_session_info(session_id):
    """Get current session state including speaker mappings."""
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        return jsonify(sessions[session_id].to_dict())


@app.route("/ai/session/<session_id>/rename-speaker", methods=["POST"])
def rename_speaker(session_id):
    """
    Manually rename a speaker (preserves existing manual naming feature).

    Request body:
    {
        "speakerIndex": 0,
        "name": "Alice"
    }

    This will NEVER be overwritten by AI detection — manual names always win.
    """
    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404

        session = sessions[session_id]
        data = request.json or {}
        idx = data.get("speakerIndex", 0)
        name = data.get("name", "")

        if name.strip():
            session.set_manual_name(idx, name)
            return jsonify({
                "success": True,
                "speakerIndex": idx,
                "name": session.get_speaker_name(idx),
                "isManual": True
            })
        else:
            # Remove manual override → revert to auto label
            session.manual_names.pop(idx, None)
            return jsonify({
                "success": True,
                "speakerIndex": idx,
                "name": session.get_speaker_name(idx),
                "isManual": False
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

        # ── Feature #3: Add transcript events to timeline ──
        if TIMELINE_AVAILABLE and session.timeline:
            normalizer = session.timeline.normalizer
            # Estimate chunk duration from segments or default to CHUNK_INTERVAL
            if merged:
                chunk_duration = max(s.get("end", 8.0) for s in merged)
            else:
                chunk_duration = 8.0
            normalizer.register_chunk(chunk_index, chunk_duration)

            for seg in merged:
                chunk_rel_start = seg.get("start", 0.0)
                abs_ts = normalizer.normalize(chunk_index, chunk_rel_start)
                event = create_event(
                    event_type="transcript",
                    source="whisper",
                    content=seg.get("text", ""),
                    timestamp=abs_ts,
                    metadata={
                        "speaker": seg.get("speaker", "Unknown"),
                        "speakerIndex": seg.get("speakerIndex", 0),
                        "chunkIndex": chunk_index,
                        "start": seg.get("start", 0.0),
                        "end": seg.get("end", 0.0),
                        "confidence": seg.get("confidence", 0.0),
                    },
                )
                # Apply correction metadata
                correction = correct_whisper_chunk_timestamp(
                    raw_chunk_relative_ms=chunk_rel_start * 1000,
                    normalized_meeting_ms=abs_ts,
                    chunk_index=chunk_index,
                )
                if correction:
                    event.raw_timestamp = correction.get("raw_timestamp", event.raw_timestamp)
                    event.corrected_timestamp = correction.get("corrected_timestamp")
                    event.timestamp_correction = correction.get("timestamp_correction", 0.0)
                    event.correction_reason = correction.get("correction_reason")
                    event.correction_confidence = correction.get("correction_confidence")

                session.timeline.add_event(event)

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
#  FEATURE #3: MULTI-MODAL TIMELINE ENDPOINTS
# ══════════════════════════════════════════════════════

@app.route("/ai/session/<session_id>/timeline", methods=["GET"])
def get_timeline(session_id):
    """Get the unified chronological timeline for a session."""
    if not TIMELINE_AVAILABLE:
        return jsonify({"error": "Timeline feature not available"}), 501

    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        session = sessions[session_id]

    if not session.timeline:
        return jsonify({"error": "Timeline not initialized"}), 500

    event_type = request.args.get("type")
    since = request.args.get("since", type=float)
    limit = request.args.get("limit", type=int)

    events = session.timeline.get_events(event_type=event_type, since_ms=since, limit=limit)
    return jsonify({
        "events": [e.to_dict() for e in events],
        "counts": session.timeline.get_event_count(),
        "meetingStartMs": session.meeting_start_ms,
    })


@app.route("/ai/session/<session_id>/timeline/note", methods=["POST"])
def add_timeline_note(session_id):
    """
    Add a typed note event to the meeting timeline.

    Request body: { "content": "...", "timestamp": <epoch_ms> (optional) }
    """
    if not TIMELINE_AVAILABLE:
        return jsonify({"error": "Timeline feature not available"}), 501

    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        session = sessions[session_id]

    if not session.timeline:
        return jsonify({"error": "Timeline not initialized"}), 500

    data = request.json or {}
    content = data.get("content", "").strip()
    if not content:
        return jsonify({"error": "No content provided"}), 400

    ts = data.get("timestamp", time.time() * 1000)

    event = create_event(
        event_type="note",
        source="user",
        content=content,
        timestamp=ts,
    )
    session.timeline.add_event(event)

    return jsonify({"success": True, "event": event.to_dict()}), 201


@app.route("/ai/session/<session_id>/timeline/screen", methods=["POST"])
def add_timeline_screen(session_id):
    """
    Add a screen capture + OCR event to the meeting timeline.

    Request body:
    {
        "captureTimestamp": <epoch_ms>,     // REQUIRED: when screenshot was taken
        "ocrText": "...",                   // optional: extracted text
        "ocrCompletionTimestamp": <epoch_ms> // optional: when OCR finished
    }

    The captureTimestamp is authoritative for timeline positioning.
    """
    if not TIMELINE_AVAILABLE:
        return jsonify({"error": "Timeline feature not available"}), 501

    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        session = sessions[session_id]

    if not session.timeline:
        return jsonify({"error": "Timeline not initialized"}), 500

    data = request.json or {}
    capture_ts = data.get("captureTimestamp")
    if not capture_ts:
        return jsonify({"error": "captureTimestamp is required"}), 400

    ocr_text = data.get("ocrText", "")
    ocr_done_ts = data.get("ocrCompletionTimestamp")

    # Add screen capture event at capture time
    screen_event = create_event(
        event_type="screen",
        source="screen_capture",
        content="[Screen Capture]",
        timestamp=capture_ts,
    )
    session.timeline.add_event(screen_event)

    result = {"success": True, "screenEvent": screen_event.to_dict()}

    # Add OCR event with timestamp correction if OCR text provided
    if ocr_text:
        if ocr_done_ts and ocr_done_ts != capture_ts:
            correction = correct_ocr_timestamp(capture_ts, ocr_done_ts)
            ocr_event = create_event(
                event_type="ocr",
                source="ocr_engine",
                content=ocr_text,
                timestamp=correction["timestamp"],
                metadata={"processing_delay_ms": ocr_done_ts - capture_ts},
            )
            ocr_event.raw_timestamp = correction["raw_timestamp"]
            ocr_event.corrected_timestamp = correction.get("corrected_timestamp")
            ocr_event.timestamp_correction = correction["timestamp_correction"]
            ocr_event.correction_reason = correction.get("correction_reason")
            ocr_event.correction_confidence = correction.get("correction_confidence")
        else:
            ocr_event = create_event(
                event_type="ocr",
                source="ocr_engine",
                content=ocr_text,
                timestamp=capture_ts,
            )
        session.timeline.add_event(ocr_event)
        result["ocrEvent"] = ocr_event.to_dict()

    return jsonify(result), 201


@app.route("/ai/session/<session_id>/timeline/export", methods=["GET"])
def export_timeline(session_id):
    """Export the full timeline as JSON."""
    if not TIMELINE_AVAILABLE:
        return jsonify({"error": "Timeline feature not available"}), 501

    with sessions_lock:
        if session_id not in sessions:
            return jsonify({"error": "Session not found"}), 404
        session = sessions[session_id]

    if not session.timeline:
        return jsonify({"error": "Timeline not initialized"}), 500

    return jsonify({
        "sessionId": session_id,
        "meetingStartMs": session.meeting_start_ms,
        "events": session.timeline.to_list(),
        "counts": session.timeline.get_event_count(),
        "exportedAt": datetime.utcnow().isoformat() + "Z",
    })


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
    if has_enrollment:
        print(f"            /ai/voice/enroll, /ai/voice/profiles, /ai/voice/profile/*")
    print(f"{'=' * 55}\n")

    # Pre-load models if desired (uncomment to load at startup)
    # get_whisper_model()
    # get_diarization_pipeline()

    app.run(host=host, port=port, debug=debug)
