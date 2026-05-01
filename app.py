"""
AirScribe Backend — Flask + MongoDB
Full REST API for the AirScribe smart notes application.
"""

from flask import Flask, jsonify, request, send_from_directory, session, redirect
from flask_cors import CORS
from pymongo import MongoClient
from bson.objectid import ObjectId
from datetime import datetime, timedelta
import os
import secrets
import json as json_module
import urllib.request
from functools import wraps
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
try:
    from flask_dance.contrib.google import make_google_blueprint, google
except ImportError:
    google = None
    make_google_blueprint = None

load_dotenv()

# ── App Setup ──
app = Flask(__name__, static_folder=None)
app.secret_key = os.getenv("SECRET_KEY", secrets.token_hex(32))

# ── Persistent session (7 days) ──
app.permanent_session_lifetime = timedelta(days=7)

# ── Secure cookie config ──
ENV_MODE = os.getenv("ENV", "development").lower()
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = (ENV_MODE == "production")

CORS(app, resources={r"/api/*": {"origins": "*"}, r"/auth/*": {"origins": "*"}})

if make_google_blueprint:
    # Allow insecure transport for local dev
    os.environ["OAUTHLIB_INSECURE_TRANSPORT"] = "1"
    
    google_bp = make_google_blueprint(
        client_id=os.getenv("GOOGLE_CLIENT_ID"),
        client_secret=os.getenv("GOOGLE_CLIENT_SECRET"),
        scope=["profile", "email"],
        redirect_url="/login/google/authorized"
    )
    app.register_blueprint(google_bp, url_prefix="/login")

# ── MongoDB Setup ──
MONGO_URI = os.getenv("MONGO_URI", "mongodb+srv://archit:archit123@airscribe.c5pcugl.mongodb.net/airscribe?retryWrites=true&w=majority")
DB_NAME = os.getenv("DB_NAME", "airscribe")

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

# Collections
notes_col = db["notes"]
trash_col = db["trash"]
folders_col = db["folders"]
tags_col = db["tags"]
activity_col = db["activity"]
analytics_col = db["analytics"]
users_col = db["users"]


# ── Helpers ──
def serialize_doc(doc):
    """Convert MongoDB document to JSON-serializable dict."""
    if doc is None:
        return None
    doc["_id"] = str(doc["_id"])
    return doc


def serialize_list(cursor):
    """Convert a MongoDB cursor to a list of serialized dicts."""
    return [serialize_doc(doc) for doc in cursor]


def get_analytics():
    """Get or create the singleton analytics document."""
    doc = analytics_col.find_one({"_type": "analytics"})
    if not doc:
        doc = {"_type": "analytics", "daily": {}, "hourly": {}, "streak": 0, "lastWriteDate": None}
        analytics_col.insert_one(doc)
        doc = analytics_col.find_one({"_type": "analytics"})
    return doc


# ── Seed default data if empty ──
def seed_defaults():
    """Seed the database with default folders and tags if they don't exist."""
    if folders_col.count_documents({}) == 0:
        default_folders = [
            {"name": "General", "color": "#4fd1ff"},
            {"name": "Study", "color": "#a78bfa"},
            {"name": "Work", "color": "#34d399"},
            {"name": "Personal", "color": "#fbbf24"},
        ]
        folders_col.insert_many(default_folders)
        print("✓ Seeded default folders")

    if tags_col.count_documents({}) == 0:
        default_tags = ["college", "ideas", "project", "important", "todo"]
        tags_col.insert_many([{"name": t} for t in default_tags])
        print("✓ Seeded default tags")

    get_analytics()
    print("✓ Analytics initialized")


# ═══════════════════════════════════════════════════
#                    API ROUTES
# ═══════════════════════════════════════════════════

# ── NOTES ──

@app.route("/api/notes", methods=["GET"])
def get_notes():
    """Get all notes, sorted by most recently updated."""
    notes = list(notes_col.find().sort("updatedAt", -1))
    return jsonify(serialize_list(notes))


@app.route("/api/notes/<note_id>", methods=["GET"])
def get_note(note_id):
    """Get a single note by ID."""
    note = notes_col.find_one({"_id": ObjectId(note_id)})
    if not note:
        return jsonify({"error": "Note not found"}), 404
    return jsonify(serialize_doc(note))


@app.route("/api/notes", methods=["POST"])
def create_note():
    """Create a new note."""
    data = request.json
    now = datetime.utcnow().isoformat() + "Z"
    note = {
        "name": data.get("name", "Untitled"),
        "text": data.get("text", ""),
        "content": data.get("content", ""),
        "folder": data.get("folder", "General"),
        "color": data.get("color", ""),
        "reminder": data.get("reminder"),
        "pinned": data.get("pinned", False),
        "tags": data.get("tags", []),
        "linkedNotes": data.get("linkedNotes", []),
        "createdAt": data.get("createdAt", now),
        "updatedAt": data.get("updatedAt", now),
    }
    result = notes_col.insert_one(note)
    note["_id"] = str(result.inserted_id)
    return jsonify(note), 201


@app.route("/api/notes/<note_id>", methods=["PUT"])
def update_note(note_id):
    """Update an existing note."""
    data = request.json
    now = datetime.utcnow().isoformat() + "Z"
    update_data = {
        "name": data.get("name", "Untitled"),
        "text": data.get("text", ""),
        "content": data.get("content", ""),
        "folder": data.get("folder", "General"),
        "color": data.get("color", ""),
        "reminder": data.get("reminder"),
        "pinned": data.get("pinned", False),
        "tags": data.get("tags", []),
        "linkedNotes": data.get("linkedNotes", []),
        "updatedAt": data.get("updatedAt", now),
    }
    result = notes_col.update_one(
        {"_id": ObjectId(note_id)},
        {"$set": update_data}
    )
    if result.matched_count == 0:
        return jsonify({"error": "Note not found"}), 404
    updated = notes_col.find_one({"_id": ObjectId(note_id)})
    return jsonify(serialize_doc(updated))


@app.route("/api/notes/<note_id>", methods=["DELETE"])
def delete_note(note_id):
    """Move a note to trash (soft delete)."""
    note = notes_col.find_one({"_id": ObjectId(note_id)})
    if not note:
        return jsonify({"error": "Note not found"}), 404

    # Move to trash
    now = datetime.utcnow().isoformat() + "Z"
    note["deletedAt"] = now
    note["originalId"] = str(note["_id"])
    del note["_id"]
    trash_col.insert_one(note)

    # Remove from notes
    notes_col.delete_one({"_id": ObjectId(note_id)})
    return jsonify({"message": "Note moved to trash"})


@app.route("/api/notes/<note_id>/pin", methods=["PATCH"])
def toggle_pin(note_id):
    """Toggle the pinned status of a note."""
    note = notes_col.find_one({"_id": ObjectId(note_id)})
    if not note:
        return jsonify({"error": "Note not found"}), 404

    new_pin = not note.get("pinned", False)
    notes_col.update_one(
        {"_id": ObjectId(note_id)},
        {"$set": {"pinned": new_pin}}
    )
    return jsonify({"pinned": new_pin})


@app.route("/api/notes/<note_id>/duplicate", methods=["POST"])
def duplicate_note(note_id):
    """Duplicate a note."""
    note = notes_col.find_one({"_id": ObjectId(note_id)})
    if not note:
        return jsonify({"error": "Note not found"}), 404

    now = datetime.utcnow().isoformat() + "Z"
    new_note = {
        "name": note["name"] + " (Copy)",
        "text": note.get("text", ""),
        "content": note.get("content", ""),
        "folder": note.get("folder", "General"),
        "color": note.get("color", ""),
        "reminder": None,
        "pinned": False,
        "tags": note.get("tags", []),
        "linkedNotes": note.get("linkedNotes", []),
        "createdAt": now,
        "updatedAt": now,
    }
    result = notes_col.insert_one(new_note)
    new_note["_id"] = str(result.inserted_id)
    return jsonify(new_note), 201


# ── TRASH ──

@app.route("/api/trash", methods=["GET"])
def get_trash():
    """Get all trashed notes."""
    trashed = list(trash_col.find().sort("deletedAt", -1))
    return jsonify(serialize_list(trashed))


@app.route("/api/trash/<trash_id>/restore", methods=["POST"])
def restore_from_trash(trash_id):
    """Restore a note from trash back to notes."""
    trashed = trash_col.find_one({"_id": ObjectId(trash_id)})
    if not trashed:
        return jsonify({"error": "Trashed note not found"}), 404

    now = datetime.utcnow().isoformat() + "Z"
    restored = {
        "name": trashed["name"],
        "text": trashed.get("text", ""),
        "content": trashed.get("content", ""),
        "folder": trashed.get("folder", "General"),
        "color": trashed.get("color", ""),
        "reminder": trashed.get("reminder"),
        "pinned": trashed.get("pinned", False),
        "tags": trashed.get("tags", []),
        "linkedNotes": trashed.get("linkedNotes", []),
        "createdAt": trashed.get("createdAt", now),
        "updatedAt": now,
    }
    notes_col.insert_one(restored)
    trash_col.delete_one({"_id": ObjectId(trash_id)})
    return jsonify({"message": "Note restored"})


@app.route("/api/trash/<trash_id>", methods=["DELETE"])
def permanent_delete(trash_id):
    """Permanently delete a note from trash."""
    result = trash_col.delete_one({"_id": ObjectId(trash_id)})
    if result.deleted_count == 0:
        return jsonify({"error": "Trashed note not found"}), 404
    return jsonify({"message": "Note permanently deleted"})


@app.route("/api/trash/empty", methods=["DELETE"])
def empty_trash():
    """Empty all trash."""
    result = trash_col.delete_many({})
    return jsonify({"message": f"Deleted {result.deleted_count} notes from trash"})


# ── FOLDERS ──

@app.route("/api/folders", methods=["GET"])
def get_folders():
    """Get all folders."""
    folders = list(folders_col.find())
    # Add note count per folder
    for f in folders:
        f["count"] = notes_col.count_documents({"folder": f["name"]})
    return jsonify(serialize_list(folders))


@app.route("/api/folders", methods=["POST"])
def create_folder():
    """Create a new folder."""
    data = request.json
    name = data.get("name", "").strip()
    color = data.get("color", "#4fd1ff")

    if not name:
        return jsonify({"error": "Folder name is required"}), 400

    existing = folders_col.find_one({"name": {"$regex": f"^{name}$", "$options": "i"}})
    if existing:
        return jsonify({"error": "Folder already exists"}), 409

    folder = {"name": name, "color": color}
    result = folders_col.insert_one(folder)
    folder["_id"] = str(result.inserted_id)
    folder["count"] = 0
    return jsonify(folder), 201


@app.route("/api/folders/<folder_id>", methods=["DELETE"])
def delete_folder(folder_id):
    """Delete a folder. Notes in this folder are moved to 'General'."""
    folder = folders_col.find_one({"_id": ObjectId(folder_id)})
    if not folder:
        return jsonify({"error": "Folder not found"}), 404

    if folder["name"] == "General":
        return jsonify({"error": "Cannot delete the General folder"}), 400

    # Move notes to General
    notes_col.update_many(
        {"folder": folder["name"]},
        {"$set": {"folder": "General"}}
    )
    folders_col.delete_one({"_id": ObjectId(folder_id)})
    return jsonify({"message": f"Folder '{folder['name']}' deleted"})


# ── TAGS ──

@app.route("/api/tags", methods=["GET"])
def get_tags():
    """Get all tags."""
    tags = list(tags_col.find())
    return jsonify(serialize_list(tags))


@app.route("/api/tags", methods=["POST"])
def create_tag():
    """Create a new tag."""
    data = request.json
    name = data.get("name", "").strip().lower()
    if not name:
        return jsonify({"error": "Tag name is required"}), 400

    existing = tags_col.find_one({"name": name})
    if existing:
        return jsonify(serialize_doc(existing))

    tag = {"name": name}
    result = tags_col.insert_one(tag)
    tag["_id"] = str(result.inserted_id)
    return jsonify(tag), 201


@app.route("/api/tags/<tag_id>", methods=["DELETE"])
def delete_tag(tag_id):
    """Delete a tag."""
    result = tags_col.delete_one({"_id": ObjectId(tag_id)})
    if result.deleted_count == 0:
        return jsonify({"error": "Tag not found"}), 404
    return jsonify({"message": "Tag deleted"})


# ── ACTIVITY ──

@app.route("/api/activity", methods=["GET"])
def get_activity():
    """Get recent activity log (max 60 entries)."""
    activities = list(activity_col.find().sort("time", -1).limit(60))
    return jsonify(serialize_list(activities))


@app.route("/api/activity", methods=["POST"])
def log_activity():
    """Log a new activity entry."""
    data = request.json
    entry = {
        "msg": data.get("msg", ""),
        "color": data.get("color", "#4fd1ff"),
        "time": data.get("time", datetime.utcnow().isoformat() + "Z"),
    }
    activity_col.insert_one(entry)

    # Keep only latest 60 entries
    total = activity_col.count_documents({})
    if total > 60:
        oldest = list(activity_col.find().sort("time", 1).limit(total - 60))
        if oldest:
            ids = [doc["_id"] for doc in oldest]
            activity_col.delete_many({"_id": {"$in": ids}})

    return jsonify({"message": "Activity logged"}), 201


# ── ANALYTICS ──

@app.route("/api/analytics", methods=["GET"])
def get_analytics_route():
    """Get analytics data."""
    doc = get_analytics()
    return jsonify(serialize_doc(doc))


@app.route("/api/analytics", methods=["PUT"])
def update_analytics():
    """Update analytics data (daily, hourly, streak, etc)."""
    data = request.json
    update_fields = {}
    if "daily" in data:
        update_fields["daily"] = data["daily"]
    if "hourly" in data:
        update_fields["hourly"] = data["hourly"]
    if "streak" in data:
        update_fields["streak"] = data["streak"]
    if "lastWriteDate" in data:
        update_fields["lastWriteDate"] = data["lastWriteDate"]

    if update_fields:
        analytics_col.update_one(
            {"_type": "analytics"},
            {"$set": update_fields}
        )
    doc = get_analytics()
    return jsonify(serialize_doc(doc))


# ── DASHBOARD STATS ──

@app.route("/api/dashboard", methods=["GET"])
def get_dashboard():
    """Get aggregated dashboard stats."""
    total_notes = notes_col.count_documents({})
    folder_count = folders_col.count_documents({})
    pinned_count = notes_col.count_documents({"pinned": True})

    week_ago = (datetime.utcnow() - timedelta(days=7)).isoformat() + "Z"
    recent_count = notes_col.count_documents({
        "$or": [
            {"updatedAt": {"$gte": week_ago}},
            {"createdAt": {"$gte": week_ago}}
        ]
    })

    trash_count = trash_col.count_documents({})

    return jsonify({
        "totalNotes": total_notes,
        "folderCount": folder_count,
        "pinnedCount": pinned_count,
        "recentCount": recent_count,
        "trashCount": trash_count,
    })


# ── SEARCH ──

@app.route("/api/search", methods=["GET"])
def search_notes():
    """Search notes by query string across name, text, tags, and folder."""
    q = request.args.get("q", "").strip()
    if not q:
        return jsonify([])

    query = {
        "$or": [
            {"name": {"$regex": q, "$options": "i"}},
            {"text": {"$regex": q, "$options": "i"}},
            {"tags": {"$regex": q, "$options": "i"}},
            {"folder": {"$regex": q, "$options": "i"}},
        ]
    }
    results = list(notes_col.find(query).limit(20))
    return jsonify(serialize_list(results))


# ── EXPORT ALL NOTES ──

@app.route("/api/export", methods=["GET"])
def export_all():
    """Export all notes, folders, and tags as JSON."""
    notes = serialize_list(notes_col.find())
    folders = serialize_list(folders_col.find())
    tags = [doc["name"] for doc in tags_col.find()]
    return jsonify({
        "notes": notes,
        "folders": folders,
        "tags": tags,
        "exportedAt": datetime.utcnow().isoformat() + "Z",
    })


# ── IMPORT (Bulk create from localStorage migration) ──

@app.route("/api/import", methods=["POST"])
def import_data():
    """
    Bulk import data (used for migrating from localStorage).
    Expects: { notes: [], folders: [], tags: [], trash: [], activity: [], analytics: {} }
    """
    data = request.json

    imported = {"notes": 0, "folders": 0, "tags": 0, "trash": 0, "activity": 0}

    # Import notes
    if data.get("notes"):
        for n in data["notes"]:
            n.pop("_id", None)
            notes_col.insert_one(n)
            imported["notes"] += 1

    # Import folders
    if data.get("folders"):
        for f in data["folders"]:
            f.pop("_id", None)
            existing = folders_col.find_one({"name": {"$regex": f"^{f['name']}$", "$options": "i"}})
            if not existing:
                folders_col.insert_one(f)
                imported["folders"] += 1

    # Import tags
    if data.get("tags"):
        for t in data["tags"]:
            name = t if isinstance(t, str) else t.get("name", "")
            if name and not tags_col.find_one({"name": name}):
                tags_col.insert_one({"name": name})
                imported["tags"] += 1

    # Import trash
    if data.get("trash"):
        for t in data["trash"]:
            t.pop("_id", None)
            trash_col.insert_one(t)
            imported["trash"] += 1

    # Import activity
    if data.get("activity"):
        for a in data["activity"]:
            a.pop("_id", None)
            activity_col.insert_one(a)
            imported["activity"] += 1

    # Import analytics
    if data.get("analytics"):
        analytics_col.update_one(
            {"_type": "analytics"},
            {"$set": data["analytics"]},
            upsert=True
        )

    return jsonify({"message": "Import complete", "imported": imported}), 201


# ── MEETING STREAM (Live Meeting Recording) ──

@app.route("/api/meeting-stream", methods=["POST"])
def meeting_stream():
    """
    Receive live meeting transcript chunks from the frontend.
    This endpoint is called during active meeting recording to stream
    partial transcripts for optional server-side processing.
    
    Payload: { text: str, timestamp: int, elapsed: int, speaker: str }
    Returns: { received: true, processedText: str (optional) }
    """
    data = request.json
    text = data.get("text", "")
    timestamp = data.get("timestamp")
    elapsed = data.get("elapsed", 0)
    speaker = data.get("speaker", "Unknown")

    if not text.strip():
        return jsonify({"received": True}), 200

    # Store the chunk in the database for later retrieval/analysis
    chunk = {
        "text": text,
        "timestamp": timestamp,
        "elapsed": elapsed,
        "speaker": speaker,
        "receivedAt": datetime.utcnow().isoformat() + "Z",
    }
    
    try:
        meeting_stream_col = db["meeting_stream"]
        meeting_stream_col.insert_one(chunk)
    except Exception as e:
        print(f"Meeting stream storage error: {e}")

    # For now, return the text as-is.
    # Future: Add NLP processing, formatting, or summarization here.
    return jsonify({
        "received": True,
        "processedText": None  # Set to processed string when NLP is added
    }), 200


# ── HEALTH CHECK ──

@app.route("/api/health", methods=["GET"])
def health():
    """Health check endpoint."""
    try:
        client.admin.command("ping")
        return jsonify({"status": "ok", "database": "connected"})
    except Exception as e:
        return jsonify({"status": "error", "database": str(e)}), 500


# ═══════════════════════════════════════════════════
#                 AUTHENTICATION SYSTEM
# ═══════════════════════════════════════════════════


def login_required_page(f):
    """Decorator: redirect to /login if not authenticated (for HTML pages)."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user' not in session:
            return redirect('/login')
        return f(*args, **kwargs)
    return decorated


# ── Auth Config ──

@app.route("/auth/config", methods=["GET"])
def auth_config():
    """Return Google Client ID for frontend Google Sign-In."""
    return jsonify({"google_client_id": os.getenv("GOOGLE_CLIENT_ID", "")})


@app.route("/auth/status", methods=["GET"])
def auth_status():
    """Check if user is logged in and return user info."""
    if 'user' in session:
        return jsonify({"authenticated": True, "user": session['user']})
    return jsonify({"authenticated": False})


# ── Register ──

@app.route("/auth/register", methods=["POST"])
def auth_register():
    """Register a new user with email and password."""
    data = request.json
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")
    name = data.get("name", "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters"}), 400
    if not name:
        name = email.split("@")[0]

    existing = users_col.find_one({"email": email})
    if existing:
        return jsonify({"error": "An account with this email already exists"}), 409

    now = datetime.utcnow().isoformat() + "Z"
    user = {
        "email": email,
        "name": name,
        "picture": "",
        "password": generate_password_hash(password),
        "provider": "local",
        "created_at": now
    }
    result = users_col.insert_one(user)

    session.permanent = True
    session['user'] = {
        "id": str(result.inserted_id),
        "email": email,
        "name": name,
        "picture": ""
    }
    return jsonify({"message": "Account created", "user": session['user']}), 201


# ── Login ──

@app.route("/auth/login", methods=["POST"])
def auth_login():
    """Login with email and password."""
    data = request.json
    email = data.get("email", "").strip().lower()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    user = users_col.find_one({"email": email})
    if not user or not user.get("password"):
        return jsonify({"error": "Invalid email or password"}), 401

    if not check_password_hash(user["password"], password):
        return jsonify({"error": "Invalid email or password"}), 401

    session.permanent = True
    session['user'] = {
        "id": str(user["_id"]),
        "email": user["email"],
        "name": user.get("name", ""),
        "picture": user.get("picture", "")
    }
    return jsonify({"message": "Login successful", "user": session['user']})


# ── Google OAuth ──

@app.route("/auth/google", methods=["POST"])
def auth_google():
    """Verify Google ID token and login/register user."""
    data = request.json
    token = data.get("credential", "")

    if not token:
        return jsonify({"error": "No credential provided"}), 400

    # Verify token with Google's tokeninfo endpoint
    try:
        url = f'https://oauth2.googleapis.com/tokeninfo?id_token={token}'
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=10) as response:
            token_info = json_module.loads(response.read().decode())
    except Exception as e:
        print(f"Google token verification error: {e}")
        return jsonify({"error": "Invalid Google token"}), 401

    # Verify the token audience matches our client ID
    client_id = os.getenv("GOOGLE_CLIENT_ID", "")
    if client_id and token_info.get("aud") != client_id:
        return jsonify({"error": "Token not intended for this app"}), 401

    email = token_info.get("email", "")
    name = token_info.get("name", "")
    picture = token_info.get("picture", "")

    if not email:
        return jsonify({"error": "Could not get email from Google"}), 400

    # Find or create user
    now = datetime.utcnow().isoformat() + "Z"
    user = users_col.find_one({"email": email})

    if user:
        users_col.update_one(
            {"email": email},
            {"$set": {"name": name, "picture": picture}}
        )
        user_id = str(user["_id"])
    else:
        new_user = {
            "email": email, "name": name, "picture": picture,
            "password": None, "provider": "google", "created_at": now
        }
        result = users_col.insert_one(new_user)
        user_id = str(result.inserted_id)

    session.permanent = True
    session['user'] = {
        "id": user_id, "email": email,
        "name": name, "picture": picture
    }
    return jsonify({"message": "Google login successful", "user": session['user']})


@app.route("/login/google/authorized", methods=["GET", "POST"])
def google_authorized_redirect():
    """Handle Google OAuth redirect URI via Flask-Dance."""
    if not google or not google.authorized:
        return redirect("/login")
    
    resp = google.get("/oauth2/v2/userinfo")
    if not resp.ok:
        return redirect("/login")
    
    info = resp.json()
    email = info.get("email", "")
    name = info.get("name", "")
    picture = info.get("picture", "")
    
    if not email:
        return redirect("/login")
        
    now = datetime.utcnow().isoformat() + "Z"
    user = users_col.find_one({"email": email})
    
    if user:
        users_col.update_one(
            {"email": email},
            {"$set": {"name": name, "picture": picture}}
        )
        user_id = str(user["_id"])
    else:
        new_user = {
            "email": email, "name": name, "picture": picture,
            "password": None, "provider": "google", "created_at": now
        }
        result = users_col.insert_one(new_user)
        user_id = str(result.inserted_id)

    session.permanent = True
    session['user'] = {
        "id": user_id, "email": email,
        "name": name, "picture": picture
    }
    
    return redirect("/")


# ── Logout ──

@app.route("/auth/logout")
def auth_logout():
    """Logout and clear session."""
    session.clear()
    return redirect('/login')


# ═══════════════════════════════════════════════════
#              SERVE FRONTEND (with auth)
# ═══════════════════════════════════════════════════

FRONTEND_DIR = os.path.dirname(os.path.abspath(__file__))


@app.route("/login")
def serve_login_page():
    """Serve login page (no auth required)."""
    if 'user' in session:
        return redirect('/')
    return send_from_directory(FRONTEND_DIR, "login.html")


@app.route("/")
@login_required_page
def serve_home():
    return send_from_directory(FRONTEND_DIR, "home.html")


@app.route("/<path:filename>")
def serve_static(filename):
    # Protected HTML pages require login
    protected = ['home.html', 'AirScribe_v3.html']
    if filename in protected and 'user' not in session:
        return redirect('/login')
    return send_from_directory(FRONTEND_DIR, filename)


# ── Seed defaults on startup (works with both python and gunicorn) ──
seed_defaults()

# ── Run ──
if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", 5000))
    debug = os.getenv("DEBUG", "True").lower() == "true" and ENV_MODE != "production"
    print(f"\n✦ AirScribe Backend running at http://localhost:{port}")
    print(f"  Database: {DB_NAME} @ {MONGO_URI}\n")
    app.run(host=host, port=port, debug=debug)
