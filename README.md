# AirScribe AI Meeting — Multi-Speaker Transcription

AI-powered voice recognition and multi-speaker transcription enhancement for AirScribe's Live Meeting feature.

## 📁 Folder Structure

```
AirScribe_AI_Meeting/
├── backend/
│   ├── ai_meeting_server.py    # Flask server (port 5001) — Whisper + diarization
│   ├── requirements.txt        # Python dependencies
│   ├── .env                    # Configuration (model size, HF token)
│   └── temp_audio/             # Temporary audio files (auto-created)
├── frontend/
│   ├── ai_meeting.js           # Frontend module — MediaRecorder + UI
│   ├── ai_meeting.css          # Floating panel styles
│   └── ai_meeting_test.html    # Standalone test page
├── INTEGRATION_GUIDE.html      # How to integrate into AirScribe_v3.html
└── README.md                   # This file
```

## ⚙️ Installation Steps

### 1. Install Python Dependencies

```bash
cd AirScribe_AI_Meeting/backend
pip install -r requirements.txt
```

### 2. Install FFmpeg (required for audio conversion)

- **Windows**: `choco install ffmpeg` or download from https://ffmpeg.org
- **Linux**: `sudo apt install ffmpeg`
- **macOS**: `brew install ffmpeg`

### 3. (Optional) Setup Speaker Diarization

For real multi-speaker detection with pyannote:

```bash
pip install pyannote.audio torch torchaudio
```

Then get a HuggingFace token from https://huggingface.co/settings/tokens and add it to `.env`:

```
HF_TOKEN=hf_your_token_here
```

> Without pyannote, the system uses a simple silence-gap-based fallback that works for basic 2-speaker meetings.

## 🚀 How to Run

### Step 1: Start the main AirScribe backend (port 5000)

```bash
cd airscribe/backend
python app.py
```

### Step 2: Start the AI Meeting server (port 5001)

```bash
cd AirScribe_AI_Meeting/backend
python ai_meeting_server.py
```

### Step 3: Open AirScribe in your browser

```
http://localhost:5000
```

Click **Live Meeting** → Configure speakers → Start Recording. The AI transcription runs alongside the existing Web Speech API recording.

### Standalone Test

You can also test the AI system independently:

```
http://localhost:5001
```

## 🎯 How It Works

1. **User clicks "Live Meeting"** → Opens existing speaker setup modal
2. **Existing MeetingRecorder starts** → Web Speech API for real-time interim text (unchanged)
3. **AIMeeting module starts** → MediaRecorder captures audio chunks
4. **Every 8 seconds** → Audio chunk sent to AI backend
5. **Backend processes** → Whisper transcribes → Diarization assigns speakers
6. **Results return** → AI-transcribed segments inserted into Quill editor
7. **Speaker names** → Manual names always preserved; AI labels for auto-detected speakers

## ⚠️ Important Design Decisions

| Concern | Decision |
|---------|----------|
| Manual speaker names | **NEVER overwritten** — manual names always take priority |
| Existing features | **ZERO modifications** to original files |
| AI server down | **Graceful fallback** — existing Web Speech API continues working |
| Speaker consistency | Diarization labels mapped to consistent indices across chunks |
| Port conflict | AI server runs on **5001**, main backend on **5000** |

## 🔧 Configuration

Edit `backend/.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large` |
| `AI_PORT` | `5001` | Server port |
| `HF_TOKEN` | (empty) | HuggingFace token for pyannote diarization |

## 📋 Limitations

1. **Whisper processing is not real-time** — there's an 8-10 second delay for each chunk
2. **Speaker diarization accuracy** depends on audio quality and number of speakers
3. **The simple fallback diarization** uses silence gaps and works best for 2 speakers
4. **Large Whisper models** require significant GPU memory (use `base` or `small` on CPU)
5. **MediaRecorder format** (webm) requires FFmpeg for conversion to WAV
6. **Browser compatibility**: Requires Chrome, Edge, or Firefox (MediaRecorder API support)
7. **No GPU acceleration** by default — install CUDA toolkit for faster Whisper inference
