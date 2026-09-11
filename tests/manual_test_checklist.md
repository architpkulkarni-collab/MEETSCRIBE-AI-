# AirScribe — Speaker Diarization & Recognition: Manual Test Checklist

> Run these tests manually after installation to verify the full end-to-end flow.

---

## Prerequisites

- [ ] Python 3.10+ installed
- [ ] `pip install -r requirements.txt` completed
- [ ] ffmpeg installed and on PATH (`ffmpeg -version` works)
- [ ] MongoDB Atlas connection working (`app.py` starts without errors)
- [ ] `.env` file created from `.env.example` with valid `HF_TOKEN`
- [ ] AI server running: `python -m ai_server.app` on port 5001
- [ ] Main backend running: `python app.py` on port 5000

---

## Test 1: AI Server Health Check

1. Open browser: `http://localhost:5001/ai/health`
2. Verify response contains:
   - `"status": "ok"`
   - `"modules"` with whisper, diarization, recognition, enrollment sections
3. **Expected**: 200 OK with JSON status

---

## Test 2: Microphone Test

1. Navigate to AirScribe → **Voice Profile** (sidebar)
2. Click **Test** button in "Microphone Test" section
3. Allow mic permission if prompted
4. Wait 2 seconds for recording + analysis
5. **Expected**: "Microphone working (volume: XX%)" with green indicator

---

## Test 3: Voice Enrollment

1. Navigate to **Voice Profile** page
2. Click **Start Recording**
3. Speak naturally for 30 seconds (read aloud, describe your day)
4. Observe:
   - Progress ring counting down from 30
   - Mic level bars animating
   - Status showing "Recording... Xs / 30s"
5. After 30 seconds, recording stops automatically
6. Wait for "Processing your voice..." (should take < 5 seconds)
7. **Expected**: "Voice Profile Enrolled!" success message
8. Profile appears in "Enrolled Profiles" list below

---

## Test 4: Profile Appears in Database

1. Open MongoDB Atlas dashboard
2. Navigate to `airscribe` database → `voice_profiles` collection
3. **Expected**: Document exists with:
   - `userId` matching logged-in user
   - `userName` matching user's name
   - `voiceEmbedding` array of 192 floats
   - `embeddingModel`: "speechbrain-ecapa-voxceleb"
   - `sampleDuration`: ~30

---

## Test 5: Meeting with Speaker Recognition

1. Navigate to **New Note** → Click **Meeting** button
2. Enter 2 speakers (one should be the enrolled user)
3. Start the meeting (both Web Speech API and AI Meeting should start)
4. Speak for at least 10 seconds
5. Have another person speak (or simulate)
6. **Expected**:
   - Enrolled speaker's name appears with ✓ confidence mark
   - Unknown speakers appear as "Speaker 1" / "Speaker 2"
   - Transcript shows in the Quill editor in real-time

---

## Test 6: Speaker Labels Persist

1. Continue the meeting for 2+ minutes
2. Switch speakers multiple times
3. **Expected**:
   - Same speaker always gets the same label
   - No label flickering or inconsistency
   - Timestamps inserted at ~90 second intervals

---

## Test 7: Meeting Without AI Server

1. Stop the AI server (Ctrl+C on `python -m ai_server.app`)
2. Start a new meeting in AirScribe
3. **Expected**:
   - Meeting still works via Web Speech API fallback
   - No errors in browser console
   - Speakers labelled as "Speaker 1", "Speaker 2" etc.

---

## Test 8: Delete Voice Profile

1. Navigate to **Voice Profile** page
2. Click the trash icon on an enrolled profile
3. Confirm deletion
4. **Expected**:
   - Profile disappears from the list
   - MongoDB document is removed
   - Next meeting uses default "Speaker X" labels

---

## Test 9: Re-enrollment

1. After deleting, click **Start Recording** again
2. Record new 30-second sample
3. **Expected**:
   - New profile created successfully
   - Old profile data completely replaced

---

## Test 10: GPU Memory Check (RTX 3050)

1. Open `nvidia-smi` in a terminal
2. Start a meeting with AI transcription + recognition active
3. **Expected**:
   - Total VRAM usage < 3.5 GB
   - No out-of-memory errors
   - Models load sequentially (not all at once)

---

## Test 11: Summary Generation

1. Complete a meeting with at least 1 minute of dialogue
2. Click **Summarize** button
3. **Expected**:
   - Summary includes key points
   - Action items extracted (if any "I will...", "We need to..." phrases were said)
   - Speaker statistics shown

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| "HF_TOKEN not set" warning | Add HF_TOKEN to .env file |
| "pyannote.audio not installed" | `pip install pyannote.audio` |
| "SpeechBrain not installed" | `pip install speechbrain` |
| CUDA out of memory | Change `WHISPER_MODEL=tiny` in .env |
| "Could not connect to AI server" | Ensure port 5001 is running |
| Mic permission denied | Check browser settings → Allow microphone |
| ffmpeg not found | Install ffmpeg and add to PATH |
