# Voice Intelligence & Overlapping-Speech Attribution Engine
**Owner:** Vedant (Member A)  
**Feature:** Overlapping-Speech Attribution Suite & Voice Intelligence

---

## 🎙️ Overview
A high-accuracy acoustic and temporal intelligence suite built for multi-speaker meetings. Provides word-level transcription timestamps, pyannote speaker diarization, sweep-line overlapping speech clash detection, 192-dimensional ECAPA-TDNN voice fingerprinting with deterministic spectral fallback, probabilistic soft attribution decomposition, and persistent voice profile registration.

---

## 📜 Task 1: Word-Level Transcription (faster-whisper) (Complete)
- **Engine:** faster-whisper with CTranslate2 backend (float16 / int8).
- **Capabilities:** Word-level timestamps, voice activity probability detection, automatic language detection.
- **Run Verification:** `python voice_intelligence/test_task1_whisper.py`

---

## 👥 Task 2: pyannote Speaker Diarization Engine (Complete)
- **Engine:** pyannote.audio speaker diarization with intelligent speech interval fallback.
- **Capabilities:** Preserves multi-track speaker turns to allow collision detection rather than flattening simultaneous speakers. Matches segments against registered profiles when available.
- **Run Verification:** `python voice_intelligence/test_task2_diarization.py`

---

## ⚡ Task 3: Overlapping Clash Detection (Complete)
- **Engine:** Sweep-line geometric collision detection algorithm.
- **Capabilities:** Pinpoints exact millisecond intervals where two or more speakers speak simultaneously. Collates clashing word tokens.
- **Run Verification:** `python voice_intelligence/test_task3_overlap.py`

---

## 🧬 Task 4: Voice Fingerprinting (SpeechBrain ECAPA-TDNN) (Complete)
- **Engine:** 192-dimensional speaker embeddings using `speechbrain/spkrec-ecapa-voxceleb` with deterministic acoustic spectral fallback.
- **Enrollment:** Automatically enrolls meeting participants from pristine clean segments identified in Task 3.
- **Mixed-Signal Projection:** Extracts acoustic vectors directly from collision audio slices.
- **Run Verification:** `python voice_intelligence/test_task4_fingerprint.py`

---

## 🎯 Task 5: Probabilistic Attribution Engine (Complete)
- **Engine:** Temperature-scaled softmax over cosine similarity projections:
  $$P(S_k \mid \text{mixed}) = \frac{\exp(\text{sim}(S_k) / \tau)}{\sum_j \exp(\text{sim}(S_j) / \tau)}$$
- **Output:** Soft attribution breakdown (e.g. `SPEAKER_00: 72%`, `SPEAKER_01: 28%`), dominant vs. secondary speaker classification, and word-level attribution confidence.
- **Run Verification:** `python voice_intelligence/test_task5_attribution.py`

---

## 🏆 Task 6: Patent Proof Benchmark Evaluation (Complete)
- **Engine:** Automated scientific validation harness comparing against traditional single-track diarization.
- **Metrics Evaluated:**
  - Clash Detection Precision, Recall, and F1-score ($>90\%$).
  - Word Attribution Error Reduction ($\text{WAER} > 80\%$).
  - Real-Time Factor ($\text{RTF} < 0.1\times$).
- **Report Output:** Saved automatically to `benchmarks/patent_proof_report.json` and `benchmarks/patent_proof_report.md`.
- **Run Verification:** `python voice_intelligence/test_task6_benchmarks.py`

---

## 🎙️ Voice Registration & Saved Profile Recognition (Feature Update)
- **Voice Enrollment:** Register individual voices (e.g. Person 1, Person 2, etc.) from audio clips or mic recordings.
- **Persistent Saved Data:** Automatically saves registered voice profiles to persistent JSON storage (`saved_profiles.json`) and syncs across restarts.
- **Zero-Overlap Voice Differentiation:** Uses 192-dimensional embeddings combining pitch autocorrelation (F0 & harmonics), 32 Mel filterbanks, and formant ratios to cleanly differentiate Person 1 from Person 2 (intra-speaker similarity > 0.85, inter-speaker distance < 0.40).
- **Meeting Diarization Integration:** Attributes each speech segment in meeting audio to the recognized registered person from saved data.
- **REST APIs:**
  - `POST /ai/voice/register`: Register a person's voice by name and persist to disk.
  - `GET /ai/voice/saved-profiles`: Retrieve all enrolled profiles from saved data.
  - `POST /ai/voice/recognize`: Identify the speaker in any audio sample.
  - `DELETE /ai/voice/saved-profile/<name>`: Remove a registered profile.
- **Run Verification:**
  ```bash
  python voice_intelligence/test_voice_registration_and_recognition.py
  ```

---

## 🚀 Run All Remaining Tasks (Master Runner)
To run and verify Tasks 3, 4, 5, 6, and the Voice Registration & Recognition suite in a single sequence:

```bash
python voice_intelligence/test_all_remaining_tasks.py
```
