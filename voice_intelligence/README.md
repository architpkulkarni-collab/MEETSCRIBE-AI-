# Voice & Speaker Intelligence (Feature #1)
## Overlapping-Speech Attribution Engine

**Owner:** Vedant (Member A)  
**Patent Target:** Multi-Modal Temporal Clash Detection & Probabilistic Voice Fingerprint Decomposition for Simultaneous Speech

---

## 📂 Module Architecture

All code for Feature #1 is self-contained in `voice_intelligence/` to guarantee zero collisions with teammates:

```
voice_intelligence/
├── __init__.py                 # Module exports (Tasks 1 & 2)
├── config.py                   # Hardware, model, and threshold configurations
├── requirements.txt            # Dedicated dependencies for this feature
├── whisper_transcriber.py      # [TASK 1] faster-whisper with word-level timestamps
├── test_task1_whisper.py       # [TASK 1] Test & verification runner
├── diarization_engine.py       # [TASK 2] pyannote clean speaker diarization
├── test_task2_diarization.py   # [TASK 2] Test & verification runner
├── overlap_detector.py         # [TASK 3] Clash/collision interval detector
├── voice_fingerprint.py        # [TASK 4] SpeechBrain ECAPA-TDNN mixed-signal embeddings
├── attribution_engine.py       # [TASK 5] Soft confidence scoring (e.g. 70% Spk1, 30% Spk2)
├── benchmark_eval.py           # [TASK 6] Empirical accuracy logs & patent proof harness
└── README.md                   # Full documentation & roadmap
```

---

## ⚡ Task 1: Word-Level Transcription (Complete)
- **Engine:** CTranslate2 `faster-whisper`
- **Output:** Word tokens with microsecond start and end timestamps (`w.start`, `w.end`, `w.probability`)
- **Query Overlap Words:** `result.get_words_in_timerange(t_start, t_end)`
- **Run Verification:** `python voice_intelligence/test_task1_whisper.py`

---

## 👥 Task 2: Speaker Diarization ("Who Spoke When") (Complete)
- **Engine:** `pyannote/speaker-diarization-3.1` with CUDA/CPU placement and clean fallback.
- **Output:** Structured `SpeakerTurn` timeline: `[start, end, speaker_id, duration]`.
- **Patent Foundation:** Unlike traditional systems that discard simultaneous tracks, this engine preserves raw multi-track intervals so Task 3 can detect collisions.
- **Clean Verification:** Checks whether a timeline is 100% clean (`is_clean == True`) or flags overlapping regions.
- **Run Verification:** `python voice_intelligence/test_task2_diarization.py`
