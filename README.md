# Voice & Speaker Intelligence (Feature #1)
## Overlapping-Speech Attribution Engine

**Owner:** Vedant (Member A)  
**Patent Target:** Multi-Modal Temporal Clash Detection & Probabilistic Voice Fingerprint Decomposition for Simultaneous Speech

---

## 📂 Module Architecture

All code for Feature #1 is self-contained in `voice_intelligence/` to guarantee zero collisions with teammates:

```
voice_intelligence/
├── __init__.py                     # Module exports (Tasks 1 through 6)
├── config.py                       # Hardware, model, and threshold configurations
├── requirements.txt                # Dedicated dependencies for this feature
├── whisper_transcriber.py          # [TASK 1] faster-whisper with word-level timestamps
├── test_task1_whisper.py           # [TASK 1] Test & verification runner
├── diarization_engine.py           # [TASK 2] pyannote clean speaker diarization
├── test_task2_diarization.py       # [TASK 2] Test & verification runner
├── overlap_detector.py             # [TASK 3] Sweep-line clash/collision detector
├── test_task3_overlap.py           # [TASK 3] Test & verification runner
├── voice_fingerprint.py            # [TASK 4] SpeechBrain ECAPA-TDNN mixed-signal embeddings
├── test_task4_fingerprint.py       # [TASK 4] Test & verification runner
├── attribution_engine.py           # [TASK 5] Soft confidence scoring (e.g. 70% Spk1, 30% Spk2)
├── test_task5_attribution.py       # [TASK 5] Test & verification runner
├── benchmark_eval.py               # [TASK 6] Empirical accuracy logs & patent proof harness
├── test_task6_benchmarks.py        # [TASK 6] Test & verification runner
├── test_all_remaining_tasks.py     # Master runner for Tasks 3-6
├── benchmarks/                     # Benchmark logs & patent reports
│   ├── patent_proof_report.json    # Machine-readable evaluation metrics
│   └── patent_proof_report.md      # Formatted patent evidence report
└── overlap_suite/                  # [DEDICATED SUITE FOLDER] Modular implementation
    ├── __init__.py                 # Suite exports
    ├── audio_utils.py              # WAV sub-second slicing & collision synthesis
    ├── overlap_detector.py         # Task 3 core detector
    ├── voice_fingerprint.py        # Task 4 core fingerprint engine
    ├── attribution_engine.py       # Task 5 core probabilistic decomposition
    ├── benchmark_eval.py           # Task 6 core benchmark harness
    └── pipeline.py                 # Full End-to-End multi-modal pipeline
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

---

## 💥 Task 3: Overlap & Temporal Clash Detection (Complete)
- **Engine:** Interval sweep-line collision detector.
- **Output:** `CollisionSummary` with `OverlapSegment`s $[t_{\text{start}}, t_{\text{end}}]$, clashing speaker pairs, and colliding word tokens from Task 1.
- **Pristine Reference Isolation:** Extracts clean single-speaker segments (`clean_segments`) for pristine voice enrollment.
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

## 🚀 Run All Remaining Tasks (Master Runner)
To run and verify Tasks 3, 4, 5, and 6 in a single sequence:

```bash
python voice_intelligence/test_all_remaining_tasks.py
```
