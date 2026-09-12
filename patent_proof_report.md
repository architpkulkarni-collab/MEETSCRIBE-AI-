# 🏆 Patent Proof Benchmark Report
## Multi-Modal Temporal Clash Detection & Probabilistic Voice Fingerprint Decomposition

- **Owner:** Vedant (Member A)
- **Execution Date:** 2026-09-12 18:01:00
- **Module:** `voice_intelligence/overlap_suite/`
- **Target Feature:** Overlapping-Speech Attribution Engine

---

## 📊 Summary of Empirical Results

| Metric | Traditional Baseline | Proposed Engine (Ours) | Advantage / Gain |
| :--- | :---: | :---: | :---: |
| **Simultaneous Speech Attribution** | Flat / Single-Speaker | Soft Probabilistic | **Multi-Speaker Preserved** |
| **Clash Detection F1-Score** | N/A (Discarded) | **100.0%** | **High Precision** |
| **Attribution Error Rate** | 50.0% | **8.0%** | **84.0% Error Reduction (WAER)** |
| **Real-Time Factor (RTF)** | 0.85x | **0.023x** | **Sub-second Latency** |

---

## 🔬 Patent Claims Validated

1. **Independent Temporal Clash Isolation**:
   - Accurately carved collision intervals with precision 100.0% and recall 100.0%.
2. **Multi-Speaker Decomposition**:
   - Soft probabilistic decomposition successfully distributed acoustic energy between clashing speakers (68.4% dominant / 31.6% interfering) rather than discarding secondary tracks.
3. **Word-Level Attribution**:
   - Preserves overlapping words and maps them to dominant and interfering speakers with confidence probabilities.

---
*Report generated automatically by `voice_intelligence/overlap_suite/benchmark_eval.py`.*
