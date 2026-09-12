"""
Test Script for Task 6: Empirical Benchmark & Patent Proof Harness
==================================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Run:
    python voice_intelligence/test_task6_benchmarks.py [optional_audio_path]

Verifies that:
  1. Automated test bench executes end-to-end evaluation.
  2. Computes temporal clash detection Precision, Recall, and F1-score.
  3. Validates Word Attribution Error Reduction (WAER) over traditional baseline.
  4. Generates JSON and Markdown patent reports in voice_intelligence/benchmarks/.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    from voice_intelligence.config import BENCHMARK_DIR  # type: ignore
    from voice_intelligence.benchmark_eval import PatentBenchmarkHarness, get_benchmark_harness  # type: ignore
except ImportError:
    from config import BENCHMARK_DIR  # type: ignore
    from benchmark_eval import PatentBenchmarkHarness, get_benchmark_harness  # type: ignore


def run_task6_verification(audio_path: str = None) -> bool:
    print("\n" + "=" * 68)
    print("🚀 TASK 6 VERIFICATION: Empirical Benchmark & Patent Proof Harness")
    print("   Owner: Vedant | Feature: Overlapping-Speech Attribution")
    print("=" * 68)

    print("\n[Step 1] Initializing Patent Benchmark Harness...")
    harness = get_benchmark_harness()
    print(f"✓ Harness initialized (Output Directory: {harness.benchmark_dir})")

    print("\n[Step 2] Executing Benchmark Evaluation against Multi-Speaker Ground Truth...")
    metrics = harness.run_benchmark(audio_path=audio_path)

    print(f"\n[Step 3] Benchmark Quantitative Performance Metrics:")
    print(f"  • Dataset:                  {metrics.dataset_name}")
    print(f"  • Audio Duration:           {metrics.audio_duration_sec:.2f}s")
    print(f"  • Ground-Truth Clashes:     {metrics.ground_truth_clashes}")
    print(f"  • Detected Clashes:         {metrics.detected_clashes}")
    print(f"  • Clash Detection F1-Score: {metrics.clash_detection_f1 * 100:.1f}%")
    print(f"  • Baseline Error Rate:      {metrics.baseline_attribution_error * 100:.1f}% (Traditional single-speaker)")
    print(f"  • Proposed Engine Error:    {metrics.proposed_attribution_error * 100:.1f}% (Soft probabilistic)")
    print(f"  • Attribution Improvement:  {metrics.word_attribution_error_reduction:.1f}% WAER Reduction")
    print(f"  • Processing Speed (RTF):   {metrics.real_time_factor:.4f}x (real-time factor)")

    # Step 4: Verify generated report artifacts
    print("\n[Step 4] Verifying Patent Proof Artifact Generation:")
    json_report = os.path.join(harness.benchmark_dir, "patent_proof_report.json")
    md_report = os.path.join(harness.benchmark_dir, "patent_proof_report.md")

    assert os.path.exists(json_report), f"Missing JSON report at {json_report}"
    assert os.path.exists(md_report), f"Missing Markdown report at {md_report}"

    print(f"  ✓ JSON Evidence Log:  {json_report} ({os.path.getsize(json_report)} bytes)")
    print(f"  ✓ Markdown Report:    {md_report} ({os.path.getsize(md_report)} bytes)")

    # Assertions
    assert metrics.clash_detection_f1 >= 0.70, "Clash detection F1 score below target!"
    assert metrics.word_attribution_error_reduction > 50.0, "WAER reduction below target!"

    print("\n" + "=" * 68)
    print("✅ TASK 6 COMPLETE & VERIFIED: Benchmark & Patent Proof Harness operational!")
    print("=" * 68 + "\n")
    return True


if __name__ == "__main__":
    path_arg = sys.argv[1] if len(sys.argv) > 1 else None
    run_task6_verification(path_arg)
