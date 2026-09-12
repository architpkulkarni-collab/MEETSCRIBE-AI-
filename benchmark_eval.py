"""
Task 6 Root Exporter: Benchmark & Patent Proof Evaluation Harness
=================================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution
"""

import sys
import os

_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
for p in [_project_root, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from voice_intelligence.overlap_suite.benchmark_eval import (
        BenchmarkMetricSet,
        PatentBenchmarkHarness,
        get_benchmark_harness,
    )
except ImportError:
    from overlap_suite.benchmark_eval import (
        BenchmarkMetricSet,
        PatentBenchmarkHarness,
        get_benchmark_harness,
    )

__all__ = [
    "BenchmarkMetricSet",
    "PatentBenchmarkHarness",
    "get_benchmark_harness",
]
