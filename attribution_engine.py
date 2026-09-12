"""
Task 5 Root Exporter: Attribution Engine (Probabilistic Decomposition)
======================================================================
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
    from voice_intelligence.overlap_suite.attribution_engine import (
        WordAttribution,
        AttributedOverlapSegment,
        AttributionReport,
        AttributionEngine,
        get_attribution_engine,
    )
except ImportError:
    from overlap_suite.attribution_engine import (
        WordAttribution,
        AttributedOverlapSegment,
        AttributionReport,
        AttributionEngine,
        get_attribution_engine,
    )

__all__ = [
    "WordAttribution",
    "AttributedOverlapSegment",
    "AttributionReport",
    "AttributionEngine",
    "get_attribution_engine",
]
