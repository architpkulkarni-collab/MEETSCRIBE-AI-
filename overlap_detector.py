"""
Task 3 Root Exporter: Overlap & Temporal Clash Detector
=======================================================
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
    from voice_intelligence.overlap_suite.overlap_detector import (
        OverlapSegment,
        CollisionSummary,
        OverlapDetector,
        get_overlap_detector,
    )
except ImportError:
    from overlap_suite.overlap_detector import (
        OverlapSegment,
        CollisionSummary,
        OverlapDetector,
        get_overlap_detector,
    )

__all__ = [
    "OverlapSegment",
    "CollisionSummary",
    "OverlapDetector",
    "get_overlap_detector",
]
