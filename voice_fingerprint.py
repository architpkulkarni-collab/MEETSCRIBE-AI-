"""
Task 4 Root Exporter: Voice Fingerprinting (SpeechBrain ECAPA-TDNN)
===================================================================
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
    from voice_intelligence.overlap_suite.voice_fingerprint import (
        VoiceFingerprint,
        SpeakerProfileRegistry,
        VoiceFingerprintEngine,
        get_fingerprint_engine,
    )
except ImportError:
    from overlap_suite.voice_fingerprint import (
        VoiceFingerprint,
        SpeakerProfileRegistry,
        VoiceFingerprintEngine,
        get_fingerprint_engine,
    )

__all__ = [
    "VoiceFingerprint",
    "SpeakerProfileRegistry",
    "VoiceFingerprintEngine",
    "get_fingerprint_engine",
]
