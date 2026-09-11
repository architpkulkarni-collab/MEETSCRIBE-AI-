"""
Pytest configuration for AirScribe test suite.
Ensures the project root is in sys.path for imports.
"""

import os
import sys

# Add project root to path so tests can import encryption, note_embeddings, etc.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
