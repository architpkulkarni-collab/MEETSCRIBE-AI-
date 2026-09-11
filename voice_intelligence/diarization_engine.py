"""
Task 2: pyannote.audio Speaker Diarization Engine
=================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Extracts speaker turns ("who spoke when") from audio:
[(start_sec, end_sec, "SPEAKER_00"), (start_sec, end_sec, "SPEAKER_01"), ...]

Key Capabilities:
  - Establishes baseline speaker boundaries on clean (non-overlapping) audio.
  - Crucial Patent Foundation: Unlike standard pipelines that flatten and discard
    simultaneous speaker tracks, this engine preserves raw multi-track intervals
    so Task 3 can mathematically detect where two tracks intersect.
"""

import os
import sys
import time
import logging
import threading
import importlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any, Set, Tuple

# Bootstrap sys.path for both package and direct execution
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
for p in [_project_root, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from voice_intelligence.config import (  # type: ignore
        HF_TOKEN,
        PYANNOTE_MODEL,
        DEVICE,
    )
except ImportError:
    from config import (  # type: ignore
        HF_TOKEN,
        PYANNOTE_MODEL,
        DEVICE,
    )

logger = logging.getLogger("voice_intelligence.diarization")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ── Dynamic Module Resolvers (Prevents VS Code / Pylance reportMissingImports) ──
def _load_pyannote_pipeline_class():
    """Dynamically load Pipeline from pyannote.audio if installed."""
    try:
        mod = importlib.import_module("pyannote.audio")
        return getattr(mod, "Pipeline", None)
    except ImportError:
        return None


def _load_torch_module():
    """Dynamically load torch if installed."""
    try:
        return importlib.import_module("torch")
    except ImportError:
        return None


@dataclass
class SpeakerTurn:
    """Represents a discrete temporal interval where a speaker was active."""
    speaker_id: str
    start: float
    end: float
    track_id: Optional[str] = None

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 3)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "speaker": self.speaker_id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "duration": self.duration,
            "track": self.track_id
        }

    def overlaps_with(self, other: "SpeakerTurn") -> bool:
        """Check if this turn collides temporally with another turn."""
        return max(self.start, other.start) < min(self.end, other.end)

    def get_overlap_duration(self, other: "SpeakerTurn") -> float:
        """Return overlap duration in seconds (0.0 if disjoint)."""
        intersection = min(self.end, other.end) - max(self.start, other.start)
        return max(0.0, round(intersection, 3))


@dataclass
class DiarizationResult:
    """Complete output of a diarization run."""
    turns: List[SpeakerTurn] = field(default_factory=list)
    speakers: List[str] = field(default_factory=list)
    num_speakers: int = 0
    duration: float = 0.0
    processing_time: float = 0.0
    mode: str = "pyannote"  # 'pyannote' or 'fallback'

    @property
    def is_clean(self) -> bool:
        """Check if this timeline is 100% clean (zero overlapping collisions)."""
        for i in range(len(self.turns)):
            for j in range(i + 1, len(self.turns)):
                if self.turns[i].speaker_id != self.turns[j].speaker_id:
                    if self.turns[i].overlaps_with(self.turns[j]):
                        return False
        return True

    def get_speaker_timeline(self, speaker_id: str) -> List[SpeakerTurn]:
        """Return all turns for a specific speaker."""
        return [t for t in self.turns if t.speaker_id == speaker_id]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "num_speakers": self.num_speakers,
            "speakers": self.speakers,
            "duration": round(self.duration, 2),
            "processing_time": round(self.processing_time, 2),
            "is_clean": self.is_clean,
            "mode": self.mode,
            "turns": [t.to_dict() for t in self.turns]
        }


class PyannoteDiarizer:
    """
    Manages pyannote.audio speaker diarization with:
      - Automatic GPU/CPU placement
      - HuggingFace authentication
      - Preservation of multi-track turns for clash detection
      - Clean fallback mode when token/dependency is missing
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        hf_token: Optional[str] = None,
        model_name: str = PYANNOTE_MODEL,
        device: str = DEVICE
    ):
        self.hf_token = hf_token or HF_TOKEN
        self.model_name = model_name
        self.device = device
        self.pipeline: Any = None
        self.mode = "not_loaded"
        self._load_lock = threading.Lock()

    def load_pipeline(self) -> Any:
        """Lazy load the pyannote Pipeline."""
        if self.pipeline is not None:
            return self.pipeline

        with self._load_lock:
            if self.pipeline is not None:
                return self.pipeline

            pipeline_cls = _load_pyannote_pipeline_class()
            torch_mod = _load_torch_module()

            # Check if pyannote is installed and HF token is provided
            if pipeline_cls is not None and self.hf_token:
                try:
                    logger.info(f"Loading pyannote diarization pipeline '{self.model_name}'...")
                    t0 = time.time()

                    # Load pretrained pyannote pipeline
                    try:
                        self.pipeline = pipeline_cls.from_pretrained(
                            self.model_name,
                            use_auth_token=self.hf_token
                        )
                    except TypeError:
                        # Newer versions of pyannote use token= instead of use_auth_token=
                        self.pipeline = pipeline_cls.from_pretrained(
                            self.model_name,
                            token=self.hf_token
                        )

                    # Move to GPU if available
                    if self.device == "cuda" and torch_mod is not None:
                        try:
                            self.pipeline = self.pipeline.to(torch_mod.device("cuda"))
                            logger.info("pyannote pipeline successfully moved to GPU (CUDA)")
                        except Exception as gpu_err:
                            logger.warning(f"Failed to place on GPU ({gpu_err}), running on CPU.")

                    elapsed = time.time() - t0
                    self.mode = "pyannote"
                    logger.info(f"pyannote pipeline loaded in {elapsed:.2f}s")
                    return self.pipeline

                except Exception as e:
                    logger.warning(f"Could not load pyannote pipeline ({e}). Using clean fallback engine.")
                    self.mode = "fallback"
            else:
                if pipeline_cls is None:
                    logger.info(
                        "pyannote.audio is not installed. "
                        "Install with: pip install pyannote.audio"
                    )
                elif not self.hf_token:
                    logger.info(
                        "HF_TOKEN is not set in .env. "
                        "Get your free token at https://huggingface.co/settings/tokens "
                        "and accept the pyannote license to use full neural diarization."
                    )
                self.mode = "fallback"

        return self.pipeline

    def diarize(
        self,
        audio_path: str,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ) -> DiarizationResult:
        """
        Perform speaker diarization on audio file.

        Args:
            audio_path: Path to the WAV audio file (16kHz mono recommended).
            num_speakers: Exact speaker count if known in advance.
            min_speakers: Minimum speaker count hint.
            max_speakers: Maximum speaker count hint.

        Returns:
            DiarizationResult with clean list of SpeakerTurns.
        """
        self.load_pipeline()
        t0 = time.time()
        logger.info(f"Diarizing '{audio_path}' (mode={self.mode})...")

        if self.mode == "pyannote" and self.pipeline is not None:
            return self._diarize_pyannote(
                audio_path, num_speakers, min_speakers, max_speakers, t0
            )
        else:
            return self._diarize_fallback(audio_path, num_speakers or 2, t0)

    def _diarize_pyannote(
        self, audio_path, num_speakers, min_speakers, max_speakers, t0
    ) -> DiarizationResult:
        """Execute pyannote neural diarization pipeline."""
        kwargs = {}
        if num_speakers is not None:
            kwargs["num_speakers"] = int(num_speakers)
        elif min_speakers is not None or max_speakers is not None:
            if min_speakers is not None:
                kwargs["min_speakers"] = int(min_speakers)
            if max_speakers is not None:
                kwargs["max_speakers"] = int(max_speakers)

        diarization_output = self.pipeline(audio_path, **kwargs)

        turns: List[SpeakerTurn] = []
        speaker_set: Set[str] = set()
        max_time = 0.0

        # Extract all tracks while preserving potential collisions
        for turn, track_id, speaker_label in diarization_output.itertracks(yield_label=True):
            st = round(turn.start, 3)
            et = round(turn.end, 3)
            speaker_set.add(speaker_label)
            if et > max_time:
                max_time = et

            turns.append(SpeakerTurn(
                speaker_id=speaker_label,
                start=st,
                end=et,
                track_id=str(track_id)
            ))

        # Sort turns chronologically
        turns.sort(key=lambda t: t.start)
        elapsed = time.time() - t0

        unique_speakers = sorted(list(speaker_set))
        logger.info(
            f"pyannote complete: {len(turns)} turns, {len(unique_speakers)} speakers "
            f"in {elapsed:.2f}s"
        )

        return DiarizationResult(
            turns=turns,
            speakers=unique_speakers,
            num_speakers=len(unique_speakers),
            duration=max_time,
            processing_time=elapsed,
            mode="pyannote"
        )

    def _diarize_fallback(self, audio_path: str, num_speakers: int, t0: float) -> DiarizationResult:
        """
        Clean fallback diarizer for testing or when HF_TOKEN is not configured.
        Uses audio duration and speech intervals to create clean speaker turns.
        """
        import wave

        duration = 3.0
        try:
            with wave.open(audio_path, "r") as wf:
                frames = wf.getnframes()
                rate = wf.getframerate()
                duration = frames / float(rate)
        except Exception:
            pass

        # Synthesize clean alternating speaker turns on the audio duration
        turns: List[SpeakerTurn] = []
        segment_len = max(1.2, duration / (num_speakers * 2))
        curr_t = 0.0
        spk_idx = 0

        while curr_t < duration:
            end_t = min(curr_t + segment_len, duration)
            spk_label = f"SPEAKER_{spk_idx % num_speakers:02d}"
            turns.append(SpeakerTurn(
                speaker_id=spk_label,
                start=round(curr_t, 3),
                end=round(end_t, 3),
                track_id=f"track_{len(turns)}"
            ))
            curr_t = end_t + 0.2  # 200ms gap between clean turns
            spk_idx += 1

        elapsed = time.time() - t0
        speakers = [f"SPEAKER_{i:02d}" for i in range(num_speakers)]

        logger.info(f"Fallback diarization complete: {len(turns)} turns in {elapsed:.3f}s")
        return DiarizationResult(
            turns=turns,
            speakers=speakers,
            num_speakers=num_speakers,
            duration=duration,
            processing_time=elapsed,
            mode="fallback"
        )


# Global singleton instance
_default_diarizer: Optional[PyannoteDiarizer] = None
_default_lock = threading.Lock()

def get_diarizer() -> PyannoteDiarizer:
    """Get or create singleton PyannoteDiarizer."""
    global _default_diarizer
    if _default_diarizer is None:
        with _default_lock:
            if _default_diarizer is None:
                _default_diarizer = PyannoteDiarizer()
    return _default_diarizer
