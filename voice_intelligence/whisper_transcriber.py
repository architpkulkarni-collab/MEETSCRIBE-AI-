"""
Task 1: faster-whisper Speech-to-Text with Word-Level Timestamps
================================================================
Owner: Vedant (Member A)
Feature: Overlapping-Speech Attribution

Provides high-speed transcription with word-by-word timestamps.
Word-level timestamps allow us to map exactly which words collide
when two speakers talk simultaneously.
"""

import os
import sys
import time
import logging
import threading
import importlib
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any

# Ensure project root and package directory are in sys.path
_current_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_current_dir)
for p in [_project_root, _current_dir]:
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from voice_intelligence.config import (  # type: ignore
        WHISPER_MODEL_SIZE,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
        WHISPER_BEAM_SIZE,
        WHISPER_VAD_FILTER,
        WHISPER_WORD_TIMESTAMPS,
    )
except ImportError:
    from config import (  # type: ignore
        WHISPER_MODEL_SIZE,
        WHISPER_DEVICE,
        WHISPER_COMPUTE_TYPE,
        WHISPER_BEAM_SIZE,
        WHISPER_VAD_FILTER,
        WHISPER_WORD_TIMESTAMPS,
    )

logger = logging.getLogger("voice_intelligence.whisper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ── Dynamic Module Resolvers (Prevents VS Code / Pylance reportMissingImports) ──
def _load_faster_whisper_class():
    """Dynamically import WhisperModel from faster_whisper if installed."""
    try:
        mod = importlib.import_module("faster_whisper")
        return getattr(mod, "WhisperModel", None)
    except ImportError:
        return None


def _load_openai_whisper_module():
    """Dynamically import openai whisper if installed."""
    try:
        return importlib.import_module("whisper")
    except ImportError:
        return None


@dataclass
class WordToken:
    """Represents an individual word with fine-grained millisecond timestamps."""
    word: str
    start: float
    end: float
    probability: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TranscriptSegment:
    """Represents a speech segment containing multiple words."""
    id: int
    start: float
    end: float
    text: str
    words: List[WordToken] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
            "words": [w.to_dict() for w in self.words]
        }

    def get_words_overlapping_range(self, t_start: float, t_end: float) -> List[WordToken]:
        """
        Return all words inside this segment that overlap with [t_start, t_end].
        Crucial for extracting words spoken during an overlapping clash segment.
        """
        overlapping_words = []
        for w in self.words:
            if max(w.start, t_start) < min(w.end, t_end):
                overlapping_words.append(w)
        return overlapping_words


@dataclass
class TranscriptionResult:
    """Full result of a transcription run."""
    text: str
    language: str
    language_probability: float
    duration: float
    segments: List[TranscriptSegment] = field(default_factory=list)
    processing_time: float = 0.0
    engine_used: str = "faster-whisper"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "language_probability": round(self.language_probability, 3),
            "duration": round(self.duration, 2),
            "processing_time": round(self.processing_time, 2),
            "engine_used": self.engine_used,
            "segments": [s.to_dict() for s in self.segments]
        }

    def get_words_in_timerange(self, t_start: float, t_end: float) -> List[WordToken]:
        """Collect all words across all segments that occur within [t_start, t_end]."""
        matched = []
        for seg in self.segments:
            if seg.end < t_start or seg.start > t_end:
                continue
            matched.extend(seg.get_words_overlapping_range(t_start, t_end))
        return matched


class FasterWhisperTranscriber:
    """
    Singleton / thread-safe manager for transcription.
    Automatically uses faster-whisper if installed, or falls back to
    openai-whisper without crashing or showing IDE import errors.
    """
    _instance = None
    _lock = threading.Lock()

    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model: Any = None
        self.engine: str = "faster-whisper"
        self._load_lock = threading.Lock()

    def load_model(self) -> Any:
        """Lazy load the speech-to-text model."""
        if self.model is not None:
            return self.model

        with self._load_lock:
            if self.model is not None:
                return self.model

            # 1. Attempt faster-whisper
            whisper_cls = _load_faster_whisper_class()
            if whisper_cls is not None:
                try:
                    logger.info(
                        f"Loading faster-whisper '{self.model_size}' "
                        f"(device={self.device}, compute_type={self.compute_type})..."
                    )
                    t0 = time.time()
                    self.model = whisper_cls(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type
                    )
                    self.engine = "faster-whisper"
                    elapsed = time.time() - t0
                    logger.info(f"faster-whisper loaded successfully in {elapsed:.2f}s")
                    return self.model
                except Exception as e:
                    if self.device == "cuda":
                        logger.warning(f"CUDA failed ({e}). Retrying faster-whisper on CPU...")
                        try:
                            self.device = "cpu"
                            self.compute_type = "int8"
                            self.model = whisper_cls(self.model_size, device="cpu", compute_type="int8")
                            self.engine = "faster-whisper"
                            logger.info("faster-whisper loaded on CPU fallback.")
                            return self.model
                        except Exception as cpu_err:
                            logger.warning(f"CPU load error: {cpu_err}")

            # 2. Attempt OpenAI Whisper fallback
            openai_mod = _load_openai_whisper_module()
            if openai_mod is not None:
                try:
                    logger.info(
                        f"faster-whisper not installed; using openai-whisper fallback "
                        f"('{self.model_size}'). Run 'pip install faster-whisper' to upgrade."
                    )
                    t0 = time.time()
                    self.model = openai_mod.load_model(self.model_size)
                    self.engine = "openai-whisper"
                    elapsed = time.time() - t0
                    logger.info(f"openai-whisper fallback loaded in {elapsed:.2f}s")
                    return self.model
                except Exception as e:
                    logger.error(f"openai-whisper load failed: {e}")

            # Neither engine is installed
            raise RuntimeError(
                "No Whisper engine found in your Python environment.\n"
                "Please run:\n"
                "    pip install faster-whisper\n"
                "or:\n"
                "    pip install openai-whisper"
            )

    def is_loaded(self) -> bool:
        return self.model is not None

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        beam_size: int = WHISPER_BEAM_SIZE,
        word_timestamps: bool = WHISPER_WORD_TIMESTAMPS,
        vad_filter: bool = WHISPER_VAD_FILTER,
        initial_prompt: Optional[str] = None,
    ) -> TranscriptionResult:
        """
        Transcribe audio and extract word-level timestamps.
        """
        model = self.load_model()
        logger.info(f"Transcribing '{audio_path}' using engine: {self.engine}...")
        t0 = time.time()

        if self.engine == "faster-whisper":
            return self._transcribe_faster_whisper(
                model, audio_path, language, beam_size, word_timestamps, vad_filter, initial_prompt, t0
            )
        else:
            return self._transcribe_openai_whisper(
                model, audio_path, language, word_timestamps, initial_prompt, t0
            )

    def _transcribe_faster_whisper(
        self, model, audio_path, language, beam_size, word_timestamps, vad_filter, initial_prompt, t0
    ) -> TranscriptionResult:
        segments_generator, info = model.transcribe(
            audio_path,
            beam_size=beam_size,
            language=language,
            word_timestamps=word_timestamps,
            vad_filter=vad_filter,
            initial_prompt=initial_prompt,
        )

        all_segments: List[TranscriptSegment] = []
        full_text_parts: List[str] = []

        for seg in segments_generator:
            words: List[WordToken] = []
            if seg.words:
                for w in seg.words:
                    words.append(WordToken(
                        word=w.word.strip(),
                        start=round(w.start, 3),
                        end=round(w.end, 3),
                        probability=round(w.probability, 3)
                    ))

            text_cleaned = seg.text.strip()
            if text_cleaned:
                full_text_parts.append(text_cleaned)

            all_segments.append(TranscriptSegment(
                id=seg.id,
                start=round(seg.start, 3),
                end=round(seg.end, 3),
                text=text_cleaned,
                words=words
            ))

        elapsed = time.time() - t0
        full_text = " ".join(full_text_parts)

        logger.info(
            f"Transcription complete: {len(all_segments)} segments in {elapsed:.2f}s "
            f"(RTFx: {info.duration / max(elapsed, 0.001):.1f}x)"
        )

        return TranscriptionResult(
            text=full_text,
            language=info.language,
            language_probability=info.language_probability,
            duration=info.duration,
            segments=all_segments,
            processing_time=elapsed,
            engine_used="faster-whisper"
        )

    def _transcribe_openai_whisper(
        self, model, audio_path, language, word_timestamps, initial_prompt, t0
    ) -> TranscriptionResult:
        kwargs = {
            "language": language,
            "word_timestamps": word_timestamps,
            "verbose": False
        }
        if initial_prompt:
            kwargs["initial_prompt"] = initial_prompt

        result = model.transcribe(audio_path, **kwargs)

        all_segments: List[TranscriptSegment] = []
        full_text_parts: List[str] = []

        for idx, seg in enumerate(result.get("segments", [])):
            words: List[WordToken] = []
            for w in seg.get("words", []):
                words.append(WordToken(
                    word=w.get("word", "").strip(),
                    start=round(w.get("start", 0.0), 3),
                    end=round(w.get("end", 0.0), 3),
                    probability=round(w.get("probability", 1.0), 3)
                ))

            text_cleaned = seg.get("text", "").strip()
            if text_cleaned:
                full_text_parts.append(text_cleaned)

            all_segments.append(TranscriptSegment(
                id=idx,
                start=round(seg.get("start", 0.0), 3),
                end=round(seg.get("end", 0.0), 3),
                text=text_cleaned,
                words=words
            ))

        elapsed = time.time() - t0
        full_text = " ".join(full_text_parts)
        duration = all_segments[-1].end if all_segments else 0.0

        return TranscriptionResult(
            text=full_text,
            language=result.get("language", language or "en"),
            language_probability=1.0,
            duration=duration,
            segments=all_segments,
            processing_time=elapsed,
            engine_used="openai-whisper (fallback)"
        )


# Global singleton instance
_default_transcriber: Optional[FasterWhisperTranscriber] = None
_default_lock = threading.Lock()

def get_transcriber(model_size: str = WHISPER_MODEL_SIZE) -> FasterWhisperTranscriber:
    """Get or create the singleton FasterWhisperTranscriber."""
    global _default_transcriber
    if _default_transcriber is None:
        with _default_lock:
            if _default_transcriber is None:
                _default_transcriber = FasterWhisperTranscriber(model_size=model_size)
    return _default_transcriber
