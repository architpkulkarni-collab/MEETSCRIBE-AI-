"""
AirScribe AI Server — Unified Processing Pipeline
=====================================================
Chains diarization → recognition → transcription into a single pipeline.

Purpose:
    This is the core module that replaces the original transcribe_and_diarize()
    function. Given an audio chunk and a meeting session, it produces
    speaker-labelled transcript segments like:
        [{"speaker": "Archit", "text": "Hello", "start": 0.0, "end": 2.1}, ...]

How it connects:
    Called by ai_meeting_server.py (via the integration bridge) when
    processing audio chunks during a live meeting.

How the pipeline works:
    1. Convert audio to WAV (16kHz mono)
    2. Run pyannote diarization → speaker segments (SPEAKER_00, SPEAKER_01, ...)
    3. Run Whisper transcription → text segments with timestamps
    4. Map Whisper segments to diarization speakers
    5. For each unique speaker, extract an embedding and match against enrolled profiles
    6. Replace SPEAKER_XX labels with real names (or "Unknown Speaker")
    7. Merge consecutive same-speaker segments

How to debug:
    Enable AI_DEBUG=True to see the full pipeline trace including
    diarization results, speaker matching scores, and timing.
"""

import os
import time
import logging
from typing import Optional

from ai_server.config import UPLOAD_FOLDER
from ai_server.whisper_service import transcribe_audio
from ai_server.diarization import (
    diarize_audio,
    simple_diarization_fallback,
    map_whisper_to_diarization,
    get_mode as get_diarization_mode
)
from ai_server.speaker_recognition import (
    extract_embedding,
    identify_speaker_from_embedding,
    extract_segment_audio,
    is_loaded as is_recognition_loaded
)
from ai_server.embeddings import get_all_embeddings
from ai_server.utils import cleanup_temp_files

logger = logging.getLogger("airscribe.ai.pipeline")


def process_audio_chunk(
    audio_path: str,
    session,
    enrolled_profiles: Optional[list[dict]] = None
) -> list[dict]:
    """
    Process an audio chunk through the full pipeline:
    diarize → recognize → transcribe → label.

    Args:
        audio_path: Path to the audio file (WAV, 16kHz mono).
        session: The MeetingSession object (from ai_meeting_server.py)
            with speaker tracking state.
        enrolled_profiles: List of enrolled voice profiles with embeddings.
            If None, profiles will be fetched from MongoDB.

    Returns:
        List of labelled segment dicts:
        [
            {
                "speaker": "Archit",        # Real name or "Unknown Speaker"
                "speakerIndex": 0,
                "text": "Hello everyone",
                "start": 0.0,
                "end": 2.1,
                "confidence": 0.85           # Recognition confidence (0-1)
            },
            ...
        ]
    """
    pipeline_start = time.time()
    results = []
    temp_segments = []  # Track temp files for cleanup

    try:
        # ── Step 1: Fetch enrolled profiles if not provided ──
        if enrolled_profiles is None:
            enrolled_profiles = get_all_embeddings()
            logger.debug(f"Loaded {len(enrolled_profiles)} enrolled profiles from DB")

        # ── Step 2: Diarize (identify speaker turns) ──
        diarization_mode = get_diarization_mode()
        diarization_segments = []

        if diarization_mode == "pyannote":
            diarization_segments = diarize_audio(
                audio_path,
                num_speakers=session.speaker_count if session.speaker_count <= 6 else None,
                max_speakers=min(session.speaker_count, 6)
            )

        # ── Step 3: Transcribe with Whisper ──
        whisper_segments = transcribe_audio(audio_path)
        if not whisper_segments:
            logger.info("No speech detected in audio chunk")
            return results

        # ── Step 4: Map Whisper segments to speakers ──
        if diarization_segments:
            # We have pyannote results — map Whisper text to speakers
            labelled_segments = map_whisper_to_diarization(
                whisper_segments, diarization_segments
            )
        else:
            # Fallback: use silence gaps to alternate speakers
            labelled_segments = simple_diarization_fallback(
                whisper_segments, session.speaker_count
            )

        if not labelled_segments:
            return results

        # ── Step 5: Speaker Recognition ──
        # For each unique diarization speaker, try to match with enrolled profiles
        speaker_name_cache = {}  # Cache: diarization_label → (name, confidence)

        for seg in labelled_segments:
            diarization_label = seg["speaker"]
            text = seg.get("text", "").strip()
            if not text:
                continue

            # Check if we already identified this speaker
            if diarization_label in speaker_name_cache:
                real_name, confidence = speaker_name_cache[diarization_label]
            else:
                # Try to identify this speaker using voice embeddings
                real_name, confidence = _identify_diarized_speaker(
                    audio_path=audio_path,
                    start=seg["start"],
                    end=seg["end"],
                    enrolled_profiles=enrolled_profiles,
                    temp_segments=temp_segments
                )
                speaker_name_cache[diarization_label] = (real_name, confidence)

            # ── Step 6: Map to session's speaker index ──
            speaker_idx = session.get_or_assign_speaker_index(diarization_label)

            # Determine final display name:
            # Priority: manual name > recognized name > auto label
            if speaker_idx in session.manual_names:
                display_name = session.manual_names[speaker_idx]
                confidence = 1.0  # Manual names are 100% confident
            elif real_name != "Unknown Speaker":
                display_name = real_name
                # Auto-set the name in the session for future chunks
                session.auto_labels[speaker_idx] = real_name
            else:
                display_name = session.get_speaker_name(speaker_idx)

            results.append({
                "speaker": display_name,
                "speakerIndex": speaker_idx,
                "text": text,
                "start": round(seg["start"], 2),
                "end": round(seg["end"], 2),
                "confidence": confidence
            })

        elapsed = round(time.time() - pipeline_start, 2)
        logger.info(
            f"Pipeline complete: {len(results)} segments, "
            f"{len(speaker_name_cache)} speakers, {elapsed}s total"
        )

    except Exception as e:
        logger.error(f"Pipeline error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Clean up any temporary segment files
        cleanup_temp_files(*temp_segments)

    return results


def _identify_diarized_speaker(
    audio_path: str,
    start: float,
    end: float,
    enrolled_profiles: list[dict],
    temp_segments: list[str]
) -> tuple[str, float]:
    """
    Try to identify a speaker from a diarized segment.

    Extracts the audio segment, computes its embedding, and matches
    against enrolled profiles.

    Args:
        audio_path: Full audio file path.
        start: Segment start time in seconds.
        end: Segment end time in seconds.
        enrolled_profiles: Enrolled voice profiles with embeddings.
        temp_segments: List to append temp file paths for cleanup.

    Returns:
        Tuple of (speaker_name, confidence).
    """
    if not enrolled_profiles or not is_recognition_loaded():
        # No profiles to match against, or SpeechBrain not loaded
        if enrolled_profiles and not is_recognition_loaded():
            # Try to load SpeechBrain (first time)
            try:
                from ai_server.speaker_recognition import get_speaker_encoder
                get_speaker_encoder()
            except Exception:
                return "Unknown Speaker", 0.0

    if not enrolled_profiles:
        return "Unknown Speaker", 0.0

    # Only attempt recognition on segments >= 1.5 seconds
    duration = end - start
    if duration < 1.5:
        logger.debug(f"Segment too short for recognition ({duration:.1f}s < 1.5s)")
        return "Unknown Speaker", 0.0

    # Extract the audio segment to a temp file
    segment_path = os.path.join(
        UPLOAD_FOLDER,
        f"seg_{int(start*100)}_{int(end*100)}_{int(time.time())}.wav"
    )
    temp_segments.append(segment_path)

    extracted = extract_segment_audio(audio_path, start, end, segment_path)
    if not extracted:
        return "Unknown Speaker", 0.0

    # Extract embedding for this segment
    segment_embedding = extract_embedding(segment_path)
    if segment_embedding is None:
        return "Unknown Speaker", 0.0

    # Match against enrolled profiles
    return identify_speaker_from_embedding(segment_embedding, enrolled_profiles)


def merge_consecutive_speakers(segments: list[dict]) -> list[dict]:
    """
    Merge consecutive segments from the same speaker into single blocks.

    This is the same logic as in the original ai_meeting_server.py,
    preserved here for the new pipeline.

    Args:
        segments: List of segment dicts with 'speakerIndex' and 'text'.

    Returns:
        Merged list of segments where consecutive same-speaker blocks
        are combined.
    """
    if not segments:
        return segments

    merged = [segments[0].copy()]

    for seg in segments[1:]:
        prev = merged[-1]
        if seg["speakerIndex"] == prev["speakerIndex"]:
            # Same speaker — merge text
            prev["text"] = prev["text"].rstrip() + " " + seg["text"].lstrip()
            prev["end"] = seg["end"]
            # Keep the higher confidence
            prev["confidence"] = max(
                prev.get("confidence", 0.0),
                seg.get("confidence", 0.0)
            )
        else:
            merged.append(seg.copy())

    return merged
