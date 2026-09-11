"""
AirScribe AI Server — Summary Generation
============================================
Generates structured meeting summaries from speaker-labelled transcripts.

Purpose:
    After a meeting ends, this module produces:
    - A brief overview paragraph
    - Key discussion points
    - Action items with assignees
    - Decisions made

    Uses extractive summarization (local, free) — no paid APIs.

How it connects:
    Called by ai_meeting_server.py when a meeting session ends.
    Can also be called standalone via the /ai/summary route.

How to configure:
    No special configuration needed. Uses only Python stdlib + basic NLP.

How to debug:
    The generate_summary() function logs the input segment count and
    output summary length.
"""

import re
import logging
from collections import Counter
from typing import Optional

logger = logging.getLogger("airscribe.ai.summary")


def generate_summary(transcript_segments: list[dict]) -> dict:
    """
    Generate a structured meeting summary from transcript segments.

    Uses extractive summarization: identifies the most important
    sentences based on word frequency and position.

    Args:
        transcript_segments: List of segment dicts with 'speaker', 'text'.
            Example:
            [
                {"speaker": "Archit", "text": "Let's discuss the timeline."},
                {"speaker": "Rahul", "text": "I think we need two more weeks."},
            ]

    Returns:
        Dict containing:
            - "overview": str (1-2 sentence summary)
            - "keyPoints": list[str] (main discussion points)
            - "actionItems": list[dict] (with 'task' and 'assignee')
            - "decisions": list[str] (key decisions made)
            - "speakerStats": dict (speaking time / turn count per speaker)
            - "duration": str (total meeting duration)
    """
    if not transcript_segments:
        return _empty_summary()

    logger.info(f"Generating summary from {len(transcript_segments)} segments")

    # ── Combine all text ──
    full_text = " ".join(seg.get("text", "") for seg in transcript_segments).strip()
    if not full_text:
        return _empty_summary()

    # ── Speaker statistics ──
    speaker_stats = _compute_speaker_stats(transcript_segments)

    # ── Extract sentences ──
    sentences = _split_sentences(full_text)

    # ── Key points (top sentences by importance) ──
    key_points = _extract_key_points(sentences, max_points=5)

    # ── Action items (heuristic extraction) ──
    action_items = _extract_action_items(transcript_segments)

    # ── Decisions (heuristic extraction) ──
    decisions = _extract_decisions(sentences)

    # ── Overview ──
    speakers = list(speaker_stats.keys())
    overview = _generate_overview(speakers, len(sentences), key_points)

    # ── Duration ──
    duration = _format_duration(transcript_segments)

    summary = {
        "overview": overview,
        "keyPoints": key_points,
        "actionItems": action_items,
        "decisions": decisions,
        "speakerStats": speaker_stats,
        "duration": duration
    }

    logger.info(
        f"Summary generated: {len(key_points)} key points, "
        f"{len(action_items)} action items, {len(decisions)} decisions"
    )

    return summary


def _empty_summary() -> dict:
    """Return an empty summary structure."""
    return {
        "overview": "No content to summarize.",
        "keyPoints": [],
        "actionItems": [],
        "decisions": [],
        "speakerStats": {},
        "duration": "0:00"
    }


def _compute_speaker_stats(segments: list[dict]) -> dict:
    """Compute per-speaker statistics (word count, turn count)."""
    stats = {}
    for seg in segments:
        speaker = seg.get("speaker", "Unknown Speaker")
        text = seg.get("text", "")
        word_count = len(text.split())

        if speaker not in stats:
            stats[speaker] = {"turns": 0, "words": 0}
        stats[speaker]["turns"] += 1
        stats[speaker]["words"] += word_count

    return stats


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex."""
    # Split on period, exclamation, or question mark followed by space or end
    sentences = re.split(r'(?<=[.!?])\s+', text)
    # Filter out very short fragments
    return [s.strip() for s in sentences if len(s.strip()) > 10]


def _extract_key_points(sentences: list[str], max_points: int = 5) -> list[str]:
    """
    Extract the most important sentences using word frequency scoring.
    Simple but effective extractive summarization.
    """
    if not sentences:
        return []

    if len(sentences) <= max_points:
        return sentences

    # Word frequency scoring
    all_words = " ".join(sentences).lower().split()
    # Remove common stop words
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "out", "off", "over",
        "under", "again", "further", "then", "once", "here", "there", "when",
        "where", "why", "how", "all", "each", "every", "both", "few", "more",
        "most", "other", "some", "such", "no", "nor", "not", "only", "own",
        "same", "so", "than", "too", "very", "just", "because", "but", "and",
        "or", "if", "while", "that", "this", "it", "its", "i", "we", "you",
        "he", "she", "they", "me", "him", "her", "us", "them", "my", "your",
        "his", "their", "our", "what", "which", "who", "whom"
    }
    meaningful_words = [w for w in all_words if w not in stop_words and len(w) > 2]
    word_freq = Counter(meaningful_words)

    # Score each sentence
    scored = []
    for i, sentence in enumerate(sentences):
        words = sentence.lower().split()
        score = sum(word_freq.get(w, 0) for w in words if w not in stop_words)
        # Boost early sentences (position bias)
        position_boost = 1.0 + (0.5 * (1.0 - i / len(sentences)))
        scored.append((score * position_boost, i, sentence))

    # Sort by score descending, take top N, then reorder by original position
    scored.sort(key=lambda x: x[0], reverse=True)
    top = scored[:max_points]
    top.sort(key=lambda x: x[1])  # Reorder by original position

    return [s[2] for s in top]


def _extract_action_items(segments: list[dict]) -> list[dict]:
    """
    Extract action items using keyword heuristics.
    Looks for patterns like "I will...", "We need to...", "Please...", etc.
    """
    action_patterns = [
        r'\b(i will|i\'ll|we will|we\'ll|we need to|we should|please|make sure|ensure|follow up)\b',
        r'\b(action item|todo|to do|assigned to|responsible for|deadline)\b',
        r'\b(let\'s|let me|can you|could you|would you)\b',
    ]
    combined_pattern = "|".join(f"({p})" for p in action_patterns)

    action_items = []
    seen_texts = set()

    for seg in segments:
        text = seg.get("text", "")
        speaker = seg.get("speaker", "Unknown")

        if re.search(combined_pattern, text.lower()):
            # Avoid duplicates
            normalized = text.strip().lower()[:50]
            if normalized not in seen_texts:
                seen_texts.add(normalized)
                action_items.append({
                    "task": text.strip(),
                    "assignee": speaker
                })

    return action_items[:10]  # Cap at 10 items


def _extract_decisions(sentences: list[str]) -> list[str]:
    """
    Extract decisions using keyword heuristics.
    Looks for patterns like "decided", "agreed", "confirmed", etc.
    """
    decision_patterns = [
        r'\b(decided|agreed|confirmed|approved|finalized|concluded)\b',
        r'\b(the decision is|we agreed|it was decided|consensus)\b',
        r'\b(going with|go ahead with|moving forward with)\b',
    ]
    combined = "|".join(f"({p})" for p in decision_patterns)

    decisions = []
    for sentence in sentences:
        if re.search(combined, sentence.lower()):
            decisions.append(sentence.strip())

    return decisions[:5]


def _generate_overview(
    speakers: list[str],
    sentence_count: int,
    key_points: list[str]
) -> str:
    """Generate a 1-2 sentence meeting overview."""
    speaker_list = ", ".join(speakers[:4])
    if len(speakers) > 4:
        speaker_list += f" and {len(speakers) - 4} others"

    overview = f"Meeting between {speaker_list}. "
    if key_points:
        overview += key_points[0]

    return overview


def _format_duration(segments: list[dict]) -> str:
    """Calculate and format meeting duration from segments."""
    if not segments:
        return "0:00"

    starts = [s.get("start", 0) for s in segments]
    ends = [s.get("end", 0) for s in segments]

    if not ends:
        return "0:00"

    total_seconds = max(ends) - min(starts) if starts else 0
    minutes = int(total_seconds // 60)
    seconds = int(total_seconds % 60)
    return f"{minutes}:{seconds:02d}"
