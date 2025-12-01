"""
evidence_summary.py
Summarisation utilities for *selected evidence* text.
"""

from __future__ import annotations
import re
from typing import Optional
from core.logging import logger

# Reasonable caps for generated summary length
MIN_SUMMARY_WORDS = 20
MAX_SUMMARY_WORDS = 50


def summarise_evidence(evidence: str, llm, force_ratio: float = 0.25) -> str:
    """
    Summarise the final evidence block.

    Args:
        evidence: Raw evidence text (already selected / concatenated sentences)
        llm: LLM client with generate_response()
        force_ratio: Fraction of evidence word count to target (clamped to bounds)

    Returns:
        Summary string (may be truncated fallback if LLM fails)
    """
    if not evidence or not evidence.strip():
        return ""

    if hasattr(llm, "clear_cache"):
        llm.clear_cache()

    words = evidence.split()
    target = max(MIN_SUMMARY_WORDS,
                 min(MAX_SUMMARY_WORDS, max(1, int(len(words) * force_ratio))))

    prompt = (
        f"Summarize the following evidence in about {target} words. "
        f"Keep it factual, neutral, and concise. Avoid repetition.\n\n"
        f"Evidence:\n{evidence}\n\nSummary:"
    )

    try:
        raw = llm.generate_response(prompt, max_tokens=target + 30).strip()
        summary = _clean_summary_output(raw)
        summary = _truncate_to_words(summary, target)
        if summary:
            return summary
    except Exception as e:
        logger.error(f"[PIPELINE] summarise_evidence LLM failed: {e}")

    return _fallback_summary(evidence, target)


def _clean_summary_output(text: str) -> str:
    """Clean up LLM output."""
    if text.startswith("Summary:"):
        text = text[8:].strip()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def _truncate_to_words(text: str, max_words: int) -> str:
    """Truncate text to max words, ending at sentence boundary if possible."""
    words = text.split()
    if len(words) <= max_words:
        return text
    
    truncated = ' '.join(words[:max_words])
    # Try to end at a sentence boundary
    last_period = truncated.rfind('.')
    if last_period > len(truncated) // 2:
        return truncated[:last_period + 1]
    return truncated + '...'


def _fallback_summary(evidence: str, target_words: int) -> str:
    """Create a simple fallback summary from the evidence."""
    words = evidence.split()
    return ' '.join(words[:target_words]) + ('...' if len(words) > target_words else '')
