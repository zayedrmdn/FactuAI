"""
input_summary.py
Summarisation utilities for the *original user input* (e.g., multi-claim or paragraph).
"""

from __future__ import annotations
import re
from typing import Optional
from core.logging import logger

DEFAULT_INPUT_SUMMARY_MAX = 120  # max tokens budget for LLM
MIN_CHARS_THRESHOLD = 10
BAD_OUTPUT_MARKERS = {"fact_claim", "opinion", "nonsense", "fact_question"}


def summarise_input_text(text: str, llm, max_tokens: int = DEFAULT_INPUT_SUMMARY_MAX) -> str:
    """
    Summarize raw user input (paragraph / multi-claim) into a concise overview.

    Args:
        text: Original user text.
        llm: LLM client.
        max_tokens: Generation cap for underlying model call.

    Returns:
        A short summary or first sentence fallback.
    """
    if not text or not text.strip():
        return ""

    if hasattr(llm, "clear_cache"):
        llm.clear_cache()

    prompt = (
        "Summarize the following user text into a concise neutral overview. "
        "Do NOT classify; just summarize factual content.\n\n"
        f"Text:\n{text}\n\nSummary:"
    )

    try:
        raw = llm.generate_response(prompt, max_tokens=max_tokens).strip()
        summary = _clean_summary(raw)
        if _looks_bad(summary):
            return _fallback_first_sentence(text)
        return summary
    except Exception as e:
        logger.error(f"[PIPELINE] summarise_input_text failed: {e}")
        return _fallback_first_sentence(text)


def _clean_summary(s: str) -> str:
    """Clean up LLM output."""
    if s.startswith("Summary:"):
        s = s[8:].strip()
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def _looks_bad(summary: str) -> bool:
    """Check if the summary looks like a classification instead of a summary."""
    if len(summary) < MIN_CHARS_THRESHOLD:
        return True
    lower = summary.lower()
    return any(marker in lower for marker in BAD_OUTPUT_MARKERS)


def _fallback_first_sentence(text: str) -> str:
    """Extract the first sentence as a fallback."""
    # Simple sentence boundary detection
    for end in ['. ', '! ', '? ']:
        idx = text.find(end)
        if idx > 0:
            return text[:idx + 1].strip()
    # If no sentence boundary, return first 100 chars
    return text[:100].strip() + ('...' if len(text) > 100 else '')
