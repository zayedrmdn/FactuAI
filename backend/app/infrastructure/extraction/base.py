"""
Base text extraction utilities.

Provides sentence extraction and text processing.
"""

import os
import re
from typing import List

from utils.logging import get_logger
from utils.helpers import is_junk

logger = get_logger(__name__)

try:
    from config import MIN_SENT_WORDS as _MIN_SENT_WORDS  # type: ignore
except Exception:
    _MIN_SENT_WORDS = int(os.getenv("MIN_SENT_WORDS", "5"))

MIN_SENT_WORDS = _MIN_SENT_WORDS


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from article text.
    
    Args:
        text: Article text
        
    Returns:
        List of sentences (min MIN_SENT_WORDS words each)
    """
    if not text:
        return []
    
    sentences = re.split(r"[.!?]+\s+", text)
    
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) >= MIN_SENT_WORDS and not is_junk(sent):
            valid_sentences.append(sent)
    
    return valid_sentences


__all__ = ["extract_sentences", "MIN_SENT_WORDS"]
