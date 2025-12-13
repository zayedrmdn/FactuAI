"""
Base text extraction utilities.

Provides sentence extraction and text processing.
"""

import re
from typing import List

from utils.logging import get_logger
from utils.helpers import is_junk
from config import MIN_SENT_WORDS

logger = get_logger(__name__)


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from article text.
    
    Args:
        text: Article text
        
    Returns:
        List of sentences (min 5 words each)
    """
    if not text:
        return []
    
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Filter by minimum word count and junk detection
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) >= MIN_SENT_WORDS and not is_junk(sent):
            valid_sentences.append(sent)
    
    return valid_sentences


__all__ = ["extract_sentences"]
