# services/classifier/cleaner.py

import re
import nltk
from nltk.tokenize import sent_tokenize
from core.logging import logger

def clean_evidence_text(
    text: str,
    max_sentences: int = 4,
    max_chars: int = 600
) -> str:
    """
    Clean evidence by removing artifacts (citations, URLs, stray punctuation)
    and keeping up to max_sentences within a character limit.
    """
    # ensure sentence tokenizer is available
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
        logger.debug("[CLASSIFIER] Downloaded NLTK punkt tokenizer")

    # strip out citation markers like [123], stray punctuation, URLs
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"[-*\d\.\)\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # split into sentences
    sentences = sent_tokenize(text)

    # collect up to max_sentences under max_chars
    cleaned = []
    total_chars = 0
    for sent in sentences:
        if total_chars + len(sent) > max_chars:
            break
        cleaned.append(sent)
        total_chars += len(sent)
        if len(cleaned) >= max_sentences:
            break

    return " ".join(cleaned)