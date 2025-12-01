"""
keyword_extractor.py
Keyword extraction using KeyBERT.
"""

from core.logging import logger

# Lazy-load KeyBERT model
_kw_model = None


def _get_keybert_model():
    """Lazy load KeyBERT model."""
    global _kw_model
    if _kw_model is None:
        try:
            from keybert import KeyBERT
            _kw_model = KeyBERT(model="all-MiniLM-L6-v2")
            logger.debug("[KEYBERT] KeyBERT model initialized.")
        except Exception as e:
            logger.warning(f"Failed to load KeyBERT: {e}")
            _kw_model = False
    return _kw_model if _kw_model is not False else None


def extract_keywords(text: str, top_n: int = 5, ngram_range=(1, 2)) -> list:
    """
    Extract keywords from text using KeyBERT.
    
    Args:
        text: Input text
        top_n: Number of keywords to extract
        ngram_range: Range of n-grams to consider
        
    Returns:
        List of keyword strings
    """
    if not text:
        return []

    kw_model = _get_keybert_model()
    if kw_model is None:
        # Fallback: simple word extraction
        words = text.split()
        # Filter out common words and short words
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                      'would', 'could', 'should', 'may', 'might', 'must', 'shall',
                      'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                      'as', 'into', 'through', 'during', 'before', 'after', 'above',
                      'below', 'between', 'under', 'again', 'further', 'then', 'once',
                      'and', 'but', 'or', 'nor', 'so', 'yet', 'both', 'either', 'neither',
                      'not', 'only', 'own', 'same', 'than', 'too', 'very', 'just', 'also'}
        keywords = [w for w in words if len(w) > 3 and w.lower() not in stop_words]
        return keywords[:top_n]

    try:
        kws = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=ngram_range,
            stop_words="english",
            top_n=top_n,
        )
        logger.debug(f"[KEYBERT] Extracted {len(kws)} keywords from text.")
        return [kw for kw, _ in kws]
    except Exception as e:
        logger.error(f"[KEYBERT] Keyword extraction failed: {e}")
        return []
