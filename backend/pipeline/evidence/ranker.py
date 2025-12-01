"""
ranker.py
Semantic similarity ranking for candidate sentences.
"""

from typing import List, Tuple
from core.logging import logger

# Lazy load embedding model to avoid imports in cloud mode
_EMBED_MODEL = None


def _get_embed_model():
    """Lazy load the embedding model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
            logger.debug("SentenceTransformer loaded on CPU for ranking")
        except Exception as e:
            logger.warning(f"Failed to load SentenceTransformer: {e}")
            _EMBED_MODEL = False  # Mark as failed
    return _EMBED_MODEL if _EMBED_MODEL is not False else None


def rank_sentences(claim: str, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Compute cosine similarity between the claim and each sentence,
    returning a list of (sentence, score) sorted by descending score.
    Falls back to zero-scores if model is unavailable.
    """
    if not sentences:
        return []

    embed_model = _get_embed_model()
    if embed_model is None:
        return [(s, 0.0) for s in sentences]

    try:
        from sentence_transformers import util
        claim_emb = embed_model.encode(claim, convert_to_tensor=True)
        sent_embs = embed_model.encode(sentences, convert_to_tensor=True)
        sims = util.cos_sim(claim_emb, sent_embs)[0].cpu().tolist()
        scored = list(zip(sentences, sims))
        return sorted(scored, key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"[RANKER] Semantic ranking failed: {e}")
        return [(s, 0.0) for s in sentences]
