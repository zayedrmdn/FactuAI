"""
ranker.py
Semantic similarity ranking for candidate sentences.
"""

from typing import List, Tuple
from core.logging import logger


def _get_embed_model():
    """Get the singleton SentenceTransformer from service_manager."""
    try:
        from services.service_manager import service_manager
        return service_manager.get_sentence_transformer()
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer: {e}")
        return None


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
        cos = util.cos_sim(claim_emb, sent_embs)
        # defensive: ensure cos is valid and has expected shape
        if cos is None:
            raise RuntimeError("cos_sim returned None")

        # cos may be a tensor; try to access first row safely
        try:
            row = cos[0]
        except Exception:
            # fallback: convert to list and take first element if possible
            row = None

        if row is None:
            # attempt to coerce to list-of-lists
            try:
                sims = list(cos)[0]
            except Exception:
                raise RuntimeError("Unexpected shape from cos_sim result")
        else:
            sims = row.cpu().tolist() if hasattr(row, "cpu") else list(row)

        scored = list(zip(sentences, sims))
        return sorted(scored, key=lambda x: x[1], reverse=True)
    except Exception as e:
        logger.error(f"[RANKER] Semantic ranking failed: {e}")
        return [(s, 0.0) for s in sentences]
