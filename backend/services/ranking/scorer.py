"""
Semantic similarity scoring service.

Provides sentence ranking based on semantic similarity to claims.
"""

from typing import List, Tuple

from utils.logging import get_logger
from services.ranking.embeddings import get_embed_model

logger = get_logger(__name__)


def rank_sentences(claim: str, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Rank sentences by semantic similarity to claim.
    
    Args:
        claim: The claim to check
        sentences: List of candidate sentences
        
    Returns:
        List of (sentence, score) tuples sorted by score (descending)
    """
    if not sentences:
        return []
    
    model = get_embed_model()
    if not model:
        # Fallback: return sentences with zero scores
        logger.warning("[RANKING] No model available, returning unranked")
        return [(s, 0.0) for s in sentences]
    
    try:
        from sentence_transformers import util
        
        claim_embedding = model.encode(claim, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        
        # Compute cosine similarity
        scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]
        scores = scores.cpu().tolist() if hasattr(scores, "cpu") else list(scores)
        
        # Combine and sort
        ranked = list(zip(sentences, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"[RANKING] Ranked {len(ranked)} sentences")
        return ranked
        
    except Exception as e:
        logger.error(f"[RANKING] Semantic ranking failed: {e}")
        return [(s, 0.0) for s in sentences]


__all__ = ["rank_sentences"]
