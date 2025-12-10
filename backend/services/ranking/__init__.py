"""
Ranking services for FactuAI.

Provides semantic similarity and ranking functionality.
"""

from services.ranking.embeddings import get_embed_model
from services.ranking.scorer import rank_sentences

__all__ = [
    "get_embed_model",
    "rank_sentences",
]
