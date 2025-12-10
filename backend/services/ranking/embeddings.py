"""
Embedding model service.

Provides singleton SentenceTransformer model for semantic similarity.
"""

import os
from pathlib import Path

# Set HF_HOME before ANY imports to suppress deprecation warning
_cache_dir = Path(__file__).resolve().parent.parent.parent.parent / ".cache" / "huggingface"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_cache_dir)

from utils.logging import get_logger

logger = get_logger(__name__)

# Global model singleton
_EMBED_MODEL = None


def get_embed_model():
    """
    Get singleton SentenceTransformer model.
    
    Returns:
        SentenceTransformer model or None if unavailable
    """
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBED_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("[RANKING] Loaded SentenceTransformer model")
        except ImportError:
            logger.warning("[RANKING] SentenceTransformer not available")
            _EMBED_MODEL = False  # Mark as unavailable
    
    return _EMBED_MODEL if _EMBED_MODEL is not False else None


__all__ = ["get_embed_model"]
