"""
Article caching service.

Provides persistent caching for scraped articles to avoid re-fetching.
"""

import json
from pathlib import Path
from typing import Dict

from utils.logging import get_logger
from config import ARTICLE_CACHE_PATH

logger = get_logger(__name__)

# Global cache singleton
_ARTICLE_CACHE = None


def get_article_cache() -> Dict[str, str]:
    """
    Load article cache from disk.
    
    Returns:
        Dictionary mapping URLs to article text
    """
    global _ARTICLE_CACHE
    if _ARTICLE_CACHE is None:
        cache_path = Path(ARTICLE_CACHE_PATH)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    _ARTICLE_CACHE = json.load(f)
                logger.debug(f"[CACHE] Loaded {len(_ARTICLE_CACHE)} cached articles")
            except Exception as e:
                logger.warning(f"[CACHE] Failed to load cache: {e}")
                _ARTICLE_CACHE = {}
        else:
            _ARTICLE_CACHE = {}
    return _ARTICLE_CACHE


def save_article_cache(cache: Dict[str, str] = None):
    """
    Save article cache to disk.
    
    Args:
        cache: Cache dictionary to save. If None, uses global cache.
    """
    if cache is None:
        cache = get_article_cache()
    
    cache_path = Path(ARTICLE_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
        logger.debug(f"[CACHE] Saved {len(cache)} articles to cache")
    except Exception as e:
        logger.warning(f"[CACHE] Failed to save cache: {e}")


def clear_article_cache():
    """Clear the article cache from memory and disk."""
    global _ARTICLE_CACHE
    _ARTICLE_CACHE = {}
    cache_path = Path(ARTICLE_CACHE_PATH)
    if cache_path.exists():
        try:
            cache_path.unlink()
            logger.info("[CACHE] Cache cleared")
        except Exception as e:
            logger.warning(f"[CACHE] Failed to clear cache file: {e}")


__all__ = ["get_article_cache", "save_article_cache", "clear_article_cache"]
