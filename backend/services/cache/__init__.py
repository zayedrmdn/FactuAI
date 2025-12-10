"""
Cache services for FactuAI.

Provides caching functionality for various data types.
"""

from services.cache.article_cache import (
    get_article_cache,
    save_article_cache,
    clear_article_cache
)

__all__ = [
    "get_article_cache",
    "save_article_cache",
    "clear_article_cache",
]
