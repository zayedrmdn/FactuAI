"""
scraping.py
Web scraping utilities for extracting article content.
"""

import os
import json
import requests
from pathlib import Path

from core.logging import logger
from core.helpers import is_junk
from pipeline.config import SCRAPING_LOG_PATH, ARTICLE_CACHE_PATH

# Lazy-load heavy dependencies
_EMBED_MODEL = None
_ARTICLE_CACHE = None

# Headers to avoid 403 blocks from websites
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


def _get_embed_model():
    """Get singleton SentenceTransformer from service_manager."""
    try:
        from services.service_manager import service_manager
        return service_manager.get_sentence_transformer()
    except Exception as e:
        logger.warning(f"Failed to load SentenceTransformer: {e}")
        return None


def _get_article_cache() -> dict:
    """Load article cache from disk."""
    global _ARTICLE_CACHE
    if _ARTICLE_CACHE is None:
        cache_path = Path(ARTICLE_CACHE_PATH)
        if cache_path.exists():
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    _ARTICLE_CACHE = json.load(f)
            except Exception:
                _ARTICLE_CACHE = {}
        else:
            _ARTICLE_CACHE = {}
    return _ARTICLE_CACHE


def _save_article_cache():
    """Save article cache to disk."""
    cache = _get_article_cache()
    cache_path = Path(ARTICLE_CACHE_PATH)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save article cache: {e}")


def fetch_article_text(url: str) -> str:
    """Extract main article text - simplified version that works."""
    logger.debug(f"[SCRAPING] Starting extraction for: {url}")
    
    # Cache check
    cache = _get_article_cache()
    cached = cache.get(url)
    if cached:
        logger.debug(f"[SCRAPING] Returning cached content for: {url}")
        return cached

    try:
        from bs4 import BeautifulSoup
        
        response = requests.get(url, timeout=10, headers=HEADERS, allow_redirects=True)
        logger.debug(f"[SCRAPING] Response status: {response.status_code}")
        
        if response.status_code == 403:
            logger.warning(f"[SCRAPING] 403 Forbidden (blocked): {url}")
            return ""
        elif response.status_code == 404:
            logger.warning(f"[SCRAPING] 404 Not Found: {url}")
            return ""
        elif response.status_code != 200:
            logger.warning(f"[SCRAPING] HTTP {response.status_code}: {url}")
            return ""
        
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text() for p in paragraphs])
        word_count = len(text.split())
        
        logger.debug(f"[SCRAPING] Extracted {word_count} words")
        
        if word_count >= 50:
            logger.debug(f"[SCRAPING] Success: Used BeautifulSoup for {url}")
            cache[url] = text.strip()
            _save_article_cache()
            return text.strip()
        else:
            logger.debug(f"[SCRAPING] Content too short: {word_count} words")
            return ""
        
    except Exception as e:
        logger.error(f"[SCRAPING] Extraction failed for {url}: {e}")
        return ""


def best_sentences(text: str, claim: str, k: int) -> list:
    """Return top-k sentences semantically closest to claim."""
    logger.debug(f"[SCRAPING] best_sentences: claim={claim!r}, text_len={len(text)}")
    if not text:
        return []
    
    try:
        import nltk
        nltk.download("punkt", quiet=True)
        nltk.download("punkt_tab", quiet=True)
        from nltk.tokenize import sent_tokenize
    except Exception as e:
        logger.warning(f"NLTK not available: {e}")
        # Fallback to simple sentence splitting
        raw = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        return raw[:k]
    
    raw = sent_tokenize(text)
    # Basic structural filter
    raw = [s.strip() for s in raw if len(s) > 30 and not is_junk(s)]
    if not raw:
        logger.debug("[SCRAPING] no valid sentences after initial filtering")
        return []

    embed_model = _get_embed_model()
    if embed_model is None:
        # Return first k sentences if no embedding model
        return raw[:k]

    try:
        from sentence_transformers import util
        
        # Use semantic similarity with a relevance threshold
        claim_emb = embed_model.encode(claim, convert_to_tensor=True)
        sent_embs = embed_model.encode(raw, convert_to_tensor=True)
        cos = util.cos_sim(claim_emb, sent_embs)
        if cos is None:
            raise RuntimeError("cos_sim returned None")

        try:
            row = cos[0]
        except Exception:
            row = None

        if row is None:
            try:
                sims = list(cos)[0]
            except Exception:
                raise RuntimeError("Unexpected shape from cos_sim result")
        else:
            sims = row.cpu().tolist() if hasattr(row, "cpu") else list(row)

        scored = list(zip(sims, raw))
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Only keep sentences with decent similarity
        relevant = [s for score, s in scored if score > 0.25]
        top_k = relevant[:k] if relevant else [s for _, s in scored[:k]]
        
        logger.debug(f"[SCRAPING] selected -> {top_k}")
        return top_k
    except Exception as e:
        logger.error(f"[SCRAPING] Semantic ranking failed: {e}")
        return raw[:k]
