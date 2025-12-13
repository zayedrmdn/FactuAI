"""
Web scraping functionality.

Extracts main article text from URLs with caching support.
"""

from pathlib import Path
from typing import Dict

import requests
from bs4 import BeautifulSoup

from utils.logging import get_logger
from utils.helpers import is_junk

try:
    from services.cache.article_cache import get_article_cache, save_article_cache  # type: ignore
except Exception:  # Fallback in case cache service is unavailable
    _ARTICLE_CACHE: Dict[str, str] = {}

    def get_article_cache() -> Dict[str, str]:
        return _ARTICLE_CACHE

    def save_article_cache(cache: Dict[str, str]) -> None:
        _ARTICLE_CACHE.update(cache)

logger = get_logger(__name__)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def scrape_article(url: str) -> str:
    """
    Extract main article text from URL.
    
    Args:
        url: Article URL
        
    Returns:
        Extracted article text, or empty string if failed
    """
    cache = get_article_cache()
    if url in cache:
        logger.debug(f"[SCRAPING] Cache hit for: {url}")
        return cache[url]
    
    try:
        response = requests.get(url, timeout=10, headers=HEADERS, allow_redirects=True)

        if response.status_code == 403:
            logger.warning(f"[SCRAPING] 403 Forbidden: {url}")
            return ""
        if response.status_code == 404:
            logger.warning(f"[SCRAPING] 404 Not Found: {url}")
            return ""
        if response.status_code != 200:
            logger.warning(f"[SCRAPING] HTTP {response.status_code}: {url}")
            return ""
        
        soup = BeautifulSoup(response.content, "html.parser")
        
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        paragraphs = soup.find_all("p")
        text = " ".join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        sentences = text.split(". ")
        clean_sentences = [s for s in sentences if not is_junk(s)]
        text = ". ".join(clean_sentences)
        
        word_count = len(text.split())
        logger.debug(f"[SCRAPING] Extracted {word_count} words from {url}")
        
        MAX_SCRAPE_WORDS = 5000
        if word_count > MAX_SCRAPE_WORDS:
            logger.warning(
                f"[SCRAPING] Truncating massive article ({word_count} words) to {MAX_SCRAPE_WORDS} words: {url}"
            )
            words = text.split()[:MAX_SCRAPE_WORDS]
            text = " ".join(words)
            word_count = MAX_SCRAPE_WORDS
        
        if text and word_count > 50:
            cache[url] = text
            save_article_cache(cache)
        
        return text

    except Exception as e:
        logger.error(f"[SCRAPING] Failed to scrape {url}: {e}")
        return ""


__all__ = ["scrape_article"]
