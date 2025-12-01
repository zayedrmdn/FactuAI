"""
candidate.py
Responsible ONLY for:
1. Collecting raw search + NewsAPI items
2. Fetching article text
3. Extracting candidate sentences per article

Used by orchestrator.py (build_evidence orchestration).
"""

from __future__ import annotations
from typing import List, Dict, Tuple

from pipeline.config import MAX_EVIDENCE_WORDS, SENTS_PER_ARTICLE_DEFAULT, MIN_SENT_WORDS
from pipeline.fetchers.newsapi import fetch_newsapi_articles
from pipeline.fetchers.scraping import fetch_article_text, best_sentences
from core.logging import logger


def collect_search_items(
    search_resp: dict,
    claim: str,
    max_google: int = 5,
    max_news: int = 5,
    timeframe: str = "RECENT"
) -> List[Dict]:
    """
    Merge Google Custom Search 'items' + freshly fetched NewsAPI articles.
    Returns unified list: [{title, url, source}, ...]
    """
    google_raw = (search_resp.get("items") or [])[:max_google]
    news_raw = fetch_newsapi_articles(claim, max_results=max_news, timeframe=timeframe)

    # Add debug logging for URLs
    logger.debug("[CANDIDATE] Google URLs collected:")
    for i, g in enumerate(google_raw):
        url = g.get("link", "")
        title = g.get("title", "")
        logger.debug(f"  {i+1}. {url}")
        logger.debug(f"     Title: {title}")
    
    logger.debug("[CANDIDATE] NewsAPI URLs collected:")
    for i, n in enumerate(news_raw):
        url = n.get("link", "")
        title = n.get("title", "")
        logger.debug(f"  {i+1}. {url}")
        logger.debug(f"     Title: {title}")

    items: List[Dict] = []
    for g in google_raw:
        items.append({
            "title": g.get("title", ""),
            "url": g.get("link", ""),
            "source": "Google"
        })
    for n in news_raw:
        items.append({
            "title": n.get("title", ""),
            "url": n.get("link", ""),
            "source": "NewsAPI"
        })

    logger.debug(
        f"[PIPELINE] (candidates) collected search items: "
        f"google={len(google_raw)} news={len(news_raw)} total={len(items)}"
    )
    return items


def extract_candidates(
    items: List[Dict],
    claim: str,
    sents_per_article: int = SENTS_PER_ARTICLE_DEFAULT
) -> Tuple[List[Dict], List[str]]:
    """Extract candidate sentences from articles."""
    candidates: List[Dict] = []
    urls: List[str] = []

    for it in items:
        url = it.get("url")
        if not url:
            continue

        text = fetch_article_text(url)
        if not text:
            continue

        urls.append(url)

        # Get sentences from scraping
        raw = best_sentences(text, claim, k=sents_per_article * 2)
        
        # Filter by minimum word count
        for sent in raw:
            if len(sent.split()) >= MIN_SENT_WORDS:
                candidates.append({
                    "text": sent,
                    "url": url,
                    "source": it.get("source", "Unknown"),
                    "title": it.get("title", ""),
                    "score": 0.0  # Will be set by ranker
                })

    logger.debug(f"[CANDIDATE] Extracted {len(candidates)} candidates from {len(urls)} URLs")
    return candidates, urls
