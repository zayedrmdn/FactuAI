"""
newsapi.py
NewsAPI integration for fetching news articles.
"""

from datetime import datetime, timedelta
from core.logging import logger
from pipeline.utils import extract_keywords
from pipeline.config import NEWS_API_KEY

# Lazy-load NewsAPI client
_newsapi = None


def _get_newsapi():
    """Lazy load NewsAPI client."""
    global _newsapi
    if _newsapi is None:
        try:
            from newsapi import NewsApiClient
            _newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        except Exception as e:
            logger.warning(f"Failed to initialize NewsAPI client: {e}")
            _newsapi = False
    return _newsapi if _newsapi is not False else None


def fetch_newsapi_articles(
    claim: str,
    max_results: int = 2,
    timeframe: str = "RECENT"
) -> list:
    """
    Fetch up to `max_results` articles for `claim`.
    - RECENT -> last 30 days
    - any other timeframe -> no date filter (free plan will still cap you at ~30 days)
    """
    newsapi = _get_newsapi()
    if newsapi is None:
        logger.warning("[NEWSAPI] Client not available")
        return []

    # 1) build your search query
    keywords = extract_keywords(claim)
    if keywords:
        query = " OR ".join(f'"{kw}"' for kw in keywords)
    else:
        query = claim

    # 2) core params
    params = {
        "q": query,
        "language": "en",
        "sort_by": "relevancy",
        "page_size": max_results
    }

    cutoff = None
    # 3) only add a 30-day from_param if the user really wants "RECENT"
    if timeframe.upper() == "RECENT":
        cutoff = (datetime.utcnow() - timedelta(days=30)).strftime("%Y-%m-%d")
        params["from_param"] = cutoff
        logger.debug(f"[NEWSAPI] 30-day window from {cutoff}")
    
    logger.debug(f"[NEWSAPI] Fetching articles: {params!r}")
    
    try:
        res = newsapi.get_everything(**params)

        # 4) if NewsAPI complains, retry once without from_param
        if res.get("status") != "ok":
            logger.warning(f"[NEWSAPI] error {res.get('code')}: {res.get('message')}, retrying without date filter")
            params.pop("from_param", None)
            res = newsapi.get_everything(**params)

        articles = res.get("articles", [])
        logger.debug(f"[NEWSAPI] Retrieved {len(articles)} articles")

        return [
            {
                "title": a.get("title", ""),
                "link": a.get("url", ""),
                "snippet": a.get("description", "")
            }
            for a in articles
        ]
    except Exception as e:
        logger.error(f"[NEWSAPI] Failed to fetch articles: {e}")
        return []
