"""
NewsAPI integration.

Provides news article search using NewsAPI.
"""

from typing import List, Dict
from datetime import datetime, timedelta

from utils.logging import get_logger
from config import NEWS_API_KEY

logger = get_logger(__name__)


def search_newsapi(query: str, num_results: int = 5, timeframe: str = "RECENT") -> List[Dict[str, str]]:
    """
    Search NewsAPI for recent articles using official client.
    
    Args:
        query: Simple keywords (e.g., "COVID vaccines" not full sentences)
        num_results: Number of results (max 100 for free tier)
        timeframe: Time filter (RECENT, WEEK, MONTH, YEAR, LONG_AGO)
        
    Returns:
        List of dicts with 'title', 'url', 'source' keys
    """
    if not NEWS_API_KEY:
        logger.warning("[SEARCH] NewsAPI key not configured")
        return []
    
    try:
        from newsapi import NewsApiClient
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        
        # Extract simple keywords from the original claim/query
        # NewsAPI works best with 2-3 key terms, not full sentences
        words = query.lower().split()
        
        # Filter out common stop words and take key terms
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'that', 'this', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        
        key_terms = [word for word in words if len(word) > 2 and word not in stop_words and not word.isdigit()]
        
        # Take first 2-3 most important terms (prioritize proper nouns and important keywords)
        keywords = []
        for term in key_terms[:4]:  # Take up to 4 terms
            # Prioritize terms that look like proper nouns or important keywords
            if term[0].isupper() or term in ['covid', 'vaccine', 'vaccines', 'microchip', 'microchips', 'track', 'location', 'thoughts']:
                keywords.append(term)
                if len(keywords) >= 3:  # Max 3 keywords for NewsAPI
                    break
        
        # Fallback to first 2-3 terms if no priority terms found
        if not keywords:
            keywords = key_terms[:3]
        
        # Ensure we have at least 1 keyword
        if not keywords:
            keywords = ['news']  # Fallback
        
        keywords_str = ' '.join(keywords)
        
        # Convert timeframe to date range
        to_date = datetime.now()
        timeframe_map = {
            "RECENT": 7,
            "WEEK": 7,
            "MONTH": 30,
            "YEAR": 365,
            "LONG_AGO": None
        }
        
        days = timeframe_map.get(timeframe, 7)
        from_date = to_date - timedelta(days=days) if days else None
        
        # Use get_everything endpoint with keyword search
        response = newsapi.get_everything(
            q=keywords_str,
            language='en',
            sort_by='relevancy',
            page_size=min(num_results, 100),  # API limit
            from_param=from_date.strftime('%Y-%m-%d') if from_date else None,
            to=to_date.strftime('%Y-%m-%d')
        )
        
        if response.get('status') != 'ok':
            logger.warning(f"[SEARCH] NewsAPI error: {response.get('message')}")
            return []
        
        results = []
        for article in response.get('articles', []):
            url = article.get('url', '').strip()
            if url.startswith(('http://', 'https://')):
                source_name = article.get('source', {}).get('name', 'NewsAPI')
                results.append({
                    'title': article.get('title', ''),
                    'url': url,
                    'source': source_name
                })
        
        logger.debug(f"[SEARCH] NewsAPI returned {len(results)} results for keywords: '{keywords_str}'")
        return results
        
    except ImportError:
        logger.error("[SEARCH] newsapi-python not installed. Run: pip install newsapi-python")
        return []
    except Exception as e:
        logger.error(f"[SEARCH] NewsAPI search failed: {e}")
        return []


__all__ = ["search_newsapi"]
