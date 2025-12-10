"""
Google Custom Search integration.

Provides search functionality using Google Custom Search API.
"""

import requests
from typing import List, Dict

from utils.logging import get_logger
from utils.helpers import SearchError
from config import GOOGLE_API_KEY, GOOGLE_CX_ID

logger = get_logger(__name__)


def search_google(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search Google Custom Search API.
    
    Args:
        query: Search query string
        num_results: Number of results to return
        
    Returns:
        List of dicts with 'title', 'url', 'source' keys
        
    Raises:
        SearchError: If search fails
    """
    if not GOOGLE_API_KEY or not GOOGLE_CX_ID:
        logger.error(
            "[SEARCH] Google API credentials not configured (GOOGLE_API_KEY, GOOGLE_CX_ID/GOOGLE_CSE_ID)"
        )
        raise SearchError("Google API credentials not configured")
    
    try:
        # Try official client first
        try:
            from googleapiclient.discovery import build
            service = build("customsearch", "v1", developerKey=GOOGLE_API_KEY)
            response = service.cse().list(
                q=query,
                cx=GOOGLE_CX_ID,
                num=num_results
            ).execute()
            items = response.get("items", [])
        except ImportError:
            # Fallback to requests
            logger.info("[SEARCH] googleapiclient not installed; falling back to requests client")
            response = requests.get(
                "https://www.googleapis.com/customsearch/v1",
                params={
                    "key": GOOGLE_API_KEY,
                    "cx": GOOGLE_CX_ID,
                    "q": query,
                    "num": num_results
                },
                timeout=10
            ).json()
            
            if response.get("error"):
                raise SearchError(response["error"].get("message", "Unknown API error"))
            
            items = response.get("items", [])
        
        # Format results
        results = []
        for item in items:
            url = item.get("link", "").strip()
            if url.startswith(("http://", "https://")):
                results.append({
                    "title": item.get("title", ""),
                    "url": url,
                    "source": "Google"
                })
        
        logger.debug(f"[SEARCH] Google returned {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"[SEARCH] Google search failed: {e}")
        raise SearchError(f"Google search failed: {e}")


__all__ = ["search_google"]
