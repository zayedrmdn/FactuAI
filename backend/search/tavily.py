"""
Tavily AI Search integration.

Provides AI-powered search with answer generation.
"""

from typing import List, Dict

from utils.logging import get_logger
from utils.helpers import SearchError
from config import TAVILY_API_KEY

logger = get_logger(__name__)


def search_tavily(query: str, num_results: int = 5) -> List[Dict[str, str]]:
    """
    Search using Tavily API with AI-generated answer.
    
    This function provides answer-seeking capabilities where we want
    a direct, AI-generated answer to verification questions.
    
    Args:
        query: Natural language verification question
        num_results: Maximum number of search results to return (default: 5)
        
    Returns:
        List of dicts with keys: title, url, source, score, content, answer
        First result always contains the AI-generated answer if available.
        
    Raises:
        SearchError: If Tavily API key not configured or search fails
    """
    if not TAVILY_API_KEY:
        raise SearchError("Tavily API key not configured (TAVILY_API_KEY)")
    
    if not query or not query.strip():
        raise SearchError("Query cannot be empty")
    
    try:
        # Lazy-load Tavily client
        from tavily import TavilyClient
        client = TavilyClient(TAVILY_API_KEY)
        
        logger.info(f"[TAVILY] Searching: {query}")
        
        # Execute search with advanced answer generation
        response = client.search(
            query=query,
            max_results=num_results,
            include_answer="advanced"  # Get detailed AI-generated answer
        )
        
        results = []
        
        # Add AI-generated answer as the first "result" if available
        answer = response.get("answer")
        if answer:
            results.append({
                "title": "AI-Generated Answer",
                "url": "",
                "source": "Tavily AI",
                "score": 1.0,
                "content": answer,
                "answer": answer  # Mark this as the AI answer
            })
        
        # Add search results
        for item in response.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "source": "Tavily",
                "score": item.get("score", 0.0),
                "content": item.get("content", "")
            })
        
        logger.info(
            f"[TAVILY] Found {len(results)} results "
            f"(answer={'Yes' if answer else 'No'})"
        )
        
        return results
        
    except ImportError as e:
        raise SearchError("tavily-python package not installed. Run: pip install tavily-python") from e
    except Exception as e:
        logger.error(f"[TAVILY] Search failed: {e}", exc_info=True)
        raise SearchError(f"Tavily search failed: {e}") from e


__all__ = ["search_tavily"]
