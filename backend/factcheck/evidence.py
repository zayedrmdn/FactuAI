"""
Evidence Collection for FactuAI

Consolidated evidence gathering module combining:
- Search (Google Custom Search)
- Web scraping (article extraction)
- Ranking (semantic similarity)

Simplified from 15 files to 1 module with clear functions.
"""

# Set HF_HOME before ANY imports to suppress deprecation warning
import os
import sys
from pathlib import Path

_cache_dir = Path(__file__).resolve().parent.parent.parent / ".cache" / "huggingface"
_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_cache_dir)
# os.environ["TRANSFORMERS_CACHE"] = str(_cache_dir)  # Removed to fix deprecation warning

import json
import re
import requests
from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
from bs4 import BeautifulSoup

from utils.logging import get_logger
from utils.helpers import is_junk, SearchError, ScrapingError, EvidenceError
from config import (
    GOOGLE_API_KEY, GOOGLE_CX_ID, NEWS_API_KEY, TAVILY_API_KEY,
    ARTICLE_CACHE_PATH, MAX_EVIDENCE_WORDS,
    SENTS_PER_ARTICLE_DEFAULT, MIN_SENT_WORDS
)
from factcheck.providers import SUPPORTED_PROVIDERS, PROVIDER_CONFIG, SearchProvider, QueryType

logger = get_logger(__name__)

# Global cache and models
_ARTICLE_CACHE = None
_EMBED_MODEL = None

# HTTP headers to avoid 403 blocks
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1'
}


# ==========================================================================
# Search Functions
# ==========================================================================

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
        from datetime import datetime, timedelta
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


# build_search_query function removed - now handled by LLM in detect_intent()


# ==========================================================================
# Web Scraping Functions
# ==========================================================================

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
        logger.warning(f"[SCRAPING] Failed to save cache: {e}")


def scrape_article(url: str) -> str:
    """
    Extract main article text from URL.
    
    Args:
        url: Article URL
        
    Returns:
        Extracted article text, or empty string if failed
    """
    # Check cache first
    cache = _get_article_cache()
    if url in cache:
        logger.debug(f"[SCRAPING] Cache hit for: {url}")
        return cache[url]
    
    try:
        response = requests.get(url, timeout=10, headers=HEADERS, allow_redirects=True)
        
        if response.status_code == 403:
            logger.warning(f"[SCRAPING] 403 Forbidden: {url}")
            return ""
        elif response.status_code == 404:
            logger.warning(f"[SCRAPING] 404 Not Found: {url}")
            return ""
        elif response.status_code != 200:
            logger.warning(f"[SCRAPING] HTTP {response.status_code}: {url}")
            return ""
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style", "nav", "header", "footer"]):
            script.decompose()
        
        # Extract all paragraph text
        paragraphs = soup.find_all('p')
        text = ' '.join([p.get_text().strip() for p in paragraphs if p.get_text().strip()])
        
        # Filter out junk
        sentences = text.split('. ')
        clean_sentences = [s for s in sentences if not is_junk(s)]
        text = '. '.join(clean_sentences)
        
        word_count = len(text.split())
        logger.debug(f"[SCRAPING] Extracted {word_count} words from {url}")
        
        # Enforce strict word limit to prevent massive context
        MAX_SCRAPE_WORDS = 5000  # Hard limit per article
        if word_count > MAX_SCRAPE_WORDS:
            logger.warning(f"[SCRAPING] Truncating massive article ({word_count} words) to {MAX_SCRAPE_WORDS} words: {url}")
            words = text.split()[:MAX_SCRAPE_WORDS]
            text = ' '.join(words)
            word_count = MAX_SCRAPE_WORDS
        
        # Cache the result
        if text and word_count > 50:
            cache[url] = text
            _save_article_cache()
        
        return text
        
    except Exception as e:
        logger.error(f"[SCRAPING] Failed to scrape {url}: {e}")
        return ""


def extract_sentences(text: str) -> List[str]:
    """
    Extract sentences from article text.
    
    Args:
        text: Article text
        
    Returns:
        List of sentences (min 5 words each)
    """
    if not text:
        return []
    
    # Simple sentence splitting
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Filter by minimum word count and junk detection
    valid_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if len(sent.split()) >= MIN_SENT_WORDS and not is_junk(sent):
            valid_sentences.append(sent)
    
    return valid_sentences


# ==========================================================================
# Ranking Functions
# ==========================================================================

def _get_embed_model():
    """Get singleton SentenceTransformer model."""
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


def rank_sentences(claim: str, sentences: List[str]) -> List[Tuple[str, float]]:
    """
    Rank sentences by semantic similarity to claim.
    
    Args:
        claim: The claim to check
        sentences: List of candidate sentences
        
    Returns:
        List of (sentence, score) tuples sorted by score (descending)
    """
    if not sentences:
        return []
    
    model = _get_embed_model()
    if not model:
        # Fallback: return sentences with zero scores
        logger.warning("[RANKING] No model available, returning unranked")
        return [(s, 0.0) for s in sentences]
    
    try:
        from sentence_transformers import util
        
        claim_embedding = model.encode(claim, convert_to_tensor=True)
        sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
        
        # Compute cosine similarity
        scores = util.cos_sim(claim_embedding, sentence_embeddings)[0]
        scores = scores.cpu().tolist() if hasattr(scores, "cpu") else list(scores)
        
        # Combine and sort
        ranked = list(zip(sentences, scores))
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        logger.debug(f"[RANKING] Ranked {len(ranked)} sentences")
        return ranked
        
    except Exception as e:
        logger.error(f"[RANKING] Semantic ranking failed: {e}")
        return [(s, 0.0) for s in sentences]


# ==========================================================================
# High-Level Evidence Collection
# ==========================================================================

# Map providers to their implementation functions
PROVIDER_FUNCTIONS = {
    SearchProvider.GOOGLE: search_google,
    SearchProvider.NEWSAPI: search_newsapi,
    SearchProvider.TAVILY: search_tavily
}

def collect_evidence(
    claim: str,
    google_query: str,
    newsapi_query: str,
    num_google: int = 5,
    num_news: int = 5,
    num_tavily: int = 5,
    top_k: int = 10,
    enabled_providers: Optional[List[str]] = None,
    verification_question: Optional[str] = None,
    tavily_answer: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Complete evidence collection pipeline.
    
    1. Search enabled providers (Google, NewsAPI, Tavily) using provider-specific queries
    2. Scrape articles
    3. Extract and rank sentences
    4. Return top-k evidence items
    
    Args:
        claim: The claim to fact-check (for ranking)
        google_query: Optimized search query for Google (from intent detection)
        newsapi_query: Optimized search query for NewsAPI (from intent detection)
        num_google: Number of Google results
        num_news: Number of NewsAPI results
        num_tavily: Number of Tavily results
        top_k: Number of top evidence items to return
        enabled_providers: List of enabled search providers ['google', 'newsapi', 'tavily'].
                          Defaults to all if None. At least one must be enabled.
        verification_question: Optional natural language question for Tavily answer-seeking
        tavily_answer: Optional pre-fetched Tavily answer (if already obtained)
        
    Returns:
        List of evidence dicts with 'text', 'url', 'source', 'score' keys.
        If Tavily is enabled and returns an answer, it's included as a high-priority item.
        
    Raises:
        EvidenceError: If no providers are enabled or all searches fail
    """
    logger.info(f"[EVIDENCE] Collecting evidence for: {claim[:100]}...")
    
    # Validate and normalize enabled providers
    if enabled_providers is None:
        enabled_providers = list(SUPPORTED_PROVIDERS)
    else:
        # Normalize to lowercase
        enabled_providers = [p.lower() for p in enabled_providers if p]
        
        # Validate at least one provider is enabled
        if not enabled_providers:
            raise EvidenceError("At least one search provider must be enabled")
        
        # Validate only known providers
        known_providers = SUPPORTED_PROVIDERS
        invalid = [p for p in enabled_providers if p not in known_providers]
        if invalid:
            logger.warning(f"[EVIDENCE] Unknown providers ignored: {invalid}")
            enabled_providers = [p for p in enabled_providers if p in known_providers]
            
            if not enabled_providers:
                raise EvidenceError(f"No valid search providers specified. Valid options: {known_providers}")
    
    logger.info(f"[EVIDENCE] Enabled providers: {enabled_providers}")
    logger.debug(f"[EVIDENCE] Google query: {google_query[:100]}, NewsAPI query: {newsapi_query[:100]}")
    if verification_question:
        logger.info(f"[EVIDENCE] Verification Question: {verification_question}")
    
    # ----------------------------------------------------------------------
    # Dynamic Search Execution
    # ----------------------------------------------------------------------
    all_search_results = []
    
    # Map limits to providers
    limits = {
        SearchProvider.GOOGLE: num_google,
        SearchProvider.NEWSAPI: num_news,
        SearchProvider.TAVILY: num_tavily
    }
    
    for provider_id in enabled_providers:
        config = PROVIDER_CONFIG.get(provider_id)
        search_func = PROVIDER_FUNCTIONS.get(provider_id)
        
        if not config or not search_func:
            logger.warning(f"[EVIDENCE] No configuration or function for provider: {provider_id}")
            continue
            
        # Determine query based on type
        query = ""
        if config['query_type'] == QueryType.GENERAL:
            query = google_query
        elif config['query_type'] == QueryType.NEWS:
            query = newsapi_query
        elif config['query_type'] == QueryType.VERIFICATION:
            query = verification_question or google_query or claim
        
        if not query:
            logger.warning(f"[EVIDENCE] No query available for provider {provider_id} (type: {config['query_type']})")
            continue
            
        # Determine limit
        limit = limits.get(provider_id, config.get('default_limit', 5))
        
        try:
            logger.info(f"[{provider_id.upper()}] Searching with query: {query[:50]}...")
            results = search_func(query, num_results=limit)
            
            # Special handling for Tavily answer
            if provider_id == SearchProvider.TAVILY and results and results[0].get('answer'):
                logger.info(f"[TAVILY] Got answer: {results[0]['content'][:150]}...")
                
            logger.info(f"[{provider_id.upper()}] Returned {len(results)} results")
            all_search_results.extend(results)
            
        except Exception as e:
            logger.error(f"[{provider_id.upper()}] Search failed: {e}")

    # Separate Tavily answer if present (it's special)
    tavily_answer_item = None
    search_results_for_scraping = []
    
    for item in all_search_results:
        if item.get('answer'):
            tavily_answer_item = item
        else:
            search_results_for_scraping.append(item)
            
    if not search_results_for_scraping and not tavily_answer_item:
        raise EvidenceError("All searches failed or returned no results")

    # ----------------------------------------------------------------------
    # Scrape & Process Articles
    # ----------------------------------------------------------------------
    
    # Deduplicate URLs
    seen_urls = set()
    unique_results = []
    for res in search_results_for_scraping:
        if res['url'] not in seen_urls:
            seen_urls.add(res['url'])
            unique_results.append(res)
    
    logger.info(f"[EVIDENCE] Scraping {len(unique_results)} unique articles...")
    
    sentences_with_meta = []
    
    # Scrape articles
    for res in unique_results:
        try:
            # If result already has content (e.g. Tavily), use it
            if res.get('content'):
                text = res['content']
            else:
                text = scrape_article(res['url'])
            
            if not text:
                continue
                
            # Extract sentences
            article_sentences = extract_sentences(text)
            
            # Store metadata with each sentence
            for sent in article_sentences:
                sentences_with_meta.append({
                    "text": sent,
                    "url": res['url'],
                    "title": res.get('title', 'Unknown Title'),
                    "source": res.get('source', 'Web')
                })
            
        except Exception as e:
            logger.warning(f"[SCRAPING] Failed to process {res['url']}: {e}")
            continue
    
    logger.info(f"[EVIDENCE] Extracted {len(sentences_with_meta)} total sentences")
    
    if not sentences_with_meta and not tavily_answer_item:
        logger.warning("[EVIDENCE] No sentences extracted from any source")
        return []

    # Rank sentences
    sentence_texts = [s["text"] for s in sentences_with_meta]
    ranked_sentences = rank_sentences(claim, sentence_texts)
    
    # Format top-k evidence
    evidence_items = []
    
    # Add Tavily answer as top evidence if available
    if tavily_answer_item:
        evidence_items.append({
            "text": f"AI Analysis: {tavily_answer_item['content']}",
            "url": tavily_answer_item['url'],
            "title": tavily_answer_item.get('title', 'Tavily AI Answer'),
            "source": "Tavily AI",
            "score": 1.0
        })
    
    # Add ranked sentences
    for sent, score in ranked_sentences[:top_k]:
        # Find the metadata for this sentence
        meta = next((s for s in sentences_with_meta if s["text"] == sent), None)
        
        if meta:
            evidence_items.append({
                "text": sent,
                "url": meta["url"],
                "title": meta["title"],
                "source": meta["source"],
                "score": score
            })
        else:
             # Fallback if not found (shouldn't happen)
             evidence_items.append({
                "text": sent,
                "url": "",
                "title": "Unknown Source",
                "source": "Web",
                "score": score
            })
        
    return evidence_items


__all__ = [
    "search_google",
    "search_newsapi",
    "scrape_article",
    "extract_sentences",
    "rank_sentences",
    "collect_evidence",
]
