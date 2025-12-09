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
    GOOGLE_API_KEY, GOOGLE_CX_ID, NEWS_API_KEY,
    ARTICLE_CACHE_PATH, MAX_EVIDENCE_WORDS,
    SENTS_PER_ARTICLE_DEFAULT, MIN_SENT_WORDS
)

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


def build_search_query(claim: str) -> str:
    """
    Extract key terms from claim for better search.
    
    Uses spaCy if available for entity extraction, otherwise uses regex.
    
    Args:
        claim: The claim text
        
    Returns:
        Optimized search query string
    """
    try:
        import spacy
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            # Model not downloaded, use regex fallback
            logger.warning("[SEARCH] spaCy model 'en_core_web_sm' not found. Run: python -m spacy download en_core_web_sm")
            raise ImportError("spaCy model not found")
        
        doc = nlp(claim)
        
        # Priority 1: Extract key action verbs (cure, cause, contain, track, prevent, etc.)
        action_verbs = [
            tok.text for tok in doc
            if (tok.pos_ == "VERB" and 
                tok.text.lower() in {"cure", "cures", "cause", "causes", "contain", "contains", 
                                     "track", "tracks", "prevent", "prevents", "treat", "treats",
                                     "reduce", "reduces", "increase", "increases"})
        ]
        
        # Priority 2: Extract important nouns (cancer, vaccine, water, microchip, etc.)
        important_nouns = [
            tok.text for tok in doc
            if (tok.pos_ in ("NOUN", "PROPN") and 
                not tok.is_stop and 
                len(tok.text) >= 3 and
                tok.text.lower() not in {"study", "conducted", "published", "research", "found", "showed", "text", "statement", "theory"})
        ]
        
        # Priority 3: Extract named entities (organizations, locations)
        entities = [ent.text for ent in doc.ents if ent.label_ in {
            "PERSON", "ORG", "PRODUCT", "EVENT", "GPE"
        }]
        
        # Combine with priority order: action verbs first, then nouns, then entities
        seen = set()
        terms = []
        for term in action_verbs + important_nouns + entities:
            term_lower = term.lower()
            if term_lower not in seen and len(term_lower) > 2:
                seen.add(term_lower)
                terms.append(term)
        
        if terms:
            return " ".join(terms[:6])  # Limit to 6 unique terms
            
    except (ImportError, OSError):
        # Fallback to regex-based extraction
        logger.info("[SEARCH] spaCy not available, using regex fallback for query building")
        
        # Priority 1: Extract critical medical/scientific keywords
        critical_terms = re.findall(
            r"\b(cancer|vaccine|vaccines|covid|coronavirus|water|cure|cures|curing|treatment|disease|" +
            r"drug|therapy|microchip|microchips|track|tracking|location|thoughts|glasses|contain|contains|" +
            r"cause|causes|prevent|prevents|5g|radiation|autism|dna|mrna|pfizer|moderna|" +
            r"hydroxychloroquine|ivermectin|bleach|disinfectant)\w*\b",
            claim,
            re.IGNORECASE
        )
        
        # Priority 2: Extract proper nouns (organizations, places)
        proper_nouns = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", claim)
        
        # Priority 3: Extract quantitative phrases ("8 glasses", "5G")
        quant_phrases = re.findall(r"\b\d+[A-Za-z]*\s+\w+|\b[0-9]+G\b", claim)
        
        # Combine with priority order
        seen = set()
        terms = []
        for term in critical_terms + proper_nouns[:2] + quant_phrases[:1]:
            term_lower = term.lower().strip()
            if term_lower and term_lower not in seen and len(term_lower) > 1:
                seen.add(term_lower)
                terms.append(term)
        
        if terms:
            return " ".join(terms[:6])
    
    # Fallback: return original claim
    return claim


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

def collect_evidence(
    claim: str,
    num_google: int = 5,
    num_news: int = 5,
    top_k: int = 10
) -> List[Dict[str, Any]]:
    """
    Complete evidence collection pipeline.
    
    1. Build search query from claim
    2. Search Google + NewsAPI
    3. Scrape articles
    4. Extract and rank sentences
    5. Return top-k evidence items
    
    Args:
        claim: The claim to fact-check
        num_google: Number of Google results
        num_news: Number of NewsAPI results
        top_k: Number of top evidence items to return
        
    Returns:
        List of evidence dicts with 'text', 'url', 'source', 'score' keys
    """
    logger.info(f"[EVIDENCE] Collecting evidence for: {claim[:100]}...")
    
    # Build optimized search query
    query = build_search_query(claim)
    logger.debug(f"[EVIDENCE] Search query: {query}")
    
    # Search both sources
    try:
        google_results = search_google(query, num_google)
    except SearchError as e:
        logger.error(f"[EVIDENCE] Google search failed: {e}")
        google_results = []
    
    news_results = search_newsapi(query, num_news)
    
    all_results = google_results + news_results
    logger.info(f"[EVIDENCE] Found {len(all_results)} articles")
    
    # Scrape articles and extract sentences
    all_sentences = []
    sentence_metadata = {}  # sentence -> (url, source, title)
    
    for result in all_results:
        url = result["url"]
        text = scrape_article(url)
        
        if not text:
            continue
        
        sentences = extract_sentences(text)
        
        for sent in sentences[:SENTS_PER_ARTICLE_DEFAULT]:
            all_sentences.append(sent)
            sentence_metadata[sent] = (
                url,
                result["source"],
                result["title"]
            )
    
    logger.info(f"[EVIDENCE] Extracted {len(all_sentences)} candidate sentences")
    
    # Rank sentences by relevance
    ranked = rank_sentences(claim, all_sentences)
    
    # Build evidence list
    evidence = []
    for sent, score in ranked[:top_k]:
        if sent in sentence_metadata:
            url, source, title = sentence_metadata[sent]
            evidence.append({
                "text": sent,
                "url": url,
                "source": source,
                "title": title,
                "score": float(score)
            })
    
    logger.info(f"[EVIDENCE] Returning {len(evidence)} evidence items")
    return evidence


__all__ = [
    "search_google",
    "search_newsapi",
    "build_search_query",
    "scrape_article",
    "extract_sentences",
    "rank_sentences",
    "collect_evidence",
]
