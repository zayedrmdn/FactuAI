"""
Base search orchestration module.

Coordinates multiple search providers and integrates with extraction and ranking services.
"""

from typing import List, Dict, Any, Optional

from utils.logging import get_logger
from utils.helpers import EvidenceError
from search.config import SearchProvider, QueryType, PROVIDER_CONFIG, SUPPORTED_PROVIDERS
from search.google import search_google
from search.newsapi import search_newsapi
from search.tavily import search_tavily
from extract.scraper import scrape_article
from extract.base import extract_sentences
from services.ranking.scorer import rank_sentences
from config import EVIDENCE_DEFAULT_COUNT

logger = get_logger(__name__)

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
    top_k: int = EVIDENCE_DEFAULT_COUNT,
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
    selected_sentences = ranked_sentences[:top_k]
    logger.info(f"[EVIDENCE] Selected top {len(selected_sentences)}/{len(ranked_sentences)} ranked sentences")
    
    for sent, score in selected_sentences:
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


__all__ = ["collect_evidence"]
