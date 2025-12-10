"""
Search module for FactuAI.

Provides search functionality across multiple providers:
- Google Custom Search
- NewsAPI
- Tavily
"""

from search.base import collect_evidence
from search.google import search_google
from search.newsapi import search_newsapi
from search.tavily import search_tavily
from search.config import (
    SearchProvider,
    QueryType,
    SUPPORTED_PROVIDERS,
    PROVIDER_CONFIG,
    get_supported_providers,
    validate_providers
)

__all__ = [
    "collect_evidence",
    "search_google",
    "search_newsapi",
    "search_tavily",
    "SearchProvider",
    "QueryType",
    "SUPPORTED_PROVIDERS",
    "PROVIDER_CONFIG",
    "get_supported_providers",
    "validate_providers",
]
