"""
Search Provider Configuration and Registry.

This module defines the supported search providers and their configuration.
To add a new provider:
1. Add the provider name to the SearchProvider class.
2. Add the provider configuration to PROVIDER_CONFIG.
3. Implement the search function in backend/factcheck/evidence.py and register it in PROVIDER_FUNCTIONS.
"""

class SearchProvider:
    GOOGLE = "google"
    NEWSAPI = "newsapi"
    TAVILY = "tavily"
    # Add new providers here (e.g., BING = "bing")

class QueryType:
    GENERAL = "general"       # Uses google_query (optimized for search engines)
    NEWS = "news"             # Uses newsapi_query (keywords)
    VERIFICATION = "verification" # Uses verification_question (natural language)

# Configuration for providers
# Defines environment variables and which query type to use
PROVIDER_CONFIG = {
    SearchProvider.GOOGLE: {
        "env_var": "GOOGLE_API_KEY",
        "query_type": QueryType.GENERAL,
        "default_limit": 5
    },
    SearchProvider.NEWSAPI: {
        "env_var": "NEWS_API_KEY",
        "query_type": QueryType.NEWS,
        "default_limit": 5
    },
    SearchProvider.TAVILY: {
        "env_var": "TAVILY_API_KEY",
        "query_type": QueryType.VERIFICATION,
        "default_limit": 5
    }
}

# Set of currently supported providers
SUPPORTED_PROVIDERS = set(PROVIDER_CONFIG.keys())

def get_supported_providers():
    """Return a list of supported provider names."""
    return list(SUPPORTED_PROVIDERS)

def validate_providers(providers):
    """
    Validate a list of providers.
    Returns a list of valid providers.
    """
    if not providers:
        return []
    
    return [p for p in providers if p in SUPPORTED_PROVIDERS]
