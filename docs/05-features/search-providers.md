# Adding Search Providers (OCP Pattern)

How to extend FactuAI with new search providers without modifying existing code.

---

## Overview

FactuAI uses the **Open/Closed Principle (OCP)**: the system is **open for extension** but **closed for modification**.

**Goal:** Add new search providers by:
1. Creating a new provider class
2. Adding its path to configuration
3. **No orchestrator changes needed** ✅

---

## Provider Interface

### Port Definition

**Location:** `backend/app/features/search/providers/base.py`

```python
from abc import ABC, abstractmethod
from typing import List

class SearchProvider(ABC):
    @abstractmethod
    async def search(self, query: str) -> List[SearchResult]:
        """Execute search and return results"""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name for logging"""
        pass
```

---

## Step-by-Step: Create a Custom Provider

### Step 1: Create Provider Class

**Location:** `backend/app/features/search/providers/custom_provider.py`

```python
from typing import List
import httpx
from app.features.search.providers.base import SearchProvider
from app.contracts.types import EvidenceSnippet

class CustomSearchProvider(SearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = httpx.AsyncClient()
    
    @property
    def name(self) -> str:
        return "CustomSearch"
    
    async def search(self, query: str) -> List[SearchResult]:
        """
        Execute search via your custom API
        """
        try:
            response = await self.client.get(
                "https://api.customsearch.com/search",
                params={"q": query, "key": self.api_key}
            )
            response.raise_for_status()
            data = response.json()
            
            # Map to SearchResult schema
            results = []
            for item in data.get("results", []):
                results.append(EvidenceSnippet(
                    url=item["url"],
                    title=item["title"],
                    snippet=item["snippet"],
                    relevance_score=item.get("score", 0.5)
                ))
            
            return results
            
        except Exception as e:
            # Log error but don't crash
            logger.error(f"CustomSearch failed: {e}")
            return []  # Return empty list on failure
    
    async def close(self):
        """Cleanup (called on shutdown)"""
        await self.client.aclose()
```

### Step 2: Register in Settings

**Location:** `backend/app/core/settings.py`

```python
class Settings(BaseSettings):
    # Existing providers
    search_provider_paths: str = "backend.app.features.search.providers.tavily.TavilyProvider"
    
    # Add your custom API key
    custom_search_api_key: str = Field(default="", env="CUSTOM_SEARCH_API_KEY")
```

### Step 3: Add to Environment

**File:** `backend/.env`

```bash
# Add custom provider to comma-separated list
SEARCH_PROVIDER_PATHS=app.features.search.providers.tavily.TavilySearchProvider,app.features.search.providers.custom_provider.CustomSearchProvider

# Add API key
CUSTOM_SEARCH_API_KEY=your_api_key_here
```

### Step 4: Update Container (DI)

**Location:** `backend/app/core/container.py`

```python
# Container automatically loads providers from SEARCH_PROVIDER_PATHS
# No changes needed if provider follows interface!
```

### Step 5: Restart Backend

```bash
uvicorn app.main:app --reload
```

**That's it!** Your provider is now integrated.

---

## How It Works Internally

### Dynamic Loading

```python
# backend/app/core/container.py
def load_search_providers():
    provider_paths = settings.search_provider_paths.split(",")
    providers = []
    
    for path in provider_paths:
        # Dynamically import class
        module_path, class_name = path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        ProviderClass = getattr(module, class_name)
        
        # Instantiate with API key from settings
        provider = ProviderClass(api_key=get_api_key_for_provider(ProviderClass))
        providers.append(provider)
    
    return providers
```

### Orchestrator Usage

```python
# backend/app/features/search/adapters/native.py
async def search(self, query: str):
    # Get all registered providers
    providers = container.get_search_providers()
    
    # Execute in parallel
    tasks = [provider.search(query) for provider in providers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Merge results
    return merge_results(results)
```

---

## Example Providers

### 1. SerpAPI Provider

```python
class SerpAPIProvider(SearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
    
    async def search(self, query: str) -> List[SearchResult]:
        # Implement SerpAPI integration
        pass
```

**Configuration:**
```bash
SEARCH_PROVIDER_PATHS=...,app.features.search.providers.serpapi.SerpAPIProvider
SERPAPI_API_KEY=your_key
```

### 2. Bing Search Provider

```python
class BingSearchProvider(SearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.endpoint = "https://api.bing.microsoft.com/v7.0/search"
    
    async def search(self, query: str) -> List[SearchResult]:
        # Implement Bing Search API integration
        pass
```

---

## Best Practices

### 1. Fail Gracefully

```python
async def search(self, query: str) -> List[SearchResult]:
    try:
        # ... search logic
    except Exception as e:
        logger.error(f"{self.name} search failed: {e}")
        return []  # Don't crash entire pipeline
```

### 2. Respect Rate Limits

```python
import asyncio
from datetime import datetime

class RateLimitedProvider(SearchProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.last_request = None
        self.min_interval = 1.0  # seconds
    
    async def search(self, query: str):
        # Wait if needed
        if self.last_request:
            elapsed = (datetime.now() - self.last_request).total_seconds()
            if elapsed < self.min_interval:
                await asyncio.sleep(self.min_interval - elapsed)
        
        self.last_request = datetime.now()
        # ... actual search
```

### 3. Cache Results (Optional)

```python
from functools import lru_cache

class CachedProvider(SearchProvider):
    @lru_cache(maxsize=100)
    async def search(self, query: str):
        # Results cached for same query
        pass
```

---

## Testing Your Provider

```python
# backend/tests/test_custom_provider.py
import pytest
from backend.app.features.search.providers.custom_provider import CustomSearchProvider

@pytest.mark.asyncio
async def test_custom_provider_search():
    provider = CustomSearchProvider(api_key="test_key")
    results = await provider.search("test query")
    
    assert isinstance(results, list)
    assert all(isinstance(r, SearchResult) for r in results)
```

---

## Removing a Provider

**To disable a provider:**

```bash
# Remove from comma-separated list
SEARCH_PROVIDER_PATHS=app.features.search.providers.tavily.TavilySearchProvider
# (CustomProvider removed)
```

**Restart backend.** Provider is no longer loaded.

---

## Code Pointers

- Port interface: `backend/app/features/search/providers/base.py`
- Tavily example: `backend/app/features/search/providers/tavily.py`
- Container loading: `backend/app/core/container.py`
- Settings: `backend/app/core/settings.py`

---

See also:
- [Backend Architecture](../03-architecture/backend.md) - DI & OCP patterns
- [Phase 2: Parallel Search](../04-pipeline/03-search.md) - How search works
- [Constitution](../01-rules/constitution.md) - OCP principle
