# Full path: backend/app/features/search/providers/tavily.py
"""
Tavily Search Provider with circuit breaker protection.

Production-Grade Configuration:
- Auto-parameters enabled for query optimization
- Social media domains excluded for noise reduction
- Answer and raw content included for richer data
- Images/favicons/usage disabled to save bandwidth
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from app.contracts.types import EvidenceSnippet
from app.core.constants import SOCIAL_MEDIA_DOMAINS
from app.core.logging import get_logger
from app.core.settings import Settings
from app.core.circuit_breaker import (
    circuit_breaker,
    CircuitOpenError,
    SEARCH_CIRCUIT_CONFIG,
)
from app.features.search.providers.base import SearchProvider

logger = get_logger(__name__)


class TavilySearchProvider(SearchProvider):
    """Tavily Search API provider with circuit breaker protection.
    
    Production-grade filtering:
    - Excludes social media domains (SOCIAL_MEDIA_DOMAINS)
    - Uses auto_parameters for query optimization
    - Includes Tavily's AI answer summary
    """

    name = "tavily"

    def __init__(self, *, settings: Settings):
        self._settings = settings

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        verification_question: Optional[str] = None,
    ) -> List[EvidenceSnippet]:
        api_key = (self._settings.tavily_api_key or "").strip()
        if not api_key:
            logger.info("[SEARCH:TAVILY] Missing TAVILY_API_KEY; skipping")
            return []

        try:
            return await self._search_with_circuit_breaker(
                query=query,
                max_results=max_results,
                api_key=api_key,
            )
        except CircuitOpenError as exc:
            logger.warning(f"[SEARCH:TAVILY] Circuit breaker open: {exc}")
            return []  # Graceful degradation - return empty results

    @circuit_breaker("search_tavily", SEARCH_CIRCUIT_CONFIG)
    async def _search_with_circuit_breaker(
        self,
        *,
        query: str,
        max_results: int,
        api_key: str,
    ) -> List[EvidenceSnippet]:
        """Internal search method wrapped with circuit breaker.
        
        Production-grade Tavily payload configuration.
        """
        payload = {
            "api_key": api_key,
            "query": query,
            "auto_parameters": True,
            "search_depth": "basic",
            "max_results": min(int(max_results), 5),
            "include_answer": True,
            "include_raw_content": True,
            "include_images": False,
            "include_image_descriptions": False,
            "include_favicon": False,
            "include_usage": False,
            "exclude_domains": SOCIAL_MEDIA_DOMAINS,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

        # Extract Tavily's AI summary (answer field)
        ai_overview = (data.get("answer") or "").strip() or None

        results = []
        for r in (data.get("results") or [])[: int(max_results)]:
            results.append(
                EvidenceSnippet(
                    text=(r.get("content") or "").strip(),
                    url=(r.get("url") or "").strip(),
                    title=(r.get("title") or None),
                    source_domain="tavily",
                    score=float(r.get("score") or 0.0),
                    ai_overview=ai_overview,
                    content=(r.get("raw_content") or "").strip() or None,
                )
            )

        return [r for r in results if r.get("url") and r.get("text")]

