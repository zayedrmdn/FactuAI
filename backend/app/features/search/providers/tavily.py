# Full path: backend/app/features/search/providers/tavily.py
"""
Tavily Search Provider with circuit breaker protection.

This provider uses the Tavily Search API for web search.
Circuit breaker protects against cascading failures when the API is experiencing issues.
"""
from __future__ import annotations

from typing import List, Optional

import httpx

from app.contracts.types import EvidenceSnippet
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
    """Tavily Search API provider with circuit breaker protection."""
    
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
        """Internal search method wrapped with circuit breaker."""
        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": int(max_results),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        async with httpx.AsyncClient(timeout=20.0) as client:
            resp = await client.post("https://api.tavily.com/search", json=payload)
            resp.raise_for_status()
            data = resp.json()

        results = []
        for r in (data.get("results") or [])[: int(max_results)]:
            results.append(
                EvidenceSnippet(
                    text=(r.get("content") or "").strip(),
                    url=(r.get("url") or "").strip(),
                    title=(r.get("title") or None),
                    source_domain="tavily",
                    score=float(r.get("score") or 0.0),
                )
            )

        return [r for r in results if r.get("url") and r.get("text")]
