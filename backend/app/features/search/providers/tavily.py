from __future__ import annotations

from typing import List, Optional

import httpx

from app.contracts.types import EvidenceSnippet
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.search.providers.base import SearchProvider

logger = get_logger(__name__)


class TavilySearchProvider(SearchProvider):
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

        payload = {
            "api_key": api_key,
            "query": query,
            "max_results": int(max_results),
            "search_depth": "basic",
            "include_answer": False,
            "include_raw_content": False,
            "include_images": False,
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.post("https://api.tavily.com/search", json=payload)
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning(f"[SEARCH:TAVILY] Request failed: {exc}")
            return []

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
