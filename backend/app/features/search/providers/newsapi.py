from __future__ import annotations

from typing import List, Optional

import httpx

from app.contracts.types import EvidenceSnippet
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.search.providers.base import SearchProvider

logger = get_logger(__name__)


class NewsApiSearchProvider(SearchProvider):
    name = "newsapi"

    def __init__(self, *, settings: Settings):
        self._settings = settings

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        verification_question: Optional[str] = None,
    ) -> List[EvidenceSnippet]:
        api_key = (self._settings.newsapi_api_key or "").strip()
        if not api_key:
            logger.info("[SEARCH:NEWSAPI] Missing NEWSAPI_API_KEY; skipping")
            return []

        params = {
            "q": query,
            "pageSize": int(max_results),
            "language": "en",
            "sortBy": "relevancy",
        }

        try:
            async with httpx.AsyncClient(timeout=20.0) as client:
                resp = await client.get(
                    "https://newsapi.org/v2/everything",
                    params=params,
                    headers={"X-Api-Key": api_key},
                )
                resp.raise_for_status()
                data = resp.json()
        except Exception as exc:
            logger.warning(f"[SEARCH:NEWSAPI] Request failed: {exc}")
            return []

        items: List[EvidenceSnippet] = []
        for a in (data.get("articles") or [])[: int(max_results)]:
            url = (a.get("url") or "").strip()
            title = (a.get("title") or None)
            text = (a.get("description") or a.get("content") or "").strip()
            if not url or not text:
                continue

            items.append(
                EvidenceSnippet(
                    text=text,
                    url=url,
                    title=title,
                    source_domain="newsapi",
                    score=0.5,
                )
            )

        return items
