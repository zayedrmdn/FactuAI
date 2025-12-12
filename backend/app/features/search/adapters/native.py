from __future__ import annotations

import hashlib
import inspect
import json
from importlib import import_module
from typing import Any, List, Optional

from redis.asyncio import Redis

from app.contracts.types import EvidenceSnippet
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.search.ports import SearchPort
from app.features.search.providers.base import SearchProvider

logger = get_logger(__name__)


def _load_symbol(dotted_path: str) -> Any:
    module_path, _, symbol_name = dotted_path.rpartition(".")
    if not module_path:
        raise ValueError(f"Invalid dotted path: {dotted_path}")
    module = import_module(module_path)
    try:
        return getattr(module, symbol_name)
    except AttributeError as exc:
        raise ValueError(f"Symbol not found: {dotted_path}") from exc


def _instantiate_provider(dotted_path: str, *, settings: Settings) -> SearchProvider:
    cls = _load_symbol(dotted_path)
    if not callable(cls):
        raise ValueError(f"Provider is not callable: {dotted_path}")

    try:
        sig = inspect.signature(cls)
    except (TypeError, ValueError):
        return cls()

    kwargs: dict[str, Any] = {}
    for name in sig.parameters.keys():
        if name == "settings":
            kwargs[name] = settings

    return cls(**kwargs)


def _parse_paths(csv: str) -> list[str]:
    return [p.strip() for p in (csv or "").split(",") if p and p.strip()]


class NativeSearchService(SearchPort):
    """Native async search service (no legacy scripts).

    OCP: add a provider by adding a new provider class and updating SEARCH_PROVIDER_PATHS.
    """

    def __init__(self, *, settings: Settings, redis: Optional[Redis] = None):
        self._settings = settings
        self._redis = redis

        self._providers: list[SearchProvider] = []
        for path in _parse_paths(settings.search_provider_paths_csv):
            try:
                provider = _instantiate_provider(path, settings=settings)
                self._providers.append(provider)
            except Exception as exc:
                logger.warning(f"[SEARCH] Failed to load provider {path}: {exc}")

        logger.info(f"[SEARCH] Native providers: {[getattr(p, 'name', '?') for p in self._providers]}")

    async def hybrid_search(
        self,
        *,
        query: str,
        max_results: int = 8,
        providers: Optional[List[str]] = None,
        verification_question: Optional[str] = None,
    ) -> List[EvidenceSnippet]:
        query_clean = (query or "").strip()
        if not query_clean:
            return []

        enabled = set([p.strip().lower() for p in (providers or []) if p and p.strip()]) if providers else None

        cache_key = None
        if self._redis is not None:
            material = json.dumps(
                {
                    "query": query_clean,
                    "max_results": int(max_results),
                    "providers": sorted(list(enabled)) if enabled else None,
                    "verification_question": verification_question,
                },
                sort_keys=True,
            ).encode("utf-8")
            cache_key = f"search:evidence:{hashlib.sha256(material).hexdigest()}"

            cached = await self._redis.get(cache_key)
            if cached:
                try:
                    payload = json.loads(cached)
                    return [EvidenceSnippet(**item) for item in payload]
                except Exception:
                    logger.warning("[SEARCH] Failed to parse cached evidence")

        active = [p for p in self._providers if enabled is None or getattr(p, "name", "").lower() in enabled]
        if not active:
            logger.info("[SEARCH] No active providers; returning empty evidence")
            return []

        # Run providers concurrently
        import asyncio

        results = await asyncio.gather(
            *[
                p.search(query=query_clean, max_results=int(max_results), verification_question=verification_question)
                for p in active
            ],
            return_exceptions=True,
        )

        merged: list[EvidenceSnippet] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[SEARCH] Provider failed: {r}")
                continue
            merged.extend(r)

        # De-dupe by URL (best score wins)
        best_by_url: dict[str, EvidenceSnippet] = {}
        for item in merged:
            url = (item.get("url") or "").strip()
            if not url:
                continue
            current = best_by_url.get(url)
            if current is None or float(item.get("score", 0.0)) > float(current.get("score", 0.0)):
                best_by_url[url] = item

        final = sorted(best_by_url.values(), key=lambda x: float(x.get("score", 0.0)), reverse=True)[: int(max_results)]

        if cache_key is not None and self._redis is not None:
            try:
                await self._redis.setex(
                    cache_key,
                    int(self._settings.evidence_cache_ttl_seconds),
                    json.dumps([dict(item) for item in final]),
                )
            except Exception:
                logger.warning("[SEARCH] Failed to write evidence cache")

        return final
