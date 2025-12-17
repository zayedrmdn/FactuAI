from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
from importlib import import_module
from typing import Any, List, Optional

import httpx
from openai import AsyncOpenAI
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.types import EvidenceSnippet
from app.core.db import get_sessionmaker
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
    """Native async search service with RAG support.

    Features:
    - External search via pluggable providers (Tavily, NewsAPI, etc.)
    - Internal RAG search via pgvector for similar claims/evidence
    - Parallel execution for minimal latency
    - Threshold-based filtering for relevance

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
        """Perform hybrid search across external providers and internal RAG store.

        Executes external and internal searches in parallel for minimal latency.
        Merges and deduplicates results, with internal RAG filtered by threshold.
        """
        query_clean = (query or "").strip()
        if not query_clean:
            return []

        enabled = set([p.strip().lower() for p in (providers or []) if p and p.strip()]) if providers else None

        # Check cache first
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

        # Run external and internal searches in parallel
        external_task = self._search_external(
            query=query_clean,
            max_results=max_results,
            enabled_providers=enabled,
            verification_question=verification_question,
        )
        internal_task = self._search_internal(query=query_clean, max_results=max_results)

        external_results, internal_results = await asyncio.gather(
            external_task, internal_task, return_exceptions=True
        )

        # Handle exceptions gracefully
        merged: list[EvidenceSnippet] = []

        if isinstance(external_results, Exception):
            logger.warning(f"[SEARCH] External search failed: {external_results}")
        else:
            merged.extend(external_results)

        if isinstance(internal_results, Exception):
            logger.warning(f"[SEARCH] Internal RAG search failed: {internal_results}")
        else:
            merged.extend(internal_results)

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

        # Cache the merged results
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

    async def _search_external(
        self,
        *,
        query: str,
        max_results: int,
        enabled_providers: Optional[set[str]],
        verification_question: Optional[str],
    ) -> List[EvidenceSnippet]:
        """Search external providers (Tavily, NewsAPI, etc.)."""
        active = [
            p for p in self._providers
            if enabled_providers is None or getattr(p, "name", "").lower() in enabled_providers
        ]
        if not active:
            logger.info("[SEARCH] No active external providers")
            return []

        results = await asyncio.gather(
            *[
                p.search(query=query, max_results=int(max_results), verification_question=verification_question)
                for p in active
            ],
            return_exceptions=True,
        )

        merged: list[EvidenceSnippet] = []
        for r in results:
            if isinstance(r, Exception):
                logger.warning(f"[SEARCH] External provider failed: {r}")
                continue
            merged.extend(r)

        return merged

    async def _search_internal(
        self,
        *,
        query: str,
        max_results: int,
    ) -> List[EvidenceSnippet]:
        """Search internal RAG store (claims + evidence with embeddings).

        Uses pgvector cosine distance operator (<=>).
        Only returns results below the configured distance threshold.
        Fails gracefully on any error (returns empty list).
        """
        api_key = (self._settings.embedding_api_key or self._settings.llm_api_key or "").strip()
        base_url = (self._settings.embedding_api_base_url or self._settings.llm_api_base_url or "").strip()
        model = (self._settings.embedding_model or "").strip()
        threshold = float(self._settings.rag_retrieval_threshold)

        # Pre-flight: embedding service must be configured
        if not api_key or not model or not base_url:
            logger.debug("[RAG] Internal search skipped: embedding service not configured")
            return []

        # Pre-flight: health check
        try:
            health_url = f"{base_url.rstrip('/')}/health"
            async with httpx.AsyncClient(timeout=2.0) as client:
                resp = await client.get(health_url)
                if resp.status_code != 200:
                    logger.debug("[RAG] Embedding service health check failed")
                    return []
        except Exception as exc:
            logger.debug(f"[RAG] Embedding service unreachable: {exc}")
            return []

        # Generate query embedding
        try:
            oai_client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
            embed_resp = await oai_client.embeddings.create(model=model, input=[query])
            query_embedding = embed_resp.data[0].embedding
        except Exception as exc:
            logger.warning(f"[RAG] Failed to generate query embedding: {exc}")
            return []

        # Query pgvector for similar claims
        results: list[EvidenceSnippet] = []
        try:
            session_maker = get_sessionmaker()
            async with session_maker() as session:
                results.extend(await self._query_claims(session, query_embedding, threshold, max_results))
                results.extend(await self._query_evidence(session, query_embedding, threshold, max_results))
        except Exception as exc:
            logger.warning(f"[RAG] DB query failed: {exc}")
            return []

        logger.info(f"[RAG] Internal search returned {len(results)} results (threshold={threshold})")
        return results

    async def _query_claims(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        threshold: float,
        max_results: int,
    ) -> list[EvidenceSnippet]:
        """Query claims table for similar embeddings."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = text("""
            SELECT
                c.claim_text,
                c.reasoning,
                c.claim_embedding <=> :embedding AS distance,
                v.input_text
            FROM claims c
            JOIN verifications v ON c.verification_id = v.id
            WHERE c.claim_embedding IS NOT NULL
              AND c.claim_embedding <=> :embedding < :threshold
            ORDER BY distance ASC
            LIMIT :limit
        """)

        result = await session.execute(
            sql,
            {"embedding": embedding_str, "threshold": threshold, "limit": max_results}
        )
        rows = result.fetchall()

        snippets: list[EvidenceSnippet] = []
        for row in rows:
            distance = float(row.distance)
            similarity = 1.0 - distance  # Convert distance to similarity score
            # Truncate reasoning to 500 chars to prevent token bloat
            reasoning_text = (row.reasoning or row.claim_text or "")[:500]
            snippets.append(
                EvidenceSnippet(
                    title=f"[INTERNAL MEMORY] {row.claim_text[:80]}...",
                    text=reasoning_text,
                    url=f"internal://claim/{hash(row.claim_text) & 0xFFFFFFFF}",
                    source_domain="internal_memory",
                    score=similarity,
                )
            )
        return snippets

    async def _query_evidence(
        self,
        session: AsyncSession,
        query_embedding: list[float],
        threshold: float,
        max_results: int,
    ) -> list[EvidenceSnippet]:
        """Query evidence table for similar embeddings."""
        embedding_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

        sql = text("""
            SELECT
                e.snippet,
                e.relevance_score,
                e.snippet_embedding <=> :embedding AS distance,
                s.url,
                s.title,
                s.domain
            FROM evidence e
            JOIN sources s ON e.source_id = s.id
            WHERE e.snippet_embedding IS NOT NULL
              AND e.snippet_embedding <=> :embedding < :threshold
            ORDER BY distance ASC
            LIMIT :limit
        """)

        result = await session.execute(
            sql,
            {"embedding": embedding_str, "threshold": threshold, "limit": max_results}
        )
        rows = result.fetchall()

        snippets: list[EvidenceSnippet] = []
        for row in rows:
            distance = float(row.distance)
            similarity = 1.0 - distance
            snippets.append(
                EvidenceSnippet(
                    title=f"[INTERNAL MEMORY] {row.title or row.domain}",
                    text=row.snippet,
                    url=row.url,
                    source_domain="internal_memory",
                    score=similarity,
                )
            )
        return snippets
