from __future__ import annotations

import asyncio
import time
import uuid
from typing import List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.types import EvidenceSnippet
from app.core.container import Container
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.analyze._utils import extract_domain, map_verdict, normalize_url, select_model
from app.features.analyze.prompts import (
    MultiAngleQueries,
    QUERY_GENERATION_SYSTEM,
    QUERY_GENERATION_HUMAN,
    format_evidence_for_verification,
)
from app.features.analyze.schemas import AnalyzeRequest
from app.features.verification.domain import ClaimAnalysis, Evidence
from app.features.verification.learning import RagLearningService
from app.features.verification.persistence.repository import SqlAlchemyVerificationRepository

logger = get_logger(__name__)


class AnalyzeService:
    """Strategist Pipeline for robust claim verification.
    
    Features:
    - Multi-angle query generation (Factual, Hoax, Scientific)
    - Parallel search execution via asyncio.gather
    - URL deduplication for merged results
    - Rich context consumption (ai_overview, content)
    """

    def __init__(
        self,
        *,
        settings: Settings,
        container: Container,
        db: AsyncSession,
        redis: Optional[Redis],
    ):
        self._settings = settings
        self._container = container
        self._db = db
        self._redis = redis

    async def analyze(self, request: AnalyzeRequest) -> tuple[uuid.UUID, str, int, list[ClaimAnalysis]]:
        start = time.perf_counter()

        provider = request.provider or self._settings.llm_provider
        model = select_model(provider, openrouter_model=self._settings.openrouter_model)

        intent = self._container.intent()
        search = self._container.search()
        verifier = self._container.verifier()

        intent_items = await intent.parse_and_route(
            text=request.text,
            max_claims=request.max_claims,
            provider=provider,
            model=model,
        )
        if not intent_items:
            raise ValueError("No claims extracted from input")

        claim_results: list[ClaimAnalysis] = []

        for item in intent_items:
            claim_text = (item.get("claim_text") or "").strip()
            if not claim_text:
                continue

            # === STRATEGIST: Multi-Angle Query Generation ===
            queries = await self._generate_multi_queries(
                claim=claim_text,
                model=model,
            )
            logger.info(f"[ANALYZE] Generated queries: {queries}")

            # === STRATEGIST: Parallel Search Execution ===
            evidence_snippets: list[EvidenceSnippet] = []
            if request.enable_web_search and queries:
                evidence_snippets = await self._search_parallel(
                    queries=queries,
                    search=search,
                    max_results_per_query=5,
                )

            verdict_data = await verifier.verify_claim(
                claim=claim_text,
                evidence=evidence_snippets,
                provider=provider,
                model=model,
            )

            evidence: list[Evidence] = []
            for ev in verdict_data.get("evidence", []):
                url = normalize_url(ev.get("url", ""), "https://example.com/evidence")
                evidence.append(
                    Evidence(
                        snippet=ev.get("text", ""),
                        source_url=url,
                        source_title=ev.get("title"),
                        source_domain=ev.get("source_domain") or ev.get("source") or extract_domain(url),
                        relevance_score=float(ev.get("score", 0.0)),
                    )
                )

            claim_results.append(
                ClaimAnalysis(
                    claim_text=claim_text,
                    verdict=map_verdict(verdict_data.get("verdict")),
                    confidence=float(verdict_data.get("confidence", 0.0)),
                    reasoning=verdict_data.get("reasoning", ""),
                    evidence=evidence,
                )
            )

        request_id = uuid.uuid4()
        latency_ms = int((time.perf_counter() - start) * 1000)

        verification_id: int | None = None
        try:
            repo = SqlAlchemyVerificationRepository(self._db)
            verification_id = await repo.persist(
                request_id=request_id,
                input_text=request.text,
                model_used=model,
                latency_ms=latency_ms,
                claims=claim_results,
            )
        except Exception as exc:
            if self._settings.db_required:
                raise
            logger.info(f"[DB] Persist skipped (DB unavailable): {exc}")

        # Continuous Learning: asynchronously embed/store high-confidence verifications.
        try:
            if verification_id is not None:
                best_confidence = max((float(c.confidence) for c in claim_results), default=0.0)
                if best_confidence >= float(self._settings.learning_confidence_threshold):
                    learner = RagLearningService(settings=self._settings)
                    learner.schedule(verification_id)
        except Exception:
            logger.warning("[RAG] Failed to schedule learning")

        if self._redis is not None and verification_id is not None:
            logger.info(f"[ANALYZE] Persisted verification_id={verification_id}")

        return request_id, model, latency_ms, claim_results

    async def _generate_multi_queries(
        self,
        *,
        claim: str,
        model: str,
    ) -> List[str]:
        """Generate 3 strategic multi-angle search queries using LLM.
        
        Returns list of queries: [factual, hoax, scientific]
        Falls back to claim as single query on failure.
        """
        api_key = (self._settings.llm_api_key or "").strip()
        base_url = (self._settings.llm_api_base_url or "").strip()

        if not api_key:
            logger.warning("[ANALYZE] No LLM key; falling back to single query")
            return [claim]

        try:
            prompt = ChatPromptTemplate.from_messages([
                ("system", QUERY_GENERATION_SYSTEM),
                ("human", QUERY_GENERATION_HUMAN),
            ])

            try:
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.3,
                    api_key=api_key,
                    base_url=base_url or None,
                )
            except TypeError:
                llm = ChatOpenAI(
                    model=model,
                    temperature=0.3,
                    openai_api_key=api_key,
                    openai_api_base=base_url or None,
                )

            structured_llm = llm.with_structured_output(MultiAngleQueries)
            chain = prompt | structured_llm

            result: MultiAngleQueries = await chain.ainvoke({"claim": claim})

            queries = [
                result.factual_query.strip(),
                result.hoax_query.strip(),
                result.scientific_query.strip(),
            ]
            # Filter empty queries
            queries = [q for q in queries if q]

            if queries:
                return queries

        except Exception as exc:
            logger.warning(f"[ANALYZE] Query generation failed: {exc}")

        # Fallback to claim itself
        return [claim]

    async def _search_parallel(
        self,
        *,
        queries: List[str],
        search,
        max_results_per_query: int = 5,
    ) -> List[EvidenceSnippet]:
        """Execute multiple searches in parallel and deduplicate results.
        
        Uses asyncio.gather for concurrent execution.
        Deduplicates by URL, keeping highest-scoring result.
        """
        search_start = time.perf_counter()

        # Execute all searches in parallel
        search_tasks = [
            search.hybrid_search(
                query=q,
                max_results=max_results_per_query,
                providers=None,
                verification_question=None,
            )
            for q in queries
        ]

        results = await asyncio.gather(*search_tasks, return_exceptions=True)

        # Merge results
        all_snippets: List[EvidenceSnippet] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"[ANALYZE] Search query {i+1} failed: {result}")
                continue
            all_snippets.extend(result)

        # Deduplicate by URL (keep highest score)
        best_by_url: dict[str, EvidenceSnippet] = {}
        for snippet in all_snippets:
            url = (snippet.get("url") or "").strip()
            if not url:
                continue
            existing = best_by_url.get(url)
            if existing is None or float(snippet.get("score", 0.0)) > float(existing.get("score", 0.0)):
                best_by_url[url] = snippet

        # Sort by score descending
        deduplicated = sorted(
            best_by_url.values(),
            key=lambda x: float(x.get("score", 0.0)),
            reverse=True,
        )

        search_time_ms = int((time.perf_counter() - search_start) * 1000)
        logger.info(
            f"[ANALYZE] Parallel search completed in {search_time_ms}ms "
            f"({len(queries)} queries, {len(all_snippets)} total, {len(deduplicated)} unique)"
        )

        return deduplicated

