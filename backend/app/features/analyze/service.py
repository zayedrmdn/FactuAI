from __future__ import annotations

import time
import uuid
from typing import Optional

from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.contracts.types import EvidenceSnippet
from app.core.container import Container
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.analyze._utils import extract_domain, map_verdict, normalize_url, select_model
from app.features.analyze.schemas import AnalyzeRequest
from app.features.verification.domain import ClaimAnalysis, Evidence
from app.features.verification.learning import RagLearningService
from app.features.verification.persistence.repository import SqlAlchemyVerificationRepository

logger = get_logger(__name__)


class AnalyzeService:
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

            search_query = (item.get("search_query") or claim_text).strip()
            verification_question = item.get("verification_question")

            evidence_snippets: list[EvidenceSnippet] = []
            if request.enable_web_search:
                evidence_snippets = await search.hybrid_search(
                    query=search_query,
                    max_results=8,
                    providers=None,
                    verification_question=verification_question,
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
