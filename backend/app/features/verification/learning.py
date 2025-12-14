# Full Path: backend/app/features/verification/learning.py
"""
Continuous Learning (RAG Feedback Loop) with health checks.

When a verification is high-confidence, we asynchronously compute embeddings 
and store them in pgvector columns. This module implements proper health 
validation and circuit breaker protection for the embedding service.
"""
from __future__ import annotations

import asyncio
from typing import Sequence

import httpx
from openai import AsyncOpenAI

from sqlalchemy import select

from app.core.db import get_sessionmaker
from app.core.logging import get_logger
from app.core.settings import Settings
from app.core.circuit_breaker import (
    circuit_breaker,
    CircuitOpenError,
    EMBEDDING_CIRCUIT_CONFIG,
)
from app.features.verification.persistence.models import Claim, Evidence

logger = get_logger(__name__)


def _normalize_vec(vec: Sequence[float]) -> list[float]:
    return [float(x) for x in vec]


class RagLearningService:
    """Continuous learning feedback loop with health validation.

    When a verification is high-confidence, we asynchronously compute embeddings and store
    them in pgvector columns (claims.claim_embedding, evidence.snippet_embedding).
    
    Features:
    - Pre-flight health check for embedding service before expensive operations
    - Circuit breaker protection against embedding service failures
    - Graceful failure handling (logged but never blocks main request)
    """

    def __init__(self, *, settings: Settings):
        self._settings = settings

    def schedule(self, verification_id: int) -> None:
        """Schedule learning task in the background."""
        asyncio.create_task(self.learn_from_verification(verification_id))

    async def learn_from_verification(self, verification_id: int) -> None:
        """Main entry point for learning - catches all exceptions to fail safely."""
        try:
            await self._learn_from_verification(verification_id)
        except CircuitOpenError as exc:
            logger.warning(
                f"[RAG] Learning skipped verification_id={verification_id}: "
                f"Embedding service circuit breaker open (retry after {int(exc.retry_after)}s)"
            )
        except Exception as exc:
            logger.warning(f"[RAG] Learning failed verification_id={verification_id}: {exc}")

    async def _check_embedding_service_health(self) -> bool:
        """
        Quick health check for the embedding service.
        
        Returns True if the service is reachable, False otherwise.
        This prevents wasting time on DB queries if the embedding service is down.
        """
        base_url = (self._settings.embedding_api_base_url or "").strip()
        
        if not base_url:
            return False
        
        health_url = f"{base_url.rstrip('/')}/health"
        
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                response = await client.get(health_url)
                return response.status_code == 200
        except Exception as exc:
            logger.debug(f"[RAG] Embedding service health check failed: {exc}")
            return False

    async def _learn_from_verification(self, verification_id: int) -> None:
        """Internal learning implementation with health validation."""
        api_key = (self._settings.embedding_api_key or self._settings.llm_api_key or "").strip()
        base_url = (self._settings.embedding_api_base_url or self._settings.llm_api_base_url or "").strip()

        if not api_key:
            logger.info("[RAG] Embeddings not configured; skipping")
            return

        model = (self._settings.embedding_model or "").strip()
        if not model:
            logger.info("[RAG] EMBEDDING_MODEL missing; skipping")
            return

        # === PRE-FLIGHT HEALTH CHECK ===
        # Check embedding service before fetching data from DB
        if not await self._check_embedding_service_health():
            logger.warning(
                f"[RAG] Embedding service unreachable at {base_url}; "
                "skipping learning for this verification. "
                "Make sure the Infinity embedding service is running."
            )
            return

        session_maker = get_sessionmaker()

        async with session_maker() as session:
            # Claims needing embeddings
            claim_rows = (
                await session.execute(
                    select(Claim).where(Claim.verification_id == int(verification_id)).order_by(Claim.id.asc())
                )
            ).scalars().all()

            if not claim_rows:
                logger.info(f"[RAG] No claims found for verification_id={verification_id}")
                return

            # Embed claims that are missing
            claim_texts: list[str] = [c.claim_text for c in claim_rows if c.claim_embedding is None]

            # Evidence needing embeddings (limited)
            evidence_rows = (
                await session.execute(
                    select(Evidence).join(Claim, Evidence.claim_id == Claim.id).where(Claim.verification_id == int(verification_id))
                )
            ).scalars().all()

            max_evidence = int(self._settings.learning_max_evidence)
            evidence_rows = [e for e in evidence_rows if e.snippet_embedding is None][:max_evidence]
            evidence_texts: list[str] = [e.snippet for e in evidence_rows]

            if not claim_texts and not evidence_texts:
                logger.info(f"[RAG] Nothing to learn for verification_id={verification_id}")
                return

            updated_claims = 0
            updated_evidence = 0

            # Generate embeddings with circuit breaker protection
            if claim_texts:
                try:
                    vectors = await self._generate_embeddings_with_circuit_breaker(
                        texts=claim_texts,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    if vectors and len(vectors[0]) != int(self._settings.embedding_dim):
                        logger.warning(
                            f"[RAG] Embedding dim mismatch: got={len(vectors[0])} expected={self._settings.embedding_dim}; skipping claims"
                        )
                    else:
                        it = iter([_normalize_vec(v) for v in vectors])
                        for c in claim_rows:
                            if c.claim_embedding is None:
                                c.claim_embedding = next(it)
                                updated_claims += 1
                except Exception as exc:
                    logger.warning(f"[RAG] Failed to generate claim embeddings: {exc}")

            if evidence_texts:
                try:
                    vectors = await self._generate_embeddings_with_circuit_breaker(
                        texts=evidence_texts,
                        model=model,
                        api_key=api_key,
                        base_url=base_url,
                    )
                    if vectors and len(vectors[0]) != int(self._settings.embedding_dim):
                        logger.warning(
                            f"[RAG] Embedding dim mismatch: got={len(vectors[0])} expected={self._settings.embedding_dim}; skipping evidence"
                        )
                    else:
                        for e, vec in zip(evidence_rows, vectors, strict=False):
                            e.snippet_embedding = _normalize_vec(vec)
                            updated_evidence += 1
                except Exception as exc:
                    logger.warning(f"[RAG] Failed to generate evidence embeddings: {exc}")

            if updated_claims or updated_evidence:
                await session.commit()
                logger.info(
                    f"[RAG] Stored embeddings verification_id={verification_id} claims={updated_claims} evidence={updated_evidence}"
                )
            else:
                logger.info(f"[RAG] No embeddings stored for verification_id={verification_id}")

    @circuit_breaker("embedding_service", EMBEDDING_CIRCUIT_CONFIG)
    async def _generate_embeddings_with_circuit_breaker(
        self,
        *,
        texts: list[str],
        model: str,
        api_key: str,
        base_url: str,
    ) -> list[list[float]]:
        """Generate embeddings with circuit breaker protection."""
        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)
        resp = await client.embeddings.create(model=model, input=texts)
        return [d.embedding for d in resp.data]
