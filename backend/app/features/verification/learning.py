# Full Path: backend\app\features\verification\learning.py 
from __future__ import annotations

import asyncio
from typing import Iterable, Sequence

from openai import AsyncOpenAI
from sqlalchemy import select

from app.core.db import get_sessionmaker
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.verification.persistence.models import Claim, Evidence

logger = get_logger(__name__)


def _normalize_vec(vec: Sequence[float]) -> list[float]:
    return [float(x) for x in vec]


class RagLearningService:
    """Continuous learning feedback loop.

    When a verification is high-confidence, we asynchronously compute embeddings and store
    them in pgvector columns (claims.claim_embedding, evidence.snippet_embedding).
    """

    def __init__(self, *, settings: Settings):
        self._settings = settings

    def schedule(self, verification_id: int) -> None:
        asyncio.create_task(self.learn_from_verification(verification_id))

    async def learn_from_verification(self, verification_id: int) -> None:
        try:
            await self._learn_from_verification(verification_id)
        except Exception as exc:
            logger.warning(f"[RAG] Learning failed verification_id={verification_id}: {exc}")

    async def _learn_from_verification(self, verification_id: int) -> None:
        api_key = (self._settings.embedding_api_key or self._settings.llm_api_key or "").strip()
        base_url = (self._settings.embedding_api_base_url or self._settings.llm_api_base_url or "").strip()

        if not api_key:
            logger.info("[RAG] Embeddings not configured; skipping")
            return

        model = (self._settings.embedding_model or "").strip()
        if not model:
            logger.info("[RAG] EMBEDDING_MODEL missing; skipping")
            return

        session_maker = get_sessionmaker()
        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)

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

            if claim_texts:
                resp = await client.embeddings.create(model=model, input=claim_texts)
                vectors = [d.embedding for d in resp.data]
                if vectors and len(vectors[0]) != int(self._settings.embedding_dim):
                    logger.warning(
                        f"[RAG] Embedding dim mismatch: got={len(vectors[0])} expected={self._settings.embedding_dim}; skipping"
                    )
                else:
                    it = iter([_normalize_vec(v) for v in vectors])
                    for c in claim_rows:
                        if c.claim_embedding is None:
                            c.claim_embedding = next(it)
                            updated_claims += 1

            if evidence_texts:
                resp = await client.embeddings.create(model=model, input=evidence_texts)
                vectors = [d.embedding for d in resp.data]
                if vectors and len(vectors[0]) != int(self._settings.embedding_dim):
                    logger.warning(
                        f"[RAG] Embedding dim mismatch: got={len(vectors[0])} expected={self._settings.embedding_dim}; skipping"
                    )
                else:
                    for e, vec in zip(evidence_rows, vectors, strict=False):
                        e.snippet_embedding = _normalize_vec(vec)
                        updated_evidence += 1

            if updated_claims or updated_evidence:
                await session.commit()
                logger.info(
                    f"[RAG] Stored embeddings verification_id={verification_id} claims={updated_claims} evidence={updated_evidence}"
                )
            else:
                logger.info(f"[RAG] No embeddings stored for verification_id={verification_id}")
