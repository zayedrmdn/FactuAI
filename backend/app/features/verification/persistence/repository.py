# Full Path: backend\app\features\verification\persistence\repository.py
from __future__ import annotations

from typing import Optional
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.features.verification.domain import ClaimAnalysis
from app.features.verification.persistence.models import Claim, Evidence, Source, Verification


def _extract_domain(url: str, default: str = "web") -> str:
    try:
        from urllib.parse import urlparse

        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc
    except Exception:
        pass
    return default


class SqlAlchemyVerificationRepository:
    def __init__(self, db: AsyncSession):
        self._db = db

    async def persist(
        self,
        *,
        request_id: UUID,
        input_text: str,
        model_used: str,
        latency_ms: int,
        claims: list[ClaimAnalysis],
    ) -> int:
        verification = Verification(
            request_id=request_id,
            user_id=None,
            input_text=input_text,
            model_used=model_used,
            latency_ms=latency_ms,
            verdict=claims[0].verdict if claims else "unverifiable",
            confidence=claims[0].confidence if claims else 0.0,
        )
        self._db.add(verification)
        await self._db.flush()

        for claim_result in claims:
            claim_row = Claim(
                verification_id=verification.id,
                claim_text=claim_result.claim_text,
                verdict=claim_result.verdict,
                confidence=claim_result.confidence,
                reasoning=claim_result.reasoning,
                claim_embedding=None,
            )
            self._db.add(claim_row)
            await self._db.flush()

            for ev in claim_result.evidence:
                result = await self._db.execute(select(Source).where(Source.url == str(ev.source_url)))
                source = result.scalar_one_or_none()
                if not source:
                    source = Source(
                        url=str(ev.source_url),
                        title=ev.source_title,
                        domain=_extract_domain(str(ev.source_url), default=ev.source_domain),
                        credibility_score=None,
                    )
                    self._db.add(source)
                    await self._db.flush()

                evidence_row = Evidence(
                    claim_id=claim_row.id,
                    source_id=source.id,
                    snippet=ev.snippet,
                    relevance_score=ev.relevance_score,
                    snippet_embedding=None,
                )
                self._db.add(evidence_row)

            await self._db.commit()
        return int(verification.id)
