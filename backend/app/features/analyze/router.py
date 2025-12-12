from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container
from app.core.deps import get_app_settings, get_container, get_db, get_redis
from app.core.settings import Settings
from app.features.analyze.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClaimResult,
    EvidenceItem,
    ErrorResponse,
)
from app.features.analyze.service import AnalyzeService

router = APIRouter(tags=["analyze"])


def _to_response(request_id, model_used: str, latency_ms: int, claims) -> AnalyzeResponse:
    claim_results = []
    for c in claims:
        evidence_items = [
            EvidenceItem(
                snippet=e.snippet,
                source_url=e.source_url,
                source_title=e.source_title,
                source_domain=e.source_domain,
                relevance_score=e.relevance_score,
            )
            for e in c.evidence
        ]
        claim_results.append(
            ClaimResult(
                claim_text=c.claim_text,
                verdict=c.verdict,
                confidence=c.confidence,
                reasoning=c.reasoning,
                evidence=evidence_items,
            )
        )

    return AnalyzeResponse(
        request_id=request_id,
        model_used=model_used,
        latency_ms=latency_ms,
        claims=claim_results,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        status.HTTP_400_BAD_REQUEST: {"model": ErrorResponse},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
    },
)
async def analyze(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    redis: Redis | None = Depends(get_redis),
    container: Container = Depends(get_container),
):
    service = AnalyzeService(settings=settings, container=container, db=db, redis=redis)
    try:
        request_id, model_used, latency_ms, claims = await service.analyze(request)
        return _to_response(request_id, model_used, latency_ms, claims)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error") from exc
