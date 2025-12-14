# Full Path: backend/app/features/analyze/router.py
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request, status
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container
from app.core.deps import get_app_settings, get_container, get_db, get_redis, get_health_checker
from app.core.settings import Settings
from app.core.logging import get_logger
from app.core.rate_limit import limiter
from app.core.health import InfrastructureHealthChecker, InfrastructureStatus
from app.features.analyze.schemas import (
    AnalyzeRequest,
    AnalyzeResponse,
    ClaimResult,
    EvidenceItem,
    ErrorResponse,
)
from app.features.analyze.service import AnalyzeService

logger = get_logger(__name__)
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
        status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Rate limit exceeded"},
        status.HTTP_500_INTERNAL_SERVER_ERROR: {"model": ErrorResponse},
        status.HTTP_503_SERVICE_UNAVAILABLE: {"description": "Infrastructure unavailable"},
    },
)
@limiter.limit("10/minute")  # Rate limit: 10 requests per minute per IP
async def analyze(
    request: Request,
    body: AnalyzeRequest,
    db: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
    redis: Redis | None = Depends(get_redis),
    container: Container = Depends(get_container),
    health_checker: InfrastructureHealthChecker = Depends(get_health_checker),
):
    """
    Analyze text for factual claims and verify them against evidence.
    
    This endpoint implements the "Fail Fast, Fail Cheap" philosophy:
    1. Pre-flight checks validate infrastructure before expensive operations
    2. Rate limiting prevents abuse and credit drainage
    3. Circuit breakers protect against cascading failures
    
    Rate Limit: 10 requests per minute per IP address.
    """
    # === PRE-FLIGHT INFRASTRUCTURE CHECKS ===
    # Validate connectivity to downstream services BEFORE any expensive operations.
    # This prevents spending money on LLM/Search API calls when infrastructure is down.
    if settings.preflight_checks_enabled:
        report = await health_checker.check_all()
        
        # Database is required for persisting results
        if report.database.status == InfrastructureStatus.UNHEALTHY:
            logger.error(f"[PREFLIGHT] Database check failed: {report.database.error}")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Service temporarily unavailable: Database connection failed. Please try again later.",
            )
        
        # Redis is optional but we log if it's down
        if report.redis.status == InfrastructureStatus.UNHEALTHY:
            logger.warning(f"[PREFLIGHT] Redis check failed (continuing): {report.redis.error}")
        
        # Embedding service is optional - only warn if configured but unreachable
        # The actual learning step will be skipped gracefully
        if report.embedding_service.status == InfrastructureStatus.UNHEALTHY:
            logger.warning(
                f"[PREFLIGHT] Embedding service unreachable: {report.embedding_service.error}. "
                "Continuous learning will be skipped for this request."
            )

    # === MAIN PROCESSING ===
    # Now that infrastructure is validated, proceed with expensive operations.
    service = AnalyzeService(settings=settings, container=container, db=db, redis=redis)
    try:
        logger.info("[ANALYZE] Starting analysis...")
        request_id, model_used, latency_ms, claims = await service.analyze(body)
        logger.info(f"[ANALYZE] Analysis complete: {len(claims)} claims, {latency_ms}ms")
        response = _to_response(request_id, model_used, latency_ms, claims)
        logger.info(f"[ANALYZE] Response built successfully")
        return response
    except ValueError as exc:
        logger.warning(f"[ANALYZE] ValueError: {exc}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception(f"[ANALYZE] Unexpected error: {exc}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal error") from exc
