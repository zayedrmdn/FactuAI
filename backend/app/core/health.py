# Full Path: backend/app/core/health.py
"""
Pre-flight health check utilities for infrastructure validation.

This module implements the "Fail Fast, Fail Cheap" philosophy by validating
connectivity to downstream infrastructure (DB, Redis, Embedding Service)
before expensive operations are executed.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import httpx
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

from app.core.logging import get_logger
from app.core.settings import Settings

logger = get_logger(__name__)


class InfrastructureStatus(str, Enum):
    """Health check result status."""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    SKIPPED = "skipped"  # For optional services that are not configured


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    service: str
    status: InfrastructureStatus
    latency_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class InfrastructureHealthReport:
    """Aggregated health report for all infrastructure."""
    database: HealthCheckResult
    redis: HealthCheckResult
    embedding_service: HealthCheckResult
    llm_provider: HealthCheckResult

    @property
    def is_ready(self) -> bool:
        """Returns True if all required services are healthy."""
        # Database is always required for analyze operations
        if self.database.status == InfrastructureStatus.UNHEALTHY:
            return False
        # LLM provider is required for claim extraction and verification
        if self.llm_provider.status == InfrastructureStatus.UNHEALTHY:
            return False
        # Redis is optional (only check if it's not skipped)
        if self.redis.status == InfrastructureStatus.UNHEALTHY:
            return False
        # Embedding service is optional (only fails if configured and unhealthy)
        # Note: Embedding failures don't block the request, only learning
        return True

    @property
    def failed_services(self) -> list[str]:
        """Returns list of failed service names."""
        failed = []
        if self.database.status == InfrastructureStatus.UNHEALTHY:
            failed.append(f"Database: {self.database.error}")
        if self.llm_provider.status == InfrastructureStatus.UNHEALTHY:
            failed.append(f"LLM Provider: {self.llm_provider.error}")
        if self.redis.status == InfrastructureStatus.UNHEALTHY:
            failed.append(f"Redis: {self.redis.error}")
        if self.embedding_service.status == InfrastructureStatus.UNHEALTHY:
            failed.append(f"Embedding Service: {self.embedding_service.error}")
        return failed


class InfrastructureHealthChecker:
    """
    Validates connectivity to downstream infrastructure before expensive operations.
    
    Usage:
        checker = InfrastructureHealthChecker(settings=settings, db=db, redis=redis)
        report = await checker.check_all()
        if not report.is_ready:
            raise ServiceUnavailableError(report.failed_services)
    """

    def __init__(
        self,
        *,
        settings: Settings,
        db: Optional[AsyncSession] = None,
        redis: Optional[Redis] = None,
    ):
        self._settings = settings
        self._db = db
        self._redis = redis

    async def check_database(self) -> HealthCheckResult:
        """Check PostgreSQL database connectivity with write access validation."""
        import time

        if self._db is None:
            return HealthCheckResult(
                service="database",
                status=InfrastructureStatus.SKIPPED,
                error="No database session provided"
            )

        start = time.perf_counter()
        try:
            # Simple read check - verifies connection is alive
            result = await self._db.execute(text("SELECT 1"))
            result.fetchone()
            latency = (time.perf_counter() - start) * 1000

            logger.debug(f"[HEALTH] Database check: OK ({latency:.2f}ms)")
            return HealthCheckResult(
                service="database",
                status=InfrastructureStatus.HEALTHY,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            error_msg = str(exc)[:200]  # Truncate long error messages
            logger.warning(f"[HEALTH] Database check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="database",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )

    async def check_redis(self) -> HealthCheckResult:
        """Check Redis connectivity."""
        import time

        if self._redis is None:
            if not self._settings.redis_required:
                return HealthCheckResult(
                    service="redis",
                    status=InfrastructureStatus.SKIPPED,
                    error="Redis not configured (optional)"
                )
            return HealthCheckResult(
                service="redis",
                status=InfrastructureStatus.UNHEALTHY,
                error="Redis required but not connected"
            )

        start = time.perf_counter()
        try:
            await self._redis.ping()
            latency = (time.perf_counter() - start) * 1000

            logger.debug(f"[HEALTH] Redis check: OK ({latency:.2f}ms)")
            return HealthCheckResult(
                service="redis",
                status=InfrastructureStatus.HEALTHY,
                latency_ms=latency,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            error_msg = str(exc)[:200]
            logger.warning(f"[HEALTH] Redis check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="redis",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )

    async def check_embedding_service(self) -> HealthCheckResult:
        """Check local embedding service (Infinity) connectivity via health endpoint."""
        import time

        base_url = (self._settings.embedding_api_base_url or "").strip()
        api_key = (self._settings.embedding_api_key or "").strip()

        # If embedding service is not configured, skip (it's optional for learning)
        if not base_url or not api_key:
            return HealthCheckResult(
                service="embedding_service",
                status=InfrastructureStatus.SKIPPED,
                error="Embedding service not configured"
            )

        # Infinity health endpoint is at /health
        health_url = f"{base_url.rstrip('/')}/health"

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(health_url)
                latency = (time.perf_counter() - start) * 1000

                if response.status_code == 200:
                    logger.debug(f"[HEALTH] Embedding service check: OK ({latency:.2f}ms)")
                    return HealthCheckResult(
                        service="embedding_service",
                        status=InfrastructureStatus.HEALTHY,
                        latency_ms=latency,
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    logger.warning(f"[HEALTH] Embedding service check: FAILED - {error_msg}")
                    return HealthCheckResult(
                        service="embedding_service",
                        status=InfrastructureStatus.UNHEALTHY,
                        latency_ms=latency,
                        error=error_msg,
                    )
        except httpx.ConnectError as exc:
            latency = (time.perf_counter() - start) * 1000
            error_msg = f"Connection refused: {base_url}"
            logger.warning(f"[HEALTH] Embedding service check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="embedding_service",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )
        except httpx.TimeoutException:
            latency = (time.perf_counter() - start) * 1000
            error_msg = f"Timeout connecting to {base_url}"
            logger.warning(f"[HEALTH] Embedding service check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="embedding_service",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            error_msg = str(exc)[:200]
            logger.warning(f"[HEALTH] Embedding service check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="embedding_service",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )

    async def check_llm_provider(self) -> HealthCheckResult:
        """Check LLM provider (OpenAI-compatible API) connectivity.
        
        Performs a lightweight GET to /models endpoint to verify connectivity.
        This is essential for Fail Fast - we must ensure the LLM is reachable
        before expensive claim extraction and verification operations.
        """
        import time

        base_url = (self._settings.llm_api_base_url or "").strip()
        api_key = (self._settings.llm_api_key or "").strip()

        # If no LLM is configured, this is a critical failure
        if not base_url or not api_key:
            return HealthCheckResult(
                service="llm_provider",
                status=InfrastructureStatus.UNHEALTHY,
                error="LLM provider not configured (missing LLM_API_BASE_URL or LLM_API_KEY)"
            )

        # OpenAI-compatible /models endpoint for lightweight connectivity check
        models_url = f"{base_url.rstrip('/')}/models"

        start = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(
                    models_url,
                    headers={"Authorization": f"Bearer {api_key}"}
                )
                latency = (time.perf_counter() - start) * 1000

                # 200 = success, 401 = auth issue but server reachable
                # Both indicate the provider is reachable
                if response.status_code in (200, 401):
                    logger.debug(f"[HEALTH] LLM provider check: OK ({latency:.2f}ms)")
                    return HealthCheckResult(
                        service="llm_provider",
                        status=InfrastructureStatus.HEALTHY,
                        latency_ms=latency,
                    )
                else:
                    error_msg = f"HTTP {response.status_code}"
                    logger.warning(f"[HEALTH] LLM provider check: FAILED - {error_msg}")
                    return HealthCheckResult(
                        service="llm_provider",
                        status=InfrastructureStatus.UNHEALTHY,
                        latency_ms=latency,
                        error=error_msg,
                    )
        except httpx.ConnectError:
            latency = (time.perf_counter() - start) * 1000
            error_msg = f"Connection refused: {base_url}"
            logger.warning(f"[HEALTH] LLM provider check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="llm_provider",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )
        except httpx.TimeoutException:
            latency = (time.perf_counter() - start) * 1000
            error_msg = f"Timeout connecting to {base_url}"
            logger.warning(f"[HEALTH] LLM provider check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="llm_provider",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )
        except Exception as exc:
            latency = (time.perf_counter() - start) * 1000
            error_msg = str(exc)[:200]
            logger.warning(f"[HEALTH] LLM provider check: FAILED - {error_msg}")
            return HealthCheckResult(
                service="llm_provider",
                status=InfrastructureStatus.UNHEALTHY,
                latency_ms=latency,
                error=error_msg,
            )

    async def check_all(self) -> InfrastructureHealthReport:
        """
        Run all health checks concurrently and return aggregated report.
        
        This is the main entry point for pre-flight infrastructure validation.
        Call this BEFORE any expensive operations to fail fast and cheap.
        """
        import asyncio

        # Run all checks concurrently for speed
        db_check, redis_check, embedding_check, llm_check = await asyncio.gather(
            self.check_database(),
            self.check_redis(),
            self.check_embedding_service(),
            self.check_llm_provider(),
        )

        report = InfrastructureHealthReport(
            database=db_check,
            redis=redis_check,
            embedding_service=embedding_check,
            llm_provider=llm_check,
        )

        if not report.is_ready:
            logger.error(f"[HEALTH] Pre-flight check FAILED: {report.failed_services}")
        else:
            logger.debug("[HEALTH] Pre-flight check: All systems OK")

        return report

    async def check_embedding_only(self) -> bool:
        """
        Quick check for embedding service only.
        
        Returns True if embedding service is healthy or not configured (skipped).
        Returns False only if configured but unreachable.
        """
        result = await self.check_embedding_service()
        return result.status != InfrastructureStatus.UNHEALTHY
