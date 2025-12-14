# Full Path: backend/app/core/rate_limit.py
"""
Rate limiting middleware for FastAPI endpoints.

Uses slowapi for rate limiting with Redis backend (when available)
or in-memory storage as fallback.

This protects expensive endpoints from accidental spamming or
denial-of-service attacks that could drain API credits.
"""
from __future__ import annotations

from typing import Optional, Callable

from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)


def _get_client_identifier(request: Request) -> str:
    """
    Get a unique identifier for the client.
    
    Uses X-Forwarded-For header if behind a proxy, otherwise
    falls back to the direct remote address.
    """
    # Check for forwarded header (common with reverse proxies)
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Take the first IP in the chain
        return forwarded.split(",")[0].strip()
    
    # Check for real IP header (used by some proxies)
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip.strip()
    
    # Fallback to direct client address
    return get_remote_address(request)


# Create the limiter instance
# Storage will be configured during app startup based on Redis availability
# Note: headers_enabled is False because injecting headers into Pydantic model
# responses causes serialization errors. Rate limit info is returned in error responses.
limiter = Limiter(
    key_func=_get_client_identifier,
    default_limits=["100/minute"],  # Default for non-decorated endpoints
    headers_enabled=False,  # Disabled to work with Pydantic response models
    strategy="fixed-window",  # Simple fixed window strategy
)


def configure_redis_storage(redis_url: Optional[str]) -> None:
    """
    Configure the limiter to use Redis storage if available.
    
    Call this during app startup after Redis connection is established.
    """
    if redis_url:
        try:
            from slowapi.util import get_remote_address
            # slowapi will use redis if we set the storage_uri
            limiter._storage_uri = redis_url
            logger.info(f"[RATE_LIMIT] Configured with Redis storage")
        except Exception as exc:
            logger.warning(f"[RATE_LIMIT] Failed to configure Redis storage: {exc}")
            logger.info("[RATE_LIMIT] Using in-memory storage (not distributed)")
    else:
        logger.info("[RATE_LIMIT] Using in-memory storage (Redis not available)")


def rate_limit_exceeded_handler(request: Request, exc: RateLimitExceeded) -> Response:
    """
    Custom handler for rate limit exceeded errors.
    
    Returns a JSON response with details about the rate limit.
    """
    from fastapi.responses import JSONResponse
    
    retry_after = exc.detail.split("retry after ")[1] if "retry after" in exc.detail else "60"
    
    logger.warning(
        f"[RATE_LIMIT] Exceeded for {_get_client_identifier(request)} "
        f"on {request.url.path}: {exc.detail}"
    )
    
    return JSONResponse(
        status_code=429,
        content={
            "error": "rate_limit_exceeded",
            "message": "Too many requests. Please slow down.",
            "detail": str(exc.detail),
            "retry_after_seconds": retry_after,
        },
        headers={
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": exc.detail.split()[0] if exc.detail else "unknown",
        }
    )


# Rate limit decorators for different endpoint types
def analyze_rate_limit() -> Callable:
    """
    Rate limit decorator for the /api/analyze endpoint.
    
    This is the most expensive endpoint (incurs LLM and Search API costs),
    so we apply strict rate limiting.
    
    Default: 10 requests per minute per client.
    """
    settings = get_settings()
    limit = f"{settings.rate_limit_analyze_per_minute}/minute"
    return limiter.limit(limit)


def auth_rate_limit() -> Callable:
    """
    Rate limit decorator for authentication endpoints.
    
    Login/register endpoints should be rate limited to prevent
    brute force attacks.
    
    Default: 20 requests per minute per client.
    """
    settings = get_settings()
    limit = f"{settings.rate_limit_auth_per_minute}/minute"
    return limiter.limit(limit)


def default_rate_limit() -> Callable:
    """
    Rate limit decorator for general endpoints.
    
    Default: 100 requests per minute per client.
    """
    settings = get_settings()
    limit = f"{settings.rate_limit_default_per_minute}/minute"
    return limiter.limit(limit)
