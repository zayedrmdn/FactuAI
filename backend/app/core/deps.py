# Full Path: backend/app/core/deps.py
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, Request
from redis.asyncio import Redis, from_url
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container
from app.core.settings import get_settings, Settings
from app.core.db import get_sessionmaker, init_db
from app.core.logging import get_logger
from app.core.rate_limit import limiter, configure_redis_storage, rate_limit_exceeded_handler
from app.core.health import InfrastructureHealthChecker

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app):
    settings = get_settings()

    # DB init (optional by default; strict in prod with DB_REQUIRED=true)
    await init_db()

    redis: Optional[Redis] = None

    try:
        redis = from_url(settings.redis_url, decode_responses=True)
        await redis.ping()
        app.state.redis = redis
        logger.info("[REDIS] Connected")
    except Exception as exc:
        app.state.redis = None
        logger.info(f"[REDIS] Unavailable: {exc}")
        if settings.redis_required:
            raise

    # Configure rate limiting storage
    if settings.rate_limit_enabled:
        if redis is not None:
            configure_redis_storage(settings.redis_url)
        else:
            configure_redis_storage(None)  # Use in-memory storage
        
        # Add rate limiter to app state
        app.state.limiter = limiter

    app.state.container = Container(settings=settings, redis=redis)

    yield

    if getattr(app.state, "redis", None) is not None:
        await app.state.redis.close()


async def get_db() -> AsyncIterator[AsyncSession]:
    session_maker = get_sessionmaker()
    async with session_maker() as session:
        yield session


def get_app_settings() -> Settings:
    return get_settings()


def get_redis(request: Request) -> Optional[Redis]:
    return getattr(request.app.state, "redis", None)


def get_container(request: Request) -> Container:
    return getattr(request.app.state, "container")


async def get_health_checker(
    request: Request,
    db: AsyncSession = Depends(get_db),
) -> InfrastructureHealthChecker:
    """
    FastAPI dependency to get an InfrastructureHealthChecker instance.
    
    Use this in route handlers to perform pre-flight infrastructure checks.
    """
    settings = get_settings()
    redis = get_redis(request)
    return InfrastructureHealthChecker(settings=settings, db=db, redis=redis)
