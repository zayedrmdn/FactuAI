from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import Depends, Request
from redis.asyncio import Redis, from_url
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.container import Container
from app.core.settings import get_settings, Settings
from app.core.db import get_sessionmaker, init_db


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
    except Exception:
        app.state.redis = None
        if settings.redis_required:
            raise

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
