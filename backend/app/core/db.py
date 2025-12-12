from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import declarative_base

from app.core.logging import get_logger
from app.core.settings import get_settings

logger = get_logger(__name__)


@dataclass(frozen=True)
class _EngineState:
    engine: AsyncEngine
    url: str


_engine_state: _EngineState | None = None
_sessionmaker: async_sessionmaker[AsyncSession] | None = None


def _ensure_engine() -> AsyncEngine:
    """Create/configure the SQLAlchemy engine lazily.

    This avoids binding to env vars at import-time (important for uvicorn reload and tests).
    """

    global _engine_state

    settings = get_settings()
    url = settings.database_url

    # Backward-compatible normalization: accept sync URLs but run via asyncpg.
    if url.startswith("postgresql+psycopg2://"):
        url = "postgresql+asyncpg://" + url.removeprefix("postgresql+psycopg2://")
    elif url.startswith("postgresql://"):
        url = "postgresql+asyncpg://" + url.removeprefix("postgresql://")

    if _engine_state is not None and _engine_state.url == url:
        return _engine_state.engine

    engine = create_async_engine(
        url,
        echo=False,
        pool_pre_ping=True,
    )

    global _sessionmaker
    _sessionmaker = async_sessionmaker(bind=engine, expire_on_commit=False, autoflush=False)
    _engine_state = _EngineState(engine=engine, url=url)
    return engine


def get_sessionmaker() -> async_sessionmaker[AsyncSession]:
    """Return the configured async sessionmaker (creating an engine if needed)."""

    global _sessionmaker
    if _sessionmaker is None:
        _ensure_engine()
    assert _sessionmaker is not None
    return _sessionmaker

Base = declarative_base()


def _split_sql_statements(sql: str) -> list[str]:
    statements: list[str] = []
    current: list[str] = []
    in_single_quote = False
    in_double_quote = False

    for ch in sql:
        if ch == "'" and not in_double_quote:
            in_single_quote = not in_single_quote
        elif ch == '"' and not in_single_quote:
            in_double_quote = not in_double_quote

        if ch == ";" and not in_single_quote and not in_double_quote:
            stmt = "".join(current).strip()
            if stmt:
                statements.append(stmt)
            current = []
            continue

        current.append(ch)

    tail = "".join(current).strip()
    if tail:
        statements.append(tail)

    return statements


def _load_migration_files(migrations_dir: Path) -> Iterable[Path]:
    if not migrations_dir.exists():
        return []
    return sorted(migrations_dir.glob("*.sql"))


async def apply_migrations() -> None:
    # backend/app/core/db.py -> parents[2] == backend/
    migrations_dir = Path(__file__).resolve().parents[2] / "migrations"
    sql_files = list(_load_migration_files(migrations_dir))
    if not sql_files:
        logger.info(f"[DB] No migration files found in {migrations_dir}")
        return

    engine = _ensure_engine()

    async with engine.begin() as conn:
        for path in sql_files:
            sql = path.read_text(encoding="utf-8")
            for statement in _split_sql_statements(sql):
                await conn.exec_driver_sql(statement)
            logger.info(f"[DB] Applied migration {path.name}")


async def init_db() -> None:
    try:
        settings = get_settings()

        # Import models so SQLAlchemy relationships/types are registered.
        from app.features.verification.persistence import models  # noqa: F401

        if settings.db_run_migrations:
            await apply_migrations()

        engine = _ensure_engine()

        async with engine.connect() as conn:
            await conn.exec_driver_sql("SELECT 1")
        logger.info("[DB] Ready")
    except Exception as exc:
        settings = get_settings()
        if settings.db_required:
            logger.error(f"[DB] Initialization failed: {exc}")
            raise
        logger.info(f"[DB] Unavailable (continuing): {exc}")
