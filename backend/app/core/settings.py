# Full path: backend/app/core/settings.py
import os
from functools import lru_cache

from pydantic import BaseModel
from pydantic.config import ConfigDict


def _env_bool(name: str, default: str) -> bool:
    return os.getenv(name, default).lower() == "true"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


class Settings(BaseModel):
    app_name: str = "FactuAI API"

    database_url: str = os.getenv(
        "DATABASE_URL",
        os.getenv(
            "DB_URI",
            # Default aligns with docker-compose.yml (host port 5433) and async SQLAlchemy.
            "postgresql+asyncpg://postgres:postgres@localhost:5433/factuai",
        ),
    )
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

    # Defaults are "dev friendly" (service can boot without infra), but can be made strict in prod.
    db_required: bool = _env_bool("DB_REQUIRED", "false")
    db_run_migrations: bool = _env_bool("DB_RUN_MIGRATIONS", "true")

    redis_required: bool = _env_bool("REDIS_REQUIRED", "false")

    llm_provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    openrouter_model: str = os.getenv("OPENROUTER_MODEL", "tngtech/deepseek-r1t2-chimera:free")

    # OpenAI-compatible client configuration (OpenRouter, local gateways, etc.)
    llm_api_base_url: str = os.getenv(
        "LLM_API_BASE_URL",
        os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api/v1"),
    )
    llm_api_key: str = os.getenv(
        "LLM_API_KEY",
        os.getenv("OPENROUTER_API_KEY", os.getenv("OPENAI_API_KEY", "")),
    )

    # Embeddings (used by Continuous Learning / RAG feedback loop)
    embedding_api_base_url: str = os.getenv("EMBEDDING_API_BASE_URL", os.getenv("EMBEDDINGS_BASE_URL", ""))
    embedding_api_key: str = os.getenv("EMBEDDING_API_KEY", os.getenv("EMBEDDINGS_API_KEY", ""))
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
    embedding_dim: int = _env_int("EMBEDDING_DIM", 384)
    learning_confidence_threshold: float = float(os.getenv("LEARNING_CONFIDENCE_THRESHOLD", "0.85"))
    learning_max_evidence: int = _env_int("LEARNING_MAX_EVIDENCE", 8)

    # Intent LLM (Tier 1 - Fast/Cheap model for claim extraction)
    # If not set, falls back to main LLM_API_* settings
    intent_llm_api_base_url: str = os.getenv("INTENT_LLM_API_BASE_URL", "")
    intent_llm_api_key: str = os.getenv("INTENT_LLM_API_KEY", "")
    intent_llm_model: str = os.getenv("INTENT_LLM_MODEL", "qwen/qwen-2.5-7b-instruct")

    # --- Pluggable bindings (OCP-friendly) ---
    search_adapter: str = os.getenv(
        "SEARCH_ADAPTER",
        "app.features.search.adapters.native.NativeSearchService",
    )
    intent_adapter: str = os.getenv(
        "INTENT_ADAPTER",
        "app.features.intent.adapters.llm.LLMIntentAdapter",
    )
    verifier_adapter: str = os.getenv(
        "VERIFIER_ADAPTER",
        "app.features.verification.adapters.openai_compatible.OpenAICompatibleClaimVerifier",
    )

    # Search provider composition (OCP-friendly; add providers by adding new classes, not editing core logic)
    search_provider_paths_csv: str = os.getenv(
        "SEARCH_PROVIDER_PATHS",
        "app.features.search.providers.tavily.TavilySearchProvider,app.features.search.providers.newsapi.NewsApiSearchProvider",
    )

    # Provider credentials (kept in settings to avoid leaking into feature code)
    tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")
    newsapi_api_key: str = os.getenv("NEWSAPI_API_KEY", "")

    # Evidence cache
    evidence_cache_ttl_seconds: int = _env_int("EVIDENCE_CACHE_TTL_SECONDS", 1800)

    # Rate limiting
    rate_limit_enabled: bool = _env_bool("RATE_LIMIT_ENABLED", "true")
    rate_limit_analyze_per_minute: int = _env_int("RATE_LIMIT_ANALYZE_PER_MINUTE", 10)
    rate_limit_auth_per_minute: int = _env_int("RATE_LIMIT_AUTH_PER_MINUTE", 20)
    rate_limit_default_per_minute: int = _env_int("RATE_LIMIT_DEFAULT_PER_MINUTE", 60)
    preflight_checks_enabled: bool = _env_bool("PREFLIGHT_CHECKS_ENABLED", "true")

    # Verification config (kept here to avoid importing backend/config.py from feature code)
    token_estimate_ratio: float = float(os.getenv("TOKEN_ESTIMATE_RATIO", "3.0"))
    llm_max_tokens_base: int = _env_int("LLM_MAX_TOKENS_BASE", 800)
    llm_max_tokens_buffer: int = _env_int("LLM_MAX_TOKENS_BUFFER", 500)
    llm_max_tokens_max: int = _env_int("LLM_MAX_TOKENS_MAX", 4000)
    llm_max_tokens_reasoning_base: int = _env_int("LLM_MAX_TOKENS_REASONING_BASE", 2000)

    model_config = ConfigDict(frozen=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
