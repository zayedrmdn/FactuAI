# Environment Variables Reference

Complete reference for all FactuAI environment variables.

---

## Database Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@localhost:5433/factuai` | ✅ Yes | PostgreSQL connection string (async driver) |
| `DB_REQUIRED` | `true` | No | If `true`, app fails to start without DB connection |
| `DB_RUN_MIGRATIONS` | `true` | No | Auto-run SQL migrations on startup |

**Example:**
```bash
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/factuai
DB_REQUIRED=true
DB_RUN_MIGRATIONS=true
```

---

## Redis Configuration

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `REDIS_URL` | `redis://localhost:6379/0` | No | Redis connection string |
| `REDIS_REQUIRED` | `false` | No | If `true`, app fails without Redis connection |

**Example:**
```bash
REDIS_URL=redis://localhost:6379/0
REDIS_REQUIRED=false
```

---

## LLM Provider (Verification)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `LLM_API_BASE_URL` | `https://openrouter.ai/api/v1` | No | OpenAI-compatible API base URL |
| `LLM_API_KEY` | - | ✅ Yes | API key for LLM provider |
| `OPENROUTER_API_KEY` | - | No | Alias for `LLM_API_KEY` (OpenRouter-specific) |
| `OPENROUTER_MODEL` | `meta-llama/llama-3.3-70b-instruct` | No | Default model for verification |

**Example:**
```bash
LLM_API_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=sk-or-v1-...
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct
```

---

## Intent Extraction (LLM-Based)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `INTENT_ADAPTER` | `backend.app.features.intent.adapters.llm.LLMIntentAdapter` | No | Intent extraction adapter class path |
| `INTENT_LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct` | No | Model for claim extraction |
| `INTENT_LLM_API_KEY` | Falls back to `LLM_API_KEY` | No | Separate API key for intent extraction |
| `INTENT_LLM_BASE_URL` | Falls back to `LLM_API_BASE_URL` | No | Separate base URL for intent extraction |

**Example:**
```bash
INTENT_ADAPTER=backend.app.features.intent.adapters.llm.LLMIntentAdapter
INTENT_LLM_MODEL=meta-llama/llama-3.3-70b-instruct
```

---

## Search Providers

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `SEARCH_PROVIDER_PATHS` | `backend.app.features.search.providers.tavily.TavilyProvider` | No | Comma-separated list of provider class paths |
| `TAVILY_API_KEY` | - | Recommended | Tavily API key (primary search provider) |
| `NEWSAPI_API_KEY` | - | No | NewsAPI key (legacy, not recommended) |

**Example:**
```bash
SEARCH_PROVIDER_PATHS=backend.app.features.search.providers.tavily.TavilyProvider
TAVILY_API_KEY=tvly-...
```

---

## Embeddings (Continuous Learning)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `EMBEDDING_API_BASE_URL` | - | Optional | Embedding service URL (e.g., Infinity) |
| `EMBEDDING_API_KEY` | - | Optional | API key for embedding service |
| `EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` | No | Embedding model name |
| `EMBEDDING_DIM` | `384` | No | Embedding vector dimension |
| `LEARNING_CONFIDENCE_THRESHOLD` | `0.85` | No | Minimum confidence to trigger learning  |
| `RAG_RETRIEVAL_THRESHOLD` | `0.25` | No | Cosine distance cutoff for RAG retrieval |

**Example:**
```bash
EMBEDDING_API_BASE_URL=http://localhost:7997
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
LEARNING_CONFIDENCE_THRESHOLD=0.85
RAG_RETRIEVAL_THRESHOLD=0.25
```

> [!NOTE]
> Continuous learning only works when embedding service is configured. The system fails gracefully if embeddings are unavailable.

---

## Verification Adapter

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `VERIFIER_ADAPTER` | `backend.app.features.verification.adapters.native.NativeVerificationAdapter` | No | Verification adapter class path |

**Example:**
```bash
VERIFIER_ADAPTER=backend.app.features.verification.adapters.native.NativeVerificationAdapter
```

---

## Frontend (Next.js)

| Variable | Default | Required | Description |
|----------|---------|----------|-------------|
| `NEXT_PUBLIC_API_URL` | `http://127.0.0.1:8000` | No | Backend API base URL |

**Example (`frontend/.env.local`):**
```bash
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

> [!IMPORTANT]
> Frontend env vars must be prefixed with `NEXT_PUBLIC_` to be exposed to the browser.

---

## Security Best Practices

1. **Never commit `.env` files** - They're in `.gitignore` for a reason
2. **Rotate API keys regularly** - Especially for production
3. **Use separate keys per environment** - Dev, staging, production should have different credentials
4. **Principle of least privilege** - Only set what you need

---

## Quick Reference: Minimal Setup

For local development, you only need:

```bash
# backend/.env
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/factuai
LLM_API_KEY=your_openrouter_key
TAVILY_API_KEY=your_tavily_key
```

Everything else has sensible defaults.

---

See [../01-rules/constitution.md](../01-rules/constitution.md) for environment-driven configuration principles.
