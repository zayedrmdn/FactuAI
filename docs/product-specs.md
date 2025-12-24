# FactuAI Product Specifications

**Version:** 4.0.0  
**Last Updated:** 2025-12-12

## Product Overview

FactuAI is a full-stack AI-powered fact-checking system that analyzes claims, gathers evidence from multiple sources, and provides verification verdicts with confidence scores.

**Backend:** Native async FastAPI with PostgreSQL + pgvector for continuous learning (RAG feedback loop)  
**Frontend:** Next.js (App Router), TypeScript, Tailwind

## Tech Stack

**Backend:**
- FastAPI (async)
- SQLAlchemy async + asyncpg
- PostgreSQL 16+ with pgvector extension
- Redis (async) for caching
- httpx for async HTTP
- LangChain (`langchain-openai`) for LLM verification (async + structured outputs)

**Infrastructure:**
- Docker Compose (Postgres + Redis)
- Port mappings: Postgres `5433` (host) → `5432` (container), Redis `6379`

## Architecture Principles (NON-NEGOTIABLE)

1. **Async-First:** All I/O uses async/await (no sync wrappers, no thread bridges)
2. **Vertical Slice Architecture:** Features under `backend/app/features/*` own their API boundary and logic
3. **No Cross-Feature Imports:** Features never import other features; shared types live in `backend/app/contracts/`
4. **OCP + Dependency Injection:** Extend via new implementations + config in `backend/app/core/container.py`, not by modifying orchestrators
5. **No Legacy Code:** Do not reintroduce removed pipeline/Flask/sync adapter patterns

## API Endpoints

**Base URL:** `http://127.0.0.1:8000`

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Liveness check |
| POST | `/api/analyze` | Multi-claim fact-checking analysis |

### Request Schema (`POST /api/analyze`)

```json
{
  "text": "string",
  "provider": "openrouter",
  "model_id": "meta-llama/llama-3.3-70b-instruct",
  "max_claims": 3,
  "enable_web_search": true,
  "enable_kb": true,
  "analysis_mode": "deep",
  "pipeline_models": {
    "intent": {
      "model_id": "openai/gpt-4o-mini"
    },
    "extraction": {
      "model_id": "openai/gpt-4o-mini"
    },
    "reasoning": {
      "model_id": "meta-llama/llama-3.3-70b-instruct"
    }
  }
}
```

**Analysis Modes:**
- `quick`: Single direct search, no strategist, no pivot (~6-10s)
- `deep`: Full 4-phase pipeline with multi-angle queries and pivot loop (~10-16s)

### Response Schema

```json
{
  "request_id": "uuid",
  "claims": [
    {
      "claim_text": "string",
      "verdict": "TRUE|FALSE|MIXED|UNVERIFIABLE",
      "confidence": 0.0-1.0,
      "reasoning": "string",
      "evidence": [
        {
          "snippet": "string",
          "source_url": "string",
          "source_title": "string",
          "relevance_score": 0.0-1.0
        }
      ]
    }
  ],
  "model_used": "string",
  "latency_ms": 0
}
```

## Data Model

**Database:** PostgreSQL with pgvector extension

### Core Tables

**verifications**
- `id` BIGSERIAL PK
- `request_id` UUID UNIQUE
- `input_text` TEXT
- `model_used` VARCHAR(255)
- `latency_ms` INTEGER
- `verdict` VARCHAR(50)
- `confidence` NUMERIC(3,2)
- `created_at` TIMESTAMPTZ

**claims**
- `id` BIGSERIAL PK
- `verification_id` BIGINT FK → verifications
- `claim_text` TEXT NOT NULL
- `verdict` VARCHAR(50)
- `confidence` NUMERIC(3,2)
- `reasoning` TEXT
- `claim_embedding` VECTOR(384) -- pgvector for RAG
- `created_at` TIMESTAMPTZ

**sources**
- `id` BIGSERIAL PK
- `url` TEXT UNIQUE NOT NULL
- `title` TEXT
- `domain` VARCHAR(255)
- `credibility_score` NUMERIC(3,2)
- `first_seen_at`, `last_seen_at` TIMESTAMPTZ

**evidence**
- `id` BIGSERIAL PK
- `claim_id` BIGINT FK → claims
- `source_id` BIGINT FK → sources
- `snippet` TEXT NOT NULL
- `relevance_score` NUMERIC(3,2)
- `snippet_embedding` VECTOR(384)
- `captured_at` TIMESTAMPTZ
- UNIQUE(`claim_id`, `source_id`, `snippet`)

## Feature Slices

**Location:** `backend/app/features/`

1. **analyze/** - API boundary, orchestrates the full fact-check flow
2. **intent/** - Extracts structured claims from raw text
3. **search/** - Pluggable search providers (Tavily, NewsAPI, etc.)
4. **verification/** - LLM-based verdict grading (OpenAI-compatible endpoints)

Implementation note:
- The default verifier adapter uses LangChain (`langchain-openai`) to enforce strict structured output parsing and schema validation.

**Shared Contracts:** `backend/app/contracts/` (cross-feature types only)

## Environment Configuration

**Critical Variables:**

```bash
# Database (required)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/factuai
DB_REQUIRED=true
DB_RUN_MIGRATIONS=true

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
REDIS_REQUIRED=false

# Search Providers (optional)
SEARCH_PROVIDER_PATHS=backend.app.features.search.providers.tavily.TavilyProvider,backend.app.features.search.providers.newsapi.NewsApiProvider
TAVILY_API_KEY=
NEWSAPI_API_KEY=

# LLM Verification (required for real verification)
LLM_API_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=
OPENROUTER_MODEL=alibaba/tongyi-deepresearch-30b-a3b

# Embeddings for Continuous Learning (optional)
EMBEDDING_API_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
LEARNING_CONFIDENCE_THRESHOLD=0.85
```

## Key Workflows

### 1. Fact-Check Analysis Flow

**Deep Mode (Default):**
```
POST /api/analyze
  ↓
Phase 1: Intent Extraction + Strategist → Claims[] + Multi-Angle Queries
  ↓
Phase 2: Parallel Search (3 queries + RAG) → Evidence[]
  ↓
Phase 3: Pivot Loop (conditional) → Additional Evidence[]
  ↓
Phase 4: Verification LLM → Verdict + Confidence
  ↓
Persistence → DB
  ↓
[If confidence ≥ 0.85] Continuous Learning → Embeddings
  ↓
Response
```

**Quick Mode:**
```
POST /api/analyze (analysis_mode=quick)
  ↓
Phase 1: Intent Extraction → Claims[]
  ↓
Phase 2: Direct Search (15 results) → Evidence[]
  ↓
Phase 4: Verification LLM → Verdict + Confidence
  ↓
Response
```

### 2. Search Provider Extension (OCP)

1. Create provider class in `backend/app/features/search/providers/`
2. Implement required interface
3. Add dotted path to `SEARCH_PROVIDER_PATHS`
4. No orchestrator changes needed

### 3. Continuous Learning (RAG Feedback Loop)

- Triggered after high-confidence verifications (≥ `LEARNING_CONFIDENCE_THRESHOLD`)
- Computes embeddings for claims and evidence snippets
- Stores in pgvector columns for future retrieval
- Failures must log safely without breaking the main flow

## Local Development

**Start infrastructure:**
```powershell
docker-compose up -d
```

**Run backend:**
```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

**Run tests:**
```powershell
pytest -q
```

## Quality Gates

- All tests must pass (`pytest`)
- No new warnings in normal runs
- Never commit secrets or API keys
- Maintain async-first patterns
- Respect feature isolation

## Entry Points

- **Backend:** `backend/app/main.py`
- **Config:** `backend/app/core/settings.py`
- **DI Container:** `backend/app/core/container.py`
- **DB Initialization:** `backend/app/core/db.py`
- **Migrations:** `backend/migrations/*.sql`

---

**For full architectural details:** See `docs/ARCHITECTURE.md`  
**For AI agent rules:** See `CONSTITUTION.md` and `AGENTS.md`