# AGENTS.md (Single Source of Truth for AI Coders)

This file is the **primary onboarding context** for AI coding agents working in this repo.

If anything here conflicts with other docs, follow in this order:
1) `CONSTITUTION.md` (rules)
2) `docs/*` (architecture + setup)
3) code (reality)

## Repo Purpose

FactuAI is a full-stack fact-checking system.

Backend goals:
- Native async FastAPI API
- Vertical Slice Architecture (features own routers/services/ports)
- Pluggable providers via DI (Open/Closed)
- Postgres + pgvector for continuous learning (RAG feedback loop)
- Redis for caching

## Tech Stack

Backend:
- FastAPI
- SQLAlchemy async + asyncpg
- PostgreSQL + pgvector
- redis.asyncio
- httpx (async HTTP)
- LangChain (`langchain-openai`) for LLM verification (async + structured outputs)

Frontend:
- Next.js (App Router), TypeScript, Tailwind

Infra:
- docker-compose (Postgres+pgvector + Redis)

## Where to Look (Map)

Entry points:
- `backend/app/main.py` (FastAPI app)
- `backend/app/core/deps.py` (lifespan + dependency providers)

Configuration:
- `backend/app/core/settings.py` (all env vars)
- `docker-compose.yml` (local infra ports)
- `.env.example` (template)

Feature slices:
- `backend/app/features/analyze/`
- `backend/app/features/intent/`
- `backend/app/features/search/`
- `backend/app/features/verification/`

Shared contracts:
- `backend/app/contracts/` (cross-feature types only)

Frontend feature modules (Feature-Based Colocation):
- `frontend/src/features/ai-providers/` (AI config, models, stores)
- `frontend/src/features/search/` (search input, providers)
- `frontend/src/features/analyze/` (analysis results display)
- `frontend/src/features/history/` (history panel, items)

> **Warning**: Domain components *must* live in `frontend/src/features/*/`, NOT in `components/`. The `components/` directory is reserved for generic UI primitives only.

Database:
- `backend/migrations/*.sql` (authoritative schema)

## Local Dev Commands (Windows)

Start infra:

```powershell
docker-compose up -d
```

Run backend:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Run tests:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pytest -q
```

## Environment Variables (Common)

Database:
- `DATABASE_URL` (preferred) or `DB_URI`
- Default (matches docker-compose host port mapping):
  - `postgresql+asyncpg://postgres:postgres@localhost:5433/factuai`
- `DB_REQUIRED` (`true|false`)
- `DB_RUN_MIGRATIONS` (`true|false`)

Redis:
- `REDIS_URL`
- `REDIS_REQUIRED`

LLM verification (OpenAI-compatible):
- `LLM_API_BASE_URL`
- `LLM_API_KEY`
- `LLM_PROVIDER`, `OPENROUTER_MODEL`

Embeddings (learning loop):
- `EMBEDDING_API_BASE_URL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `LEARNING_CONFIDENCE_THRESHOLD`
- `RAG_RETRIEVAL_THRESHOLD` (cosine distance cutoff for internal search, default 0.25)

Search providers:
- `SEARCH_PROVIDER_PATHS` (CSV dotted paths)
- `TAVILY_API_KEY`, `NEWSAPI_API_KEY`

## Operating Constraints (from CONSTITUTION.md)

Hard rules:
- Backend is **async-first**. No sync wrappers, no thread bridges for core I/O.
- **Vertical Slice Architecture**. Feature-to-feature imports are forbidden.
- **OCP + DI**. Extend behavior via new implementations + config, not by editing orchestrators.
- **No legacy pipeline**. Do not reintroduce removed architecture patterns.
- Standard AI orchestration libraries (LangChain/LangGraph) are allowed for LLM calls and structured parsing (see `CONSTITUTION.md`).

When in doubt, open `CONSTITUTION.md` and follow it.

## Safe Change Checklist (Agent)

Before editing:
- Identify the feature slice (don’t create cross-feature coupling).
- Check `docs/ARCHITECTURE.md` for intended boundaries.

When adding new behavior:
- Prefer a new adapter/provider class + DI config.
- Add/adjust contracts in `backend/app/contracts/` if types must cross boundaries.

After editing:
- Run `pytest` in `backend/`.
- Avoid introducing new warnings.

## Environment Variable Security Protocol

**CRITICAL: AI agents must NEVER attempt to read, view, or access the live `.env` file.**

This protocol exists to prevent accidental exposure of API keys and secrets in:
- Chat logs
- Code suggestions
- Debugging output
- File content displays

**Allowed Actions:**
- Read `.env.example` for configuration templates
- Reference `backend/app/core/settings.py` for default values
- Check `docs/AI_CONTEXT.md` or `AGENTS.md` for environment variable documentation
- Ask the user to verify their `.env` configuration manually

**Forbidden Actions:**
- Opening, reading, or displaying `.env` contents
- Suggesting commands that would print `.env` contents (e.g., `cat .env`, `type .env`)
- Attempting to bypass gitignore restrictions for `.env` files

When debugging environment issues, guide the user to check their `.env` manually and report only the relevant variable names (not values) that may be misconfigured.

