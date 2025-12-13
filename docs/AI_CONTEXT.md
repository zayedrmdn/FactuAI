# AI Context (Cheat Sheet)

Use this file to onboard an AI agent quickly.

## What This Repo Is

FactuAI is a fact-checking system:
- Backend: **native async FastAPI**
- Data: PostgreSQL + **pgvector**
- Caching: Redis
- Architecture: Vertical Slice + Ports/Adapters + DI (OCP)

## Non-Negotiable Rules

- Do not reintroduce legacy modules (old internal pipeline / Flask stack / sync adapters).
- Features do not import other features. Share types via `backend/app/contracts/`.
- All I/O is async (httpx, asyncpg, redis.asyncio).
- Standard AI orchestration libraries (LangChain/LangGraph) are allowed for LLM calls and structured output parsing (async-first).

## Where Things Live

Backend entry:
- `backend/app/main.py`

DI + config:
- `backend/app/core/settings.py`
- `backend/app/core/container.py`

Feature slices:
- `backend/app/features/analyze/`
- `backend/app/features/intent/`
- `backend/app/features/search/`
- `backend/app/features/verification/`

Migrations:
- `backend/migrations/*.sql`

## Common Tasks

### Add a New Search Provider

1. Create a provider class under `backend/app/features/search/providers/`
2. Ensure it implements the provider interface used by the native search adapter
3. Add its dotted path to `SEARCH_PROVIDER_PATHS`

### Change the LLM Provider

Set:
- `LLM_API_BASE_URL`
- `LLM_API_KEY`
- model env var (e.g. `OPENROUTER_MODEL` or `NVIDIA_MODEL`)

### Continuous Learning

- Triggered after high confidence verifications
- Requires embeddings configuration (`EMBEDDING_*` vars)

## Debugging Pointers

- DB initialization and migration application live in `backend/app/core/db.py`.
- If the API boots but DB is missing, check `DB_REQUIRED` and `DATABASE_URL`.
