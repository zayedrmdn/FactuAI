# FactuAI Backend (FastAPI)

## Run (Local Dev)

Recommended: start Postgres + Redis via Docker.

From repo root:

```powershell
docker-compose up -d
```

Then:

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload
```

Health endpoint:
- http://127.0.0.1:8000/health

## Database

- Default dev DB URL (matches docker-compose host port mapping):
  - `postgresql+asyncpg://postgres:postgres@localhost:5433/factuai`
- Migrations are SQL files in `backend/migrations/*.sql`.
- Migrations run automatically on startup if `DB_RUN_MIGRATIONS=true`.

## Environment Variables

Configured in `app/core/settings.py`.

Core:
- `DATABASE_URL` (or `DB_URI`) – Postgres connection string
- `DB_REQUIRED` – `true|false` (fail startup if DB is unavailable)
- `DB_RUN_MIGRATIONS` – `true|false`
- `REDIS_URL`
- `REDIS_REQUIRED`

LLM verification (LangChain over OpenAI-compatible endpoints):
- `LLM_API_BASE_URL` (fallbacks: `OPENAI_BASE_URL`)
- `LLM_API_KEY` (fallbacks: `OPENROUTER_API_KEY`, `OPENAI_API_KEY`)
- `LLM_PROVIDER` (e.g. `nvidia`)
- `OPENROUTER_MODEL`
- `NVIDIA_MODEL`

Notes:
- The default verifier adapter uses `langchain-openai` (`ChatOpenAI`) with async invocation (`ainvoke`) and strict structured output parsing.

Embeddings (continuous learning):
- `EMBEDDING_API_BASE_URL`
- `EMBEDDING_API_KEY`
- `EMBEDDING_MODEL`
- `EMBEDDING_DIM`
- `LEARNING_CONFIDENCE_THRESHOLD`
- `LEARNING_MAX_EVIDENCE`

Search:
- `SEARCH_ADAPTER`
- `SEARCH_PROVIDER_PATHS` (CSV dotted paths)
- `TAVILY_API_KEY`
- `NEWSAPI_API_KEY`
- `EVIDENCE_CACHE_TTL_SECONDS`

## Tests

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pytest -q
```
