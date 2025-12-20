# Windows Setup (Local Dev)

This guide is for running the **native async FastAPI backend** locally on Windows.

## Prerequisites

- **Docker Desktop** (recommended) for Postgres + Redis
- **Python 3.11+**
- **Git**

Optional (frontend): Node.js 18+.

## 1) Start Infrastructure (Postgres + Redis)

From repo root:

```powershell
docker-compose up -d
```

Defaults:
- Postgres: `localhost:5433` (container `5432` → host `5433`)
- Redis: `localhost:6379`

## 2) Backend Environment Variables

Create a `.env` at the repo root (or set env vars in your shell). Suggested minimal config:

```dotenv
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/factuai
DB_REQUIRED=true
DB_RUN_MIGRATIONS=true

# Redis (optional)
REDIS_URL=redis://localhost:6379/0
REDIS_REQUIRED=false

# Search providers (optional but recommended)
TAVILY_API_KEY=
NEWSAPI_API_KEY=

# LLM verification (optional; required for real verification)
LLM_API_BASE_URL=https://openrouter.ai/api/v1
LLM_API_KEY=
OPENROUTER_MODEL=anthropic/claude-3-haiku

# Continuous learning embeddings (optional; required for embedding writeback)
EMBEDDING_API_BASE_URL=
EMBEDDING_API_KEY=
EMBEDDING_MODEL=BAAI/bge-small-en-v1.5
EMBEDDING_DIM=384
LEARNING_CONFIDENCE_THRESHOLD=0.85
```

Notes:
- The backend accepts `DATABASE_URL` or `DB_URI`. If you set `DB_URI=postgresql://...`, it is normalized to `postgresql+asyncpg://...`.
- In dev you can set `DB_REQUIRED=false` to let the API boot without Postgres.

## 3) Run the Backend

Using the included venv (already present in this workspace):

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
pip install -r requirements-dev.txt
uvicorn app.main:app --reload

Note:
- Verification uses LangChain (`langchain-openai`) with async invocation and structured output parsing against your configured OpenAI-compatible endpoint.
```

Open:
- API: http://127.0.0.1:8000
- Health: http://127.0.0.1:8000/health

## 4) Run Tests

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pytest -q
```

## Troubleshooting

- **Port conflict**: Postgres host port is `5433`. If you already have something on 5433, change `docker-compose.yml` and update `DATABASE_URL`.
- **DB not ready yet**: `docker-compose ps` should show `healthy` for `factuai-db`.
- **No external keys**: You can still run the API; some features will return empty evidence or skip verification depending on configuration.
