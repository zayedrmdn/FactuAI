# FactuAI

FactuAI is a full-stack fact-checking system with a **native async FastAPI backend** and a Next.js frontend.

**Backend highlights**
- **4-Phase Analysis Pipeline** (Strategist → Parallel Search → Pivot → Verification)
- **Multi-Hop Reasoning** (Pivot Loop detects new concepts and executes follow-up research)
- **Adversarial Query Generation** (3 multi-angle queries per claim: Factual, Hoax, Scientific)
- Vertical Slice Architecture (`backend/app/features/*`)
- Tiered Intelligence (LLM-based Intent Extraction)
- **Strict Source Filtering** (Tavily Gatekeeper blocks social media)
- Fail Fast Pre-flight Checks (validates LLM connectivity before processing)
- PostgreSQL + **pgvector** for continuous learning (RAG feedback loop with 0.80 similarity threshold)
- Redis for caching

**Frontend highlights**
- Feature-Based Colocation (`frontend/src/features/*`)
- Feature modules: `ai-providers/`, `search/`, `analyze/`, `history/`

## Tech Stack

- Backend: FastAPI, SQLAlchemy (async), asyncpg, PostgreSQL+pgvector, Redis, httpx, LangChain (`langchain-openai`) for LLM verification
- Frontend: Next.js (App Router), TypeScript, Tailwind

## Quick Start (Docker)

Assumes you have Docker Desktop installed.

```bash
docker-compose up -d
```

Backend (Windows PowerShell):

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

Open:
- API: http://127.0.0.1:8000
- Health: http://127.0.0.1:8000/health

## Documentation (Portal Pattern)

- Start here (new dev): docs/SETUP_WINDOWS.md
- Deep dive (architecture): docs/ARCHITECTURE.md
- Rules (required for humans + AI agents): CONSTITUTION.md
- Backend specifics (env vars, tests): backend/README.md
- AI coding source of truth: AGENTS.md

## AI Agent Onboarding

For a fresh chat, feed the agent:

- docs/AI_CONTEXT.md
- CONSTITUTION.md
- AGENTS.md
