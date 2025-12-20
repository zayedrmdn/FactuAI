# FactuAI

**Full-stack AI fact-checking system** with 4-phase analysis pipeline.

---

## Highlights

**Backend:**
- **4-Phase Pipeline** (Intent → Strategy → Search → Pivot → Verify)
- **Multi-Hop Reasoning** (Pivot Loop for iterative research)
- **Strict Source Filtering** (blocks social media domains)
- **Continuous Learning** (RAG feedback loop with pgvector)
- Async-first FastAPI + Vertical Slice Architecture

**Frontend:**
- **Optimistic Pipeline UI** (real-time 4-phase progress visualization)
- Feature-based colocation (`features/ai-providers`, `search`, `analyze`, `history`)
- Next.js 16 + TypeScript + Tailwind CSS v4

---

## Quick Start

### Prerequisites
- Docker Desktop
- Python 3.11+ (backend)
- Node.js 20+ & pnpm (frontend)

### 1. Start Infrastructure

```bash
docker-compose up -d
```

### 2. Configure Environment

```bash
cp backend/.env.example backend/.env
# Edit backend/.env and set your API keys:
# - LLM_API_KEY (OpenRouter)
# - TAVILY_API_KEY (search provider)
```

### 3. Start Backend

```powershell
# Windows
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

```bash
# macOS/Linux
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

### 4. Start Frontend (Optional)

```bash
cd frontend
pnpm install
pnpm dev
```

**URLs:**
- Backend API: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
- Frontend: http://localhost:3000

---

## Documentation

📚 **Start here:** [docs/00-start-here.md](docs/00-start-here.md) (navigation hub)

### Quick Links

| Topic | Link |
|-------|------|
| **Rules & Governance** | [Constitution](docs/01-rules/constitution.md) · [AI Agents](docs/01-rules/agents.md) |
| **Setup** | [Quick Start](docs/02-setup/quickstart.md) · [Windows Guide](docs/02-setup/windows.md) · [Environment Vars](docs/02-setup/environment-vars.md) |
| **Architecture** | [Overview](docs/03-architecture/overview.md) · [Backend](docs/03-architecture/backend.md) · [Frontend](docs/03-architecture/frontend.md) · [Database](docs/03-architecture/database.md) |
| **Pipeline** | [4-Phase Flow](docs/04-pipeline/00-overview.md) |
| **Features** | [Continuous Learning](docs/05-features/) · [Source Filtering](docs/05-features/) |
| **Testing** | [Benchmark Claims](docs/06-testing/test-claims.md) |
| **API** | [Endpoints](docs/07-api/) · [Schemas](docs/07-api/) |
| **Product** | [Specifications](docs/product-specs.md) |
| **Logs** | [Changelog](docs/08-logs/changelog.md) |

---

## Tech Stack

**Backend:**
- FastAPI (async), SQLAlchemy (async), asyncpg
- PostgreSQL 16 + pgvector extension
- Redis (async)
- LangChain (`langchain-openai`) for LLM orchestration

**Frontend:**
- Next.js 16 (App Router, Turbopack)
- TypeScript, Tailwind CSS v4
- Zustand (state management)

**AI/ML:**
- OpenRouter (LLM provider) - Llama 3.3 70B default
- Tavily (search API)
- Embeddings: BAAI/bge-small-en-v1.5 (384-dim)

---

## AI Agent Onboarding

For AI coding assistants, read these in order:

1. [docs/01-rules/constitution.md](docs/01-rules/constitution.md) (engineering rules - REQUIRED)
2. [docs/01-rules/agents.md](docs/01-rules/agents.md) (AI agent quick reference)
3. [docs/03-architecture/overview.md](docs/03-architecture/overview.md) (system design)
4. [docs/04-pipeline/00-overview.md](docs/04-pipeline/00-overview.md) (pipeline details)

---

## Development

**Backend tests:**
```bash
cd backend
pytest
```

**Frontend tests:**
```bash
cd frontend
pnpm test
```

**Build frontend:**
```bash
cd frontend
pnpm build
```

---

## License

MIT License - see [LICENSE](LICENSE)

---

## Support

- **Documentation:** [docs/00-start-here.md](docs/00-start-here.md)
- **Troubleshooting:** [docs/02-setup/troubleshooting.md](docs/02-setup/troubleshooting.md)
- **Bug Reports:** [docs/08-logs/bugs/template.md](docs/08-logs/bugs/template.md)
