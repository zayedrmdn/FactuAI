# Quick Setup Guide

Get FactuAI running locally in 5 minutes.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed
- Python 3.11+ (for backend)
- Node.js 20+ and pnpm (for frontend)

---

## Step 1: Clone & Setup Infrastructure

```bash
# Clone repository
git clone https://github.com/yourusername/FactuAI.git
cd FactuAI

# Start Docker services (PostgreSQL + Redis)
docker-compose up -d
```

**What this does:**
- Starts PostgreSQL 16 (with pgvector extension) on port `5433`
- Starts Redis on port `6379`

---

## Step 2: Configure Environment

```bash
# Copy environment template
cp backend/.env.example backend/.env
```

**Minimal required configuration:**

Edit `backend/.env` and set:
```bash
# Database (already configured for local Docker)
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/factuai

# LLM Provider (required for fact-checking)
LLM_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=meta-llama/llama-3.3-70b-instruct

# Search Provider (optional but recommended)
TAVILY_API_KEY=your_tavily_api_key
```

> [!TIP]
> Get your OpenRouter API key at: https://openrouter.ai/keys  
> Get your Tavily API key at: https://tavily.com

See [environment-vars.md](environment-vars.md) for full configuration reference.

---

## Step 3: Start Backend

**Windows PowerShell:**

```powershell
cd backend
.\venv\Scripts\Activate.ps1
pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

**macOS/Linux:**

```bash
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements-core.txt
uvicorn app.main:app --reload
```

**Verify backend:**
- API: http://127.0.0.1:8000
- Health check: http://127.0.0.1:8000/health
- Should see: `{"status": "healthy", "database": "connected"}`

---

## Step 4: Start Frontend (Optional)

```bash
cd frontend
pnpm install
pnpm dev
```

**Access frontend:**
- Dashboard: http://localhost:3000

---

## Verify Installation

### Test the analysis pipeline:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The Earth is flat"}'
```

**Expected response:**
```json
{
  "request_id": "...",
  "claims": [
    {
      "claim_text": "The Earth is flat",
      "verdict": "false",
      "confidence": 0.95,
      "reasoning": "...",
      "evidence": [...]
    }
  ]
}
```

---

## Common Issues

### Database connection failed
- Ensure Docker containers are running: `docker ps`
- Check PostgreSQL port: `5433` (not default `5432`)

### LLM errors (503 Service Unavailable)
- Verify `LLM_API_KEY` is set in `backend/.env`
- Check API key is valid on OpenRouter dashboard

### Frontend can't connect to backend
- Ensure backend is running on port `8000`
- Check `NEXT_PUBLIC_API_URL` in `frontend/.env.local` (default: `http://127.0.0.1:8000`)

See [troubleshooting.md](troubleshooting.md) for more solutions.

---

## Next Steps

- **Architecture**: Read [Architecture Overview](../03-architecture/overview.md)
- **Pipeline**: Understand the [4-Phase Analysis Flow](../04-pipeline/00-overview.md)
- **Testing**: Run [Benchmark Claims](../06-testing/test-claims.md)
- **Deployment**: Coming soon

---

**Platform-specific guides:**
- [Windows Setup](windows.md) - Detailed setup for Windows developers
