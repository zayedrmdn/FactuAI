# Backend Architecture

FastAPI backend with Vertical Slice Architecture, async-first patterns, and dependency injection.

---

## Directory Structure

```
backend/app/
├── main.py                  # FastAPI app entry point
├── core/                    # Core infrastructure
│   ├── settings.py          # Environment config
│   ├── container.py         # DI container
│   ├── deps.py              # FastAPI dependencies
│   ├── db.py                # Database initialization
│   └── health.py            # Pre-flight checks
├── contracts/               # Shared types (cross-feature)
├── features/               # Vertical slices
│   ├── analyze/            # Full orchestration
│   │   ├── router.py       # API endpoints
│   │   ├── service.py      # Business logic
│   │   └── schemas.py      # Pydantic models
│   ├── intent/             # Claim extraction
│   │   ├── ports.py        # Interface
│   │   └── adapters/
│   │       └── llm.py      # LLMIntentAdapter
│   ├── search/             # Search providers
│   │   ├── ports.py
│   │   └── providers/
│   │       └── tavily.py
│   ├── system/            # Config API
│   └── verification/       # Verdict generation
│       ├── ports.py
│       └── adapters/
│           └── native.py
└── infrastructure/         # Shared utilities
    └── extraction/         # Web scraping, OCR, etc.
```

---

## Vertical Slice Architecture (VSA)

Each feature owns its **complete vertical stack**:

### Feature Anatomy

```python
backend/app/features/analyze/
├── router.py      # API boundary (thin)
├── service.py     # Orchestration logic
├── schemas.py     # Request/response models
├── ports.py       # Interfaces (if needed)
└── adapters/      # Implementations (if needed)
```

### Key Rules

1. **Route handlers are thin**: Validate → Call service → Return
   ```python
   # router.py
   @router.post("/api/analyze")
   async def analyze(request: AnalyzeRequest):
       result = await analyze_service.execute(request)
       return result
   ```

2. **Features don't import features**: Use `contracts/` for shared types
   ```python
   # ✅ Good
   from backend.app.contracts.search import SearchResult
   
   # ❌ Bad
   from backend.app.features.search.schemas import SearchResult
   ```

3. **Orchestration stays in service layer**: Business logic, not in routes

---

## Dependency Injection (OCP)

### Container Configuration

**File:** `backend/app/core/container.py`

```python
# Adapters are registered via dotted paths from settings
intent_adapter_path = settings.intent_adapter
search_adapter_path = settings.search_adapter
verifier_adapter_path = settings.verifier_adapter
```

### Environment-Driven Configuration

**File:** `backend/app/core/settings.py`

```python
class Settings(BaseSettings):
    # DI bindings
    intent_adapter: str = "backend.app.features.intent.adapters.llm.LLMIntentAdapter"
    search_adapter: str = "backend.app.features.search.adapters.native.NativeSearchAdapter"
    verifier_adapter: str = "backend.app.features.verification.adapters.native.NativeVerificationAdapter"
    
    # Search provider paths (comma-separated)
    search_provider_paths: str = "backend.app.features.search.providers.tavily.TavilyProvider"
```

### Extending via OCP

To add a new search provider:

1. Create provider class in `backend/app/features/search/providers/my_provider.py`
2. Add dotted path to `SEARCH_PROVIDER_PATHS`:
   ```bash
   SEARCH_PROVIDER_PATHS=backend.app.features.search.providers.tavily.TavilyProvider,backend.app.features.search.providers.my_provider.MyProvider
   ```
3. No orchestrator changes needed ✅

See [../05-features/search-providers.md](../05-features/search-providers.md) for guide.

---

## Async Patterns

### All I/O is Truly Async

```python
# ✅ Correct: Native async
import httpx
async with httpx.AsyncClient() as client:
    response = await client.get(url)

# ✅ Correct: Async DB
from sqlalchemy.ext.asyncio import async_sessionmaker
async with session_factory() as session:
    result = await session.execute(query)

# ✅ Correct: Async Redis
import redis.asyncio as redis
async with redis.from_url(url) as conn:
    await conn.set("key", "value")

# ❌ Wrong: Sync wrappers
import requests  # Don't use in FastAPI
response = requests.get(url)  # Blocks event loop
```

### Parallel Execution

```python
import asyncio

# Execute search queries in parallel
results = await asyncio.gather(
    tavily_search(query1),
    tavily_search(query2),
    tavily_search(query3),
    rag_search(query1)
)
```

---

## Database Layer

### Migrations

**Location:** `backend/migrations/*.sql`

- Applied on startup if `DB_RUN_MIGRATIONS=true`
- Idempotent migrations (use `CREATE TABLE IF NOT EXISTS`, etc.)
- Managed in `backend/app/core/db.py`

### Session Management

```python
from backend.app.core.deps import get_db

@router.post("/api/analyze")
async def analyze(
    request: AnalyzeRequest,
    db: AsyncSession = Depends(get_db)  # Injected session
):
    # Use db session
    result = await db.execute(query)
```

### pgvector Integration

```python
from sqlalchemy import Column, Integer, Text
from pgvector.sqlalchemy import Vector

class Claim(Base):
    __tablename__ = "claims"
    
    id = Column(Integer, primary_key=True)
    claim_text = Column(Text, nullable=False)
    claim_embedding = Column(Vector(384))  # pgvector type
```

See [database.md](database.md) for schema details.

---

## Error Handling

### Pre-flight Checks

```python
# backend/app/core/health.py
async def check_llm_provider():
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{settings.llm_api_base_url}/models")
            return response.status_code == 200
    except:
        return False
```

### Graceful Degradation

```python
# Learning errors fail safely
try:
    await store_embeddings(claim)
except Exception as e:
    logger.error(f"Learning failed: {e}")
    # Don't crash main flow
```

---

## Testing

### Pytest Setup

```bash
cd backend
pip install -r requirements-dev.txt
pytest
```

### Test Structure

```
backend/tests/
├── test_intent.py
├── test_search.py
├── test_verification.py
└── test_analyze.py
```

See [../06-testing/backend-tests.md](../06-testing/backend-tests.md).

---

## Entry Points Explained

| File | Purpose |
|------|---------|
| `main.py` | FastAPI app, routers, CORS, lifespan |
| `core/settings.py` | All environment variables |
| `core/container.py` | DI registration |
| `core/deps.py` | FastAPI dependency providers |
| `core/db.py` | Async engine, session factory, migration runner |
| `core/health.py` | Pre-flight checks for `/api/analyze` |

---

## Best Practices

1. **Keep routes thin**:Routes should only validate, call service, and return - no business logic

2. **Respect feature boundaries**: Never import one feature from another

3. **Fail fast**: Validate infrastructure before expensive operations

4. **Async all the way**: No sync wrappers, no blocking calls

5. **DI over hardcoding**: Use env vars + container, not hardcoded implementations

---

See also:
- [Overview](overview.md) - High-level architecture
- [Frontend Architecture](frontend.md) - Next.js patterns
- [Database Schema](database.md) - Postgres + pgvector
