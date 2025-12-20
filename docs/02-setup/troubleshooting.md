# Troubleshooting Guide

Common issues and solutions for FactuAI.

---

## Database Issues

### ❌ `Connection refused` or `could not connect to server`

**Symptoms:**
```
sqlalchemy.exc.OperationalError: connection refused
```

**Solutions:**

1. **Check Docker is running:**
   ```bash
   docker ps
   ```
   Should show `factuai-postgres` container running.

2. **Verify port 5433 is correct:**
   ```bash
   # In backend/.env
   DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/factuai
   ```
   Note: Port is `5433` (not default `5432`) due to Docker mapping.

3. **Restart Docker containers:**
   ```bash
   docker-compose down
   docker-compose up -d
   ```

### ❌ `relation "verifications" does not exist`

**Symptoms:**
```
psql.errors.UndefinedTable: relation "verifications" does not exist
```

**Solutions:**

1. **Ensure migrations auto-run:**
   ```bash
   # In backend/.env
   DB_RUN_MIGRATIONS=true
   ```

2. **Manually apply migrations:**
   ```bash
   cd backend
   psql -h localhost -p 5433 -U postgres -d factuai -f migrations/v4_0_001_core.sql
   ```

---

## LLM / API Issues

### ❌ `503 Service Unavailable` on `/api/analyze`

**Symptoms:**
```json
{"detail": "LLM provider is unreachable"}
```

**Solutions:**

1. **Verify API key is set:**
   ```bash
   # In backend/.env
   LLM_API_KEY=sk-or-v1-...  # Should not be empty
   ```

2. **Test API key manually:**
   ```bash
   curl https://openrouter.ai/api/v1/models \
     -H "Authorization: Bearer $LLM_API_KEY"
   ```

3. **Check network/firewall:**
   - Ensure `openrouter.ai` is not blocked
   - Try from browser: https://openrouter.ai/docs

### ❌ `No claims extracted from input`

**Symptoms:**
Frontend shows: "No verifiable claims could be extracted..."

**Solutions:**

1. **Input must contain factual statements:**
   
   ❌ Bad: "I think maybe the sky is blue?"  
   ✅ Good: "The sky is blue because of Rayleigh scattering."

2. **Try benchmark claims:**
   See [../06-testing/test-claims.md](../06-testing/test-claims.md) for known working examples.

---

## Frontend Issues

### ❌ `Failed to fetch` or `CORS error`

**Symptoms:**
```
TypeError: Failed to fetch
Access to fetch at 'http://127.0.0.1:8000/api/analyze' blocked by CORS policy
```

**Solutions:**

1. **Ensure backend is running:**
   ```bash
   curl http://127.0.0.1:8000/health
   ```

2. **Check frontend env var:**
   ```bash
   # In frontend/.env.local
   NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
   ```

3. **Clear Next.js cache:**
   ```bash
   cd frontend
   rm -rf .next
   pnpm dev
   ```

### ❌ `/api/analyze/api/system/config` (Malformed URL)

**Symptoms:**
```
GET /api/analyze/api/system/config → 404 Not Found
```

**Cause:** `NEXT_PUBLIC_API_URL` includes `/api/analyze` suffix (legacy format).

**Solution:**

```bash
# In frontend/.env.local - use base URL only
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000  # ✅ Correct
# NOT: http://127.0.0.1:8000/api/analyze   # ❌ Wrong
```

The frontend code now auto-normalizes this, but it's better to fix the env var.

---

## Search / RAG Issues

### ❌ `No search results returned`

**Symptoms:**
Analysis completes but `evidence` array is empty.

**Solutions:**

1. **Verify Tavily API key:**
   ```bash
   # In backend/.env
   TAVILY_API_KEY=tvly-...
   ```

2. **Check search provider is enabled:**
   ```bash
   # In backend/.env
   SEARCH_PROVIDER_PATHS=backend.app.features.search.providers.tavily.TavilyProvider
   ```

3. **Test Tavily API manually:**
   ```bash
   curl https://api.tavily.com/search \
     -H "Content-Type: application/json" \
     -d '{"api_key":"tvly-...","query":"test"}'
   ```

### ❌ `RAG retrieval not working`

**Symptoms:**
No `[INTERNAL MEMORY]` results in search phase.

**Solutions:**

1. **Ensure pgvector extension is installed:**
   ```sql
   -- Connect to database
   psql -h localhost -p 5433 -U postgres -d factuai
   
   -- Check extension
   \dx
   -- Should show 'vector' extension
   ```

2. **Verify embeddings are being stored:**
   ```sql
   SELECT COUNT(*) FROM claims WHERE claim_embedding IS NOT NULL;
   ```
   
   If count is 0, no embeddings have been generated yet. Run some high-confidence fact-checks first.

3. **Check embedding service is running:**
   ```bash
   curl $EMBEDDING_API_BASE_URL/embeddings \
     -H "Content-Type: application/json" \
     -d '{"input":"test","model":"BAAI/bge-small-en-v1.5"}'
   ```

---

## Performance Issues

### ❌ Analysis takes > 60 seconds

**Expected latency:**
- Basic claims: < 10s
- Complex claims (with pivot): < 30s

**Solutions:**

1. **Check search provider performance:**
   - Tavily is faster than legacy providers
   - Ensure only Tavily is enabled

2. **Reduce parallel search queries:**
   Frontend can limit search results per provider (default: 10)

3. **Use faster LLM model:**
   ```bash
   # In backend/.env
   OPENROUTER_MODEL=meta-llama/llama-3.1-8b-instruct  # Faster but less accurate
   ```

---

## Development Environment

### ❌ `pnpm: command not found`

**Solution:**
```bash
npm install -g pnpm
```

### ❌ `pytest: command not found`

**Solution:**
```bash
cd backend
pip install -r requirements-dev.txt
```

### ❌ Port 8000 already in use

**Solution:**
```bash
# Find process using port 8000
lsof -i :8000  # macOS/Linux
netstat -ano | findstr :8000  # Windows

# Kill process or use different port
uvicorn app.main:app --reload --port 8001
```

---

## Still Stuck?

1. **Check logs:**
   - Backend: `uvicorn` console output
   - Frontend: Browser console (F12)
   - Docker: `docker logs factuai-postgres`

2. **Enable debug mode:**
   ```bash
   # In backend/.env
   LOG_LEVEL=DEBUG
   ```

3. **Verify system state:**
   ```bash
   curl http://127.0.0.1:8000/health
   ```

4. **Review recent changes:**
   See [../08-logs/changelog.md](../08-logs/changelog.md)

5. **Report bug:**
   Use [../08-logs/bugs/template.md](../08-logs/bugs/template.md)

---

**See also:**
- [Quick Setup](quickstart.md)
- [Environment Variables](environment-vars.md)
- [Architecture Overview](../03-architecture/overview.md)
