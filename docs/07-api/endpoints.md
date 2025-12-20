# API Endpoints Reference

Complete reference for FactuAI API endpoints.

---

## Base URL

**Local Development:** `http://127.0.0.1:8000`  
**Production:** (configured via deployment)

---

## Endpoints

### Health Check

**Purpose:** Liveness check for infrastructure monitoring

```http
GET /health
```

**Response (200 OK):**
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "llm_provider": "reachable"
}
```

**Response (503 Service Unavailable):**
```json
{
  "status": "unhealthy",
  "database": "disconnected",
  "llm_provider": "unreachable"
}
```

---

### System Configuration

**Purpose:** Expose backend configuration to frontend

```http
GET /api/system/config
```

**Response (200 OK):**
```json
{
  "models": {
    "default_reasoning": "meta-llama/llama-3.3-70b-instruct",
    "default_intent": "meta-llama/llama-3.3-70b-instruct",
    "provider": "openrouter",
    "api_base_url": "https://openrouter.ai/api/v1"
  },
  "features": {
    "tavily_enabled": true,
    "learning_enabled": true,
    "rate_limit_enabled": false,
    "preflight_checks_enabled": true
  }
}
```

**Use Case:** Frontend fetches this on startup to sync with backend defaults.

---

### fact-Check Analysis

**Purpose:** Multi-claim fact-checking with evidence gathering

```http
POST /api/analyze
Content-Type: application/json
```

** Request Body:**
```json
{
  "text": "The Earth is flat and vaccines cause autism",
  "options": {
    "use_search": true,
    "verification_enabled": true
  },
  "model_id": "meta-llama/llama-3.3-70b-instruct"
}
```

**Request Schema:**
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `text` | string | ✅ Yes | User input to fact-check |
| `options.use_search` | boolean | No | Enable search phase (default: true) |
| `options.verification_enabled` | boolean | No | Enable verification (default: true) |
| `model_id` | string | No | Override default LLM model |

**Response (200 OK):**
```json
{
  "request_id": "550e8400-e29b-41d4-a716-446655440000",
  "model_used": "meta-llama/llama-3.3-70b-instruct",
  "latency_ms": 8742,
  "claims": [
    {
      "claim_text": "The Earth is flat",
      "verdict": "FALSE",
      "confidence": 0.98,
      "reasoning": "Overwhelming scientific evidence confirms Earth is an oblate spheroid...",
      "evidence": [
        {
          "snippet": "NASA satellite images show Earth's curvature...",
          "source_url": "https://nasa.gov/...",
          "source_title": "NASA Earth Observatory",
          "relevance_score": 0.95
        }
      ]
    },
    {
      "claim_text": "Vaccines cause autism",
      "verdict": "FALSE",
      "confidence": 0.97,
      "reasoning": "Multiple large-scale studies found no causal link...",
      "evidence": [...]
    }
  ]
}
```

**Response (400 Bad Request):**
```json
{
  "detail": "No claims extracted from input"
}
```

**Response (503 Service Unavailable):**
```json
{
  "detail": "LLM provider is unreachable"
}
```

---

## Error Responses

### Standard Error Format

```json
{
  "detail": "Error message here"
}
```

### Common Error Codes

| Code | Meaning | Cause |
|------|---------|-------|
| `400` | Bad Request | Invalid input, no claims extracted |
| `422` | Validation Error | Schema validation failed |
| `429` | Too Many Requests | Rate limit exceeded |
| `503` | Service Unavailable | LLM or database unreachable |
| `500` | Internal Server Error | Unexpected server error |

---

## Rate Limiting

**Status:** Not currently implemented (planned feature)

**Future:**
- Rate limit: 10 requests/minute per IP
- Header: `X-RateLimit-Remaining`
- Response: 429 with `Retry-After` header

---

## Authentication

**Status:** Not currently implemented (public API)

**Future:** API key authentication for production deployments

```http
POST /api/analyze
Authorization: Bearer YOUR_API_KEY
```

---

## Webhook Support

**Status:** Not implemented

**Planned:** Webhook for long-running analyses

---

## CORS

**Current:** Allows all origins (`*`) for development

**Production:** Configure allowed origins via environment:
```bash
CORS_ORIGINS=https://yourdomain.com
```

---

## Request Examples

### cURL

```bash
curl -X POST http://127.0.0.1:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The Earth is flat",
    "model_id": "meta-llama/llama-3.3-70b-instruct"
  }'
```

### JavaScript (fetch)

```javascript
const response = await fetch('http://127.0.0.1:8000/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'The Earth is flat',
    model_id: 'meta-llama/llama-3.3-70b-instruct'
  })
});

const data = await response.json();
console.log(data.claims);
```

### Python (httpx)

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        'http://127.0.0.1:8000/api/analyze',
        json={
            'text': 'The Earth is flat',
            'model_id': 'meta-llama/llama-3.3-70b-instruct'
        }
    )
    data = response.json()
    print(data['claims'])
```

---

## OpenAPI Documentation

**Interactive Docs:** `http://127.0.0.1:8000/docs` (Swagger UI)  
**ReDoc:** `http://127.0.0.1:8000/redoc`  
**OpenAPI JSON:** `http://127.0.0.1:8000/openapi.json`

---

See also:
- [API Schemas](schemas.md) - Request/response types
- [Pipeline Overview](../04-pipeline/00-overview.md) - How analysis works
- [Architecture: Backend](../03-architecture/backend.md) - API implementation
