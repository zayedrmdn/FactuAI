# Model Override (Frontend Selection)

How users can override the backend's default LLM model.

---

## Overview

The frontend allows users to select which LLM model to use for verification, overriding the backend's default (`OPENROUTER_MODEL`).

**Use Cases:**
- Testing different models for accuracy
- Cost optimization (cheaper models)
- Speed optimization (faster models)
- Specialized models for specific domains

---

## How It Works

### Frontend Model Selection

**Location:** `frontend/src/features/ai-providers/`

1. User selects model from dropdown
2. Selection stored in Zustand state
3. Sent to backend in analyze request

```typescript
// frontend/src/features/ai-providers/stores/selection.ts
const { selection } = useAIStore();

// Send to backend
const response = await fetch('/api/analyze', {
  method: 'POST',
  body: JSON.stringify({
    text: input,
    model_id: selection.modelId  // ← Override
  })
});
```

---

## Backend Handling

### Model ID Resolution

**Location:** `backend/app/features/analyze/router.py`

```python
@router.post("/api/analyze")
async def analyze(request: AnalyzeRequest):
    # Use frontend override if provided, else default
    model_id = request.model_id or settings.openrouter_model
    
    # Verification uses this model
    result = await verification_service.verify(
        claim=claim,
        evidence=evidence,
        model_id=model_id
    )
```

**Schema:**
```python
class AnalyzeRequest(BaseModel):
    text: str
    model_id: Optional[str] = None  # Frontend override
```

---

## Available Models

### OpenRouter Models (Recommended)

**Location:** `frontend/src/features/ai-providers/registry.ts`

| Model | Provider | Cost | Speed | Use Case |
|-------|----------|------|-------|----------|
| `meta-llama/llama-3.3-70b-instruct` | Meta | Low | Fast | **Default** - Best balance |
| `meta-llama/llama-3.1-8b-instruct` | Meta | Very Low | Very Fast | Quick checks |
| `anthropic/claude-3.5-sonnet` | Anthropic | High | Medium | Premium accuracy |
| `google/gemini-pro` | Google | Medium | Fast | Alternative |

### Model Registry Format

```typescript
// frontend/src/features/ai-providers/registry.ts
export const modelRegistry = [
  {
    id: 'openrouter-llama-3.3-70b',
    displayName: 'Llama 3.3 70B (Recommended)',
    provider: 'openrouter',
    modelId: 'meta-llama/llama-3.3-70b-instruct',
    defaultTemperature: 0.1,
    isRecommended: true,
    isDefault: true,
  },
  // ... more models
];
```

---

## Frontend UI

### Model Selector Component

**Location:** `frontend/src/features/ai-providers/components/`

```tsx
import { useAIStore, modelRegistry } from '@/features/ai-providers';

function ModelSelector() {
  const { selection, setModelId } = useAIStore();
  
  return (
    <select value={selection.modelId} onChange={(e) => setModelId(e.target.value)}>
      {modelRegistry.map((model) => (
        <option key={model.id} value={model.id}>
          {model.displayName}
          {model.isRecommended && ' ⭐'}
        </option>
      ))}
    </select>
  );
}
```

---

## Session Overrides

Users can also override model parameters:

```typescript
interface SessionOverrides {
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
  system_prompt?: string;
}

const { selection } = useAIStore();
// selection.sessionOverrides contains user adjustments
```

**Sent to backend:**
```json
{
  "text": "...",
  "model_id": "meta-llama/llama-3.3-70b-instruct",
  "temperature": 0.1,
  "max_tokens": 4096
}
```

---

## Backend Configuration Sync

### System Config API

**Endpoint:** `GET /api/system/config`

**Purpose:** Frontend fetches backend defaults on startup

```python
# backend/app/features/system/router.py
@router.get("/api/system/config")
async def get_config():
    return {
        "models": {
            "default_reasoning": settings.openrouter_model,
            "default_intent": settings.intent_llm_model,
            "provider": "openrouter",
            "api_base_url": settings.llm_api_base_url
        },
        "features": {
            "tavily_enabled": bool(settings.tavily_api_key),
            "learning_enabled": bool(settings.embedding_api_base_url)
        }
    }
```

**Frontend Usage:**
```typescript
// On app load
const config = await fetch('/api/system/config').then(r => r.json());

// Set frontend defaults based on backend config
useAIStore.setState({
  selection: {
    modelId: config.models.default_reasoning,
    provider: config.models.provider
  }
});
```

---

## Testing Different Models

### Quick Comparison

1. **Select Model:** Llama 3.1 8B (fast)
2. **Run Claim:** "The Earth is flat"
3. **Note:** Latency ~8s, verdict FALSE

4. **Select Model:** Claude 3.5 Sonnet (accurate)
5. **Run Same Claim:** "The Earth is flat"
6. **Compare:** Latency ~12s, verdict FALSE, but reasoning is more detailed

---

## Limitations

### 1. Model Must Be OpenAI-Compatible

The backend uses OpenAI-compatible API format:
```python
response = await client.chat.completions.create(
    model=model_id,
    messages=[{"role": "user", "content": prompt}]
)
```

**Supported:** OpenRouter, OpenAI, Azure OpenAI, Together AI, etc.

### 2. API Key Must Have Access

If user selects a model their API key doesn't have access to, backend will return error:

```json
{
  "detail": "Model not available with current API key"
}
```

---

## Code Pointers

- Frontend store: `frontend/src/features/ai-providers/stores/selection.ts`
- Model registry: `frontend/src/features/ai-providers/registry.ts`
- Backend handling: `backend/app/features/analyze/router.py`
- System config API: `backend/app/features/system/router.py`

---

See also:
- [Phase 4: Verification](../04-pipeline/05-verification.md) - How models are used
- [Frontend Architecture](../03-architecture/frontend.md) - Feature modules
- [Backend Architecture](../03-architecture/backend.md) - API handling
