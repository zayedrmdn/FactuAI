# API Schemas Reference

Pydantic models and TypeScript types for FactuAI API.

---

## Backend Schemas (Pydantic)

**Location:** `backend/app/features/analyze/schemas.py`

---

### AnalyzeRequest

**Purpose:** Request body for `POST /api/analyze`

```python
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any

class AnalyzeOptions(BaseModel):
    use_search: bool = True
    verification_enabled: bool = True

class AnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, description="User input to fact-check")
    options: AnalyzeOptions = Field(default_factory=AnalyzeOptions)
    model_id: Optional[str] = Field(None, description="Override default LLM model")
    
    # Optional overrides
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
```

**Example:**
```json
{
  "text": "The Earth is flat",
  "options": {
    "use_search": true,
    "verification_enabled": true
  },
  "model_id": "meta-llama/llama-3.3-70b-instruct",
  "temperature": 0.1
}
```

---

### AnalyzeResponse

**Purpose:** Response from `POST /api/analyze`

```python
from pydantic import BaseModel, Field
from typing import List
from uuid import UUID

class EvidenceItem(BaseModel):
    snippet: str
    source_url: str
    source_title: str
    relevance_score: float = Field(ge=0.0, le=1.0)

class ClaimResult(BaseModel):
    claim_text: str
    verdict: str  # TRUE, FALSE, MIXED, MOSTLY_TRUE, MOSTLY_FALSE, UNVERIFIABLE
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    evidence: List[EvidenceItem] = []

class AnalyzeResponse(BaseModel):
    request_id: UUID
    model_used: str
    latency_ms: int
    claims: List[ClaimResult]
```

**Example:**
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
      "reasoning": "Scientific consensus confirms...",
      "evidence": [
        {
          "snippet": "NASA confirms...",
          "source_url": "https://nasa.gov/...",
          "source_title": "NASA",
          "relevance_score": 0.95
        }
      ]
    }
  ]
}
```

---

### Verdict Types

```python
from enum import Enum

class Verdict(str, Enum):
    TRUE = "TRUE"
    MOSTLY_TRUE = "MOSTLY_TRUE"
    MIXED = "MIXED"
    MOSTLY_FALSE = "MOSTLY_FALSE"
    FALSE = "FALSE"
    UNVERIFIABLE = "UNVERIFIABLE"
```

---

## Frontend Types (TypeScript)

**Location:** `frontend/src/types/dashboard/factcheck.ts`

---

### FactCheckApiResult

**Purpose:** Claim result from API (maps to `ClaimResult` from backend)

```typescript
export interface EvidenceItem {
  snippet: string;
  source_url: string;
  source_title: string;
  relevance_score: number;
}

export type Verdict = 
  | 'TRUE' 
  | 'MOSTLY_TRUE' 
  | 'MIXED' 
  | 'MOSTLY_FALSE' 
  | 'FALSE' 
  | 'UNVERIFIABLE';

export interface FactCheckApiResult {
  claim_text: string;
  verdict: Verdict;
  confidence: number;
  reasoning: string;
  evidence: EvidenceItem[];
}
```

---

### FactCheckResult

**Purpose:** Frontend-transformed result (adds UI state)

```typescript
export interface FactCheckResult {
  id: string;  // Generated on frontend
  claimText: string;
  verdict: Verdict;
  confidence: number;
  reasoning: string;
  evidence: EvidenceItem[];
  expanded?: boolean;  // UI state
  timestamp?: string;  // Added on frontend
}

// Mapping function
export function mapApiResultToFactCheckResult(
  apiResult: FactCheckApiResult
): FactCheckResult {
  return {
    id: crypto.randomUUID(),
    claimText: apiResult.claim_text,
    verdict: apiResult.verdict,
    confidence: apiResult.confidence,
    reasoning: apiResult.reasoning,
    evidence: apiResult.evidence,
    expanded: false,
    timestamp: new Date().toISOString()
  };
}
```

---

### AnalyzeRequestPayload

**Purpose:** Request payload sent to backend

```typescript
export interface AnalyzeRequestPayload {
  text: string;
  options?: {
    use_search?: boolean;
    verification_enabled?: boolean;
  };
  model_id?: string;
  temperature?: number;
  max_tokens?: number;
  top_p?: number;
}
```

---

##Shared Contracts (Backend)

**Location:** `backend/app/contracts/search.py`

---

### SearchResult

**Purpose:** Standardized search result format across providers

```python
from pydantic import BaseModel, Field
from typing import Optional

class SearchResult(BaseModel):
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    ai_overview: Optional[str] = None  # Tavily-specific
    content: Optional[str] = None      # Full article text
    source: str = "external"           # or "[INTERNAL MEMORY]"
```

---

### IntentClaim

**Purpose:** Extracted claim from intent phase

```python
class IntentClaim(BaseModel):
    claim_text: str
    search_query: str
    verification_question: str
```

---

## Validation Rules

### Text Input

- **Min length:** 1 character
- **Max length:** 10,000 characters (recommended)
- **Empty strings:** Rejected (400 error)

### Confidence Scores

- **Range:** 0.0 - 1.0 (inclusive)
- **Precision:** 2 decimal places recommended

### Model IDs

- **Format:** `provider/model-name` (e.g., `meta-llama/llama-3.3-70b-instruct`)
- **Validation:** None (backend passes to LLM provider)

---

## Type Generation

### Backend → Frontend

**Future:** Generate TypeScript types from Pydantic models

```bash
# Using pydantic2ts (planned)
pydantic2ts backend/app/features/analyze/schemas.py \
  --output frontend/src/types/generated/api.ts
```

---

## Example: Full Request/Response Flow

### Request (Frontend)

```typescript
const payload: AnalyzeRequestPayload = {
  text: "The Earth is flat",
  model_id: "meta-llama/llama-3.3-70b-instruct",
  options: {
    use_search: true,
    verification_enabled: true
  }
};

const response = await fetch('/api/analyze', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(payload)
});
```

### Response (Backend)

```python
# Backend creates response
response = AnalyzeResponse(
    request_id=uuid4(),
    model_used="meta-llama/llama-3.3-70b-instruct",
    latency_ms=8742,
    claims=[
        ClaimResult(
            claim_text="The Earth is flat",
            verdict=Verdict.FALSE,
            confidence=0.98,
            reasoning="...",
            evidence=[...]
        )
    ]
)

return response  # FastAPI auto-serializes to JSON
```

### Processing (Frontend)

```typescript
const data: {
  request_id: string;
  claims: FactCheckApiResult[];
  // ...
} = await response.json();

// Map to frontend format
const results: FactCheckResult[] = data.claims.map(
  mapApiResultToFactCheckResult
);
```

---

See also:
- [API Endpoints](endpoints.md) - Endpoint documentation
- [Backend Architecture](../03-architecture/backend.md) - Pydantic usage
- [Frontend Architecture](../03-architecture/frontend.md) - TypeScript types
