# Phase 0: Intent Extraction

LLM-based claim parsing from raw user input.

---

## Purpose

Extract structured, verifiable claims from unstructured text.

**Input:** Raw user text (e.g., "I heard that vaccines cause autism and the Earth is flat")  
**Output:** List of structured claims with search queries

---

## Implementation

### LLMIntentAdapter (Default)

**Location:** `backend/app/features/intent/adapters/llm.py`

**Process:**
1. Send user input to LLM with structured output schema
2. Extract individual claims
3. Generate search query for each claim
4. Generate verification question

**Example:**

**Input:**
```
"The Great Wall of China is visible from space and it's the only man-made structure you can see from the Moon."
```

**Output:**
```json
[
  {
    "claim_text": "The Great Wall of China is visible from space",
    "search_query": "Great Wall China visible space evidence",
    "verification_question": "Is the Great Wall of China visible from space?"
  },
  {
    "claim_text": "The Great Wall is the only man-made structure visible from the Moon",
    "search_query": "structures visible from Moon Great Wall",
    "verification_question": "Can you see the Great Wall from the Moon?"
  }
]
```

---

## LLM Configuration

**Environment Variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `INTENT_ADAPTER` | `backend.app.features.intent.adapters.llm.LLMIntentAdapter` | Adapter class path |
| `INTENT_LLM_MODEL` | `meta-llama/llama-3.3-70b-instruct` | Model for claim extraction |
| `INTENT_LLM_API_KEY` | Falls back to `LLM_API_KEY` | Separate API key (optional) |
| `INTENT_LLM_BASE_URL` | Falls back to `LLM_API_BASE_URL` | Separate base URL (optional) |

**Recommended Model:** Fast, cheap model (e.g., `meta-llama/llama-3.1-8b-instruct`) since intent extraction is simpler than verification.

---

## Structured Output Schema

```python
from pydantic import BaseModel

class IntentClaim(BaseModel):
    claim_text: str  # The extracted claim
    search_query: str  # Optimized query for search engines
    verification_question: str  # Question format for verification
```

---

## Edge Cases

### Multiple Claims in One Sentence

**Input:** "Vaccines cause autism and 5G causes cancer"

**Output:** 2 separate claims
```json
[
  {"claim_text": "Vaccines cause autism", ...},
  {"claim_text": "5G causes cancer", ...}
]
```

### Vague or Opinion-Based Input

**Input:** "I think maybe the government is hiding something"

**Expected:** No claims extracted (too vague, not verifiable)

**Actual:** System may extract `[]` or return error. Frontend displays user-friendly message.

### Questions vs. Claims

**Input:** "Is the Earth flat?"

**Extracted:** "The Earth is flat" (converted to assertion for verification)

---

## Performance

**Typical Latency:** 1-2 seconds

**Factors:**
- LLM model speed
- Number of claims in input
- Network latency to LLM provider

---

## Testing

```python
# backend/tests/test_intent.py
async def test_extract_single_claim():
    adapter = LLMIntentAdapter()
    result = await adapter.extract("The Earth is flat")
    
    assert len(result) == 1
    assert "Earth is flat" in result[0].claim_text
```

---

## Code Pointers

- Adapter: `backend/app/features/intent/adapters/llm.py`
- Port interface: `backend/app/features/intent/ports.py`
- Orchestration: `backend/app/features/analyze/service.py`

---

See also:
- [Pipeline Overview](00-overview.md)
- [Phase 1: Strategist](02-strategist.md)
- [Architecture: Backend](../03-architecture/backend.md)
