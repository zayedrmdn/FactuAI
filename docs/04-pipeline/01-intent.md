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
2. Extract individual claims + global context
3. Generate search query for each claim
4. Generate verification question (optional)

### Structured Output Schema

Uses LangChain `with_structured_output()` for guaranteed schema compliance.

```python
class _ClaimOutput(BaseModel):
    """Structured output for a single extracted claim."""
    
    claim_text: str = Field(
        description="The exact factual claim to verify, stated clearly and concisely."
    )
    search_query: str = Field(
        description="A web search query to find evidence for or against this claim."
    )
    verification_question: Optional[str] = Field(
        default=None,
        description="A yes/no question to determine if the claim is true.",
    )

class _ClaimListOutput(BaseModel):
    """Structured output for the full list of extracted claims."""
    
    global_context: str = Field(
        default="",
        description="Key entities, locations, events, and background shared across all claims. Used to ground search queries.",
    )
    claims: List[_ClaimOutput] = Field(
        default_factory=list,
        description="List of distinct, verifiable factual claims extracted from the text.",
    )
```

**Key Feature:** `global_context` extraction - identifies shared entities/locations across all claims to improve search query quality in Phase 1.

### Actual System Prompt

```
You are a claim extraction assistant. Your task is to analyze text and extract distinct, verifiable factual claims.

Rules:
1. Extract only FACTUAL claims that can be verified with evidence (true/false).
2. Ignore opinions, predictions, questions, and rhetorical statements.
3. Each claim should be self-contained and understandable without context.
4. Generate a concise web search query to find evidence for each claim.
5. Generate a verification question that can be answered with yes/no.
6. Do NOT extract duplicate or overlapping claims.
7. If no verifiable claims exist, return an empty list.
8. **IMPORTANT**: Extract a global_context summarizing key entities (people, organizations), 
   locations, events, and background information shared across all claims. This context helps 
   ground search queries. Example: "South Kalimantan, Governor Muhidin, helicopter crash site"

Examples of GOOD claims:
- "The Eiffel Tower is 330 meters tall."
- "Apple was founded in 1976."
- "Water boils at 100°C at sea level."

Examples of BAD claims (do NOT extract):
- "I think the weather will be nice." (opinion/prediction)
- "Is Python a good language?" (question)
- "Everyone knows about climate change." (vague/rhetorical)
```

### Code Flow

```python
async def _extract_claims(
    self,
    *,
    text: str,
    max_claims: int,
    model: str,
    api_key: str,
    api_base: str,
) -> IntentResult:
    """Extract claims using LLM with structured output."""
    
    llm = ChatOpenAI(
        model=model,
        temperature=0.1,  # Low temperature for consistent extraction
        api_key=api_key,
        base_url=api_base or None,
    )
    
    # Use with_structured_output for guaranteed schema compliance
    structured_llm = llm.with_structured_output(_ClaimListOutput)
    chain = prompt | structured_llm
    
    result: _ClaimListOutput = await chain.ainvoke({
        "text": text,
        "max_claims": max_claims,
    })
    
    # Convert to IntentResult format
    global_context = (result.global_context or "").strip()
    logger.info(f"[INTENT-LLM] Extracted {len(result.claims)} claim(s), context: '{global_context[:80]}...'")
    
    return IntentResult(global_context=global_context, claims=items)
```

**Graceful Degradation:** Returns empty `IntentResult(global_context="", claims=[])` on failure.

### Model Configuration Priority

1. **Frontend override** (if user selects model in UI)
2. **Intent-specific config** (`INTENT_LLM_MODEL`)
3. **Main LLM config** (`OPENROUTER_MODEL`)

**Fallback chain** for API keys/URLs:
```python
api_key = settings.intent_llm_api_key or settings.llm_api_key
api_base = settings.intent_llm_api_base_url or settings.llm_api_base_url
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
