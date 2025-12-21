# Phase 4: Verification

LLM synthesis of evidence into structured verdicts with confidence scores.

---

## Purpose

Synthesize all gathered evidence into a structured verdict with:
- Categorical verdict (TRUE, FALSE, MIXED, etc.)
- Confidence score (0.00-1.00)
- Reasoning explanation
- Evidence citations

---

## Verdict Categories

Based on actual implementation in `backend/app/features/verification/adapters/openai_compatible.py`:

| Verdict | Pattern | Typical Confidence |
|---------|---------|-------------------|
| `true` | Exact match | 0.85-1.00 |
| `false` | Exact match | 0.85-1.00 |
| `mostly_true` | Exact match | 0.70-0.90 |
| `mostly_false` | Exact match | 0.70-0.90 |
| `mixed` | Exact match | 0.50-0.75 |
| `unverifiable` | Default fallback | 0.00-0.60 |

**Validation:** Uses Pydantic regex pattern: `^(true|false|mostly_true|mostly_false|mixed|unverifiable)$`

---

## Implementation

### OpenAICompatibleClaimVerifier

**Location:** `backend/app/features/verification/adapters/openai_compatible.py`

**Features:**
- ✅ Circuit breaker protection against LLM API failures
- ✅ Automatic retries with exponential backoff
- ✅ Graceful degradation when circuit is open or LLM fails
- ✅ Token-optimized evidence formatting (excludes URLs)

### Structured Output (Pydantic)

```python
class _LLMClaimVerdict(BaseModel):
    verdict: str = Field(
        description="One of: true, false, mostly_true, mostly_false, mixed, unverifiable.",
        pattern=r"^(true|false|mostly_true|mostly_false|mixed|unverifiable)$",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)
```

**Uses:** `PydanticOutputParser` for structured parsing.

### Actual System Prompt

```python
_SYSTEM = (
    "You are a fact-checking AI. Given a claim and evidence snippets, return a structured response. "
    "Do not include markdown or any extra text outside the required format.\n\n"
    "Rules:\n"
    "- If evidence is insufficient, verdict MUST be 'unverifiable'.\n"
    "- Confidence MUST be between 0.0 and 1.0.\n\n"
    "{format_instructions}"
)
```

**Human Prompt:**
```
Claim:
{claim}

Evidence:
{evidence}
```

**Note:** Evidence is token-optimized - only includes title + text, NOT URLs (saves ~20-30 tokens per item).

### Circuit Breaker Protection

```python
@circuit_breaker("llm_verifier", LLM_CIRCUIT_CONFIG)
async def _verify_with_circuit_breaker(
    self,
    *,
    claim_clean: str,
    evidence: List[EvidenceSnippet],
    model: str,
    api_key: str,
    base_url: str,
) -> ClaimVerdict:
    # LLM call with automatic retries
    ...
```

**Graceful Degradation:**
- Empty claim → `unverifiable` (confidence: 0.0)
- No evidence → `unverifiable` (confidence: 0.0)
- No API key → `unverifiable` ("LLM is not configured")
- Circuit open → `unverifiable` ("LLM service temporarily unavailable. Please try again in N seconds.")
- Call failed → `unverifiable` ("LLM call failed. Please try again later.")

### Temperature & Model

```python
llm = ChatOpenAI(
    model=model,  # From request or OPENROUTER_MODEL
    temperature=0.2,  # Low but not zero (allows slight creativity)
    api_key=api_key,
    base_url=base_url or None,
)
```

---

## Evidence Prioritization

**Evidence is prioritized in this order:**

1. **`ai_overview`** (Tavily's AI summary) - Highest priority
2. **`content`** (Full article text) - Comprehensive context
3. **`text`** (Snippets) - Quick relevance check
4. **`[INTERNAL MEMORY]`** (RAG results) - Past verifications

**Why this order?**
- `ai_overview` is pre-processed and highly relevant
- `content` provides full context vs. cherry-picked snippets
- RAG results validate against historical knowledge

---

## LLM Prompt Structure

```python
prompt = f"""
You are a fact-checker evaluating this claim:

Claim: "{claim_text}"

Evidence from multiple sources:
{formatted_evidence}

Instructions:
1. Synthesize ALL evidence (don't cherry-pick)
2. Weigh source credibility (prioritize fact-check sites, primary sources)
3. Return structured verdict

Return JSON:
{{
  "verdict": "TRUE|FALSE|MOSTLY_TRUE|MOSTLY_FALSE|MIXED|UNVERIFIABLE",
  "confidence": 0.0-1.0,
  "reasoning": "Clear explanation of why this verdict was chosen"
}}

Guidelines:
- TRUE/FALSE requires strong consensus across sources
- MIXED when claim has both accurate and inaccurate parts
- UNVERIFIABLE when evidence is insufficient or conflicting
- Confidence reflects evidence strength and source agreement
"""
```

---

## Example: Full Verification

**Claim:** "Vaccines cause autism"

**Evidence Summary:**
- 15 sources from Tavily (10 debunking, 3 explaining studies, 2 historical context)
- 2 RAG results from past verifications (both FALSE verdicts)

**LLM Output:**
```json
{
  "verdict": "FALSE",
  "confidence": 0.98,
  "reasoning": "Overwhelming scientific consensus rejects the vaccines-autism link. The claim originated from a retracted 1998 study by Andrew Wakefield. Multiple large-scale studies (CDC, WHO) found no causal link. Fact-check sites (Snopes, PolitiFact) rated this claim as definitively false."
}
```

---

## Confidence Score Calibration

### High Confidence (0.85-1.00)
- Strong consensus across multiple credible sources
- Clear, unambiguous evidence
- No significant contradictions

### Medium Confidence (0.60-0.84)
- Moderate consensus
- Some conflicting information
- Nuanced claims requiring interpretation

### Low Confidence (0.00-0.59)
- Weak or conflicting evidence
- Insufficient sources
- Ambiguous claim

---

## Handling Edge Cases

### Conflicting Sources

**Scenario:** 3 sources say TRUE, 2 say FALSE

**Approach:**
- Weigh source credibility (fact-check sites > blogs)
- Check recency (newer evidence > older)
- Verdict: MIXED or lower confidence TRUE/FALSE

### Insufficient Evidence

**Scenario:** Only 1 weak source found

**Verdict:** UNVERIFIABLE (low confidence)

### Nuanced Claims

**Claim**: "Electric cars produce zero emissions"

**Verdict:** MOSTLY_FALSE (manufacturing emissions exist, operation is cleaner)

---

## Model Configuration

**Environment Variable:** `OPENROUTER_MODEL`

**Recommended Models:**
- **Default:** `meta-llama/llama-3.3-70b-instruct` (best balance)
- **Fast:** `meta-llama/llama-3.1-8b-instruct` (lower accuracy)
- **Premium:** `anthropic/claude-3.5-sonnet` (highest accuracy, expensive)

**Frontend Override:** Users can override via model selection UI

---

## Performance

**Typical Latency:** 2-3 seconds

**Factors:**
- LLM model speed
- Amount of evidence
- Complexity of synthesis

---

## Testing

```python
# backend/tests/test_verification.py
async def test_verify_with_strong_evidence():
    claim = "The Earth is round"
    evidence = [
        {"snippet": "NASA confirms Earth is spherical", "score": 0.98},
        {"snippet": "Satellite images show curved horizon", "score": 0.95}
    ]
    
    result = await verifier.verify(claim, evidence)
    
    assert result.verdict == "TRUE"
    assert result.confidence >= 0.90
```

---

## Code Pointers

- **Adapter:** `backend/app/features/verification/adapters/openai_compatible.py` (actual implementation)
- **Port interface:** `backend/app/features/verification/ports.py`
- **Orchestration:** `backend/app/features/analyze/service.py` (`_process_single_claim()`)
- **Circuit Breaker:** `backend/app/core/circuit_breaker.py`

---

See also:
- [Pipeline Overview](00-overview.md)
- [Phase 2: Parallel Search](03-search.md)
- [Continuous Learning](../05-features/continuous-learning.md)
