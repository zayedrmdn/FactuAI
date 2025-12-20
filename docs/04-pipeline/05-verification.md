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

| Verdict | Meaning | Typical Confidence |
|---------|---------|-------------------|
| `TRUE` | Claim is accurate | 0.85-1.00 |
| `MOSTLY_TRUE` | Largely accurate with minor caveats | 0.70-0.90 |
| `MIXED` | Contains both true and false elements | 0.50-0.75 |
| `MOSTLY_FALSE` | Largely inaccurate but has kernel of truth | 0.70-0.90 |
| `FALSE` | Claim is inaccurate | 0.85-1.00 |
| `UNVERIFIABLE` | Insufficient evidence to determine | 0.30-0.60 |

---

## Implementation

### LangChain Structured Output

**Location:** `backend/app/features/verification/adapters/native.py`

**Process:**
1. Combine claim + all evidence (Tavily + RAG + pivot results)
2. Send to LLM with structured output schema
3. Parse response into `VerificationResult`

**Example:**

```python
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

class VerificationResult(BaseModel):
    verdict: str = Field(description="TRUE, FALSE, MIXED, etc.")
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(description="Explanation of verdict")
    
# LangChain enforces this schema
parser = PydanticOutputParser(pydantic_object=VerificationResult)
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

- Adapter: `backend/app/features/verification/adapters/native.py`
- Port interface: `backend/app/features/verification/ports.py`
- Orchestration: `backend/app/features/analyze/service.py`

---

See also:
- [Pipeline Overview](00-overview.md)
- [Phase 2: Parallel Search](03-search.md)
- [Continuous Learning](../05-features/continuous-learning.md)
