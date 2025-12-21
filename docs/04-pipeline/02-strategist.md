# Phase 1: Strategist

Multi-angle query generation for comprehensive evidence gathering.

---

## Purpose

Generate 3 strategic search queries per claim to maximize evidence quality and diversity.

---

## The 3-Query Strategy

For each claim, generate queries targeting different source types:

### 1. Factual Query
**Target:** Primary sources, official statements, direct evidence  
**Approach:** Direct fact-checking  
**Example:** "Eiffel Tower official height meters"

### 2. Hoax Query
**Target:** Fact-check sites, debunking articles, exposés  
**Approach:** Actively search for debunking content  
**Keywords:** hoax, debunked, false, fact-check, snopes, politifact  
**Example:** "Eiffel Tower height hoax OR debunked"

### 3. Scientific Query
**Target:** Academic papers, research studies, expert analysis  
**Approach:** Scientific/technical angle  
**Keywords:** study, research, journal, expert, analysis  
**Example:** "Eiffel Tower height scientific measurement"

---

## Implementation

### Location

**File:** `backend/app/features/analyze/service.py`  
**Method:** `_generate_multi_queries()`

**Prompts:** `backend/app/features/analyze/prompts.py`

### Structured Output (Pydantic)

```python
class MultiAngleQueries(BaseModel):
    """Structured output for multi-angle search queries."""
    
    factual_query: str = Field(
        description="Direct fact-checking query to find primary sources confirming or denying the claim."
    )
    hoax_query: str = Field(
        description="Debunking-focused query to find fact-check articles or hoax exposés about this claim."
    )
    scientific_query: str = Field(
        description="Academic/research query to find scientific studies or expert analysis on the topic."
    )
```

### Actual Prompts

**System Prompt:**
```
Generate 3 distinct search queries to verify claims from different angles. Each query must be 5-10 words and include the core subject.

1. **Factual Query**: Direct search for primary sources (news, official data, statements).
   Example: "Eiffel Tower official height meters"

2. **Hoax Query**: Debunking search with terms: hoax, debunked, false, fact-check, snopes, politifact.
   Example: "Eiffel Tower height hoax OR debunked"

3. **Scientific Query**: Academic search with terms: study, research, journal, expert, analysis.
   Example: "Eiffel Tower height scientific measurement"

Rules: Each query targets a different angle. Never include "claim" in queries.
```

**Human Prompt:**
```
Generate 3 multi-angle search queries for this claim:

CLAIM: {claim}

CONTEXT (use to make queries more specific - include relevant entities/locations): {context}
```

**Note:** The `{context}` variable contains global context extracted during intent phase (entities, locations, background info).

### Code Flow

```python
async def _generate_multi_queries(
    self,
    *,
    claim: str,
    context: str,  # Global context from intent extraction
    model: str,
) -> List[str]:
    """Generate 3 strategic multi-angle search queries using LLM."""
    
    # Use LangChain structured output
    llm = ChatOpenAI(model=model, temperature=0.3, api_key=api_key, base_url=base_url)
    structured_llm = llm.with_structured_output(MultiAngleQueries)
    chain = prompt | structured_llm
    
    result: MultiAngleQueries = await chain.ainvoke({
        "claim": claim,
        "context": context,
    })
    
    queries = [
        result.factual_query.strip(),
        result.hoax_query.strip(),
        result.scientific_query.strip(),
    ]
    
    # Filter empty queries and return
    return [q for q in queries if q]
```

**Fallback:** If LLM fails, returns `[claim]` as single query.

---

## Example: Full Execution

**Claim:** "Coffee cures cancer"  
**Global Context:** "coffee, cancer, medical research"

**Generated Queries:**
```json
{
  "factual_query": "coffee cancer cure medical research clinical trials",
  "hoax_query": "coffee cures cancer myth debunked fact-check",
  "scientific_query": "coffee cancer prevention scientific studies peer-reviewed"
}
```

**Why this works:**
- Factual query → finds legitimate research on coffee & cancer
- Hoax query → surfaces fact-checks explicitly debunking cure claims
- Scientific query → finds nuanced academic discussion (e.g., correlation vs. causation)

---

## Rationale

**Why 3 queries instead of 1?**

1. **Source Diversity:** Different queries surface different types of sources
2. **Bias Reduction:** Approaching from multiple angles reduces confirmation bias
3. **Comprehensive Coverage:** Captures both supporting and contradicting evidence
4. **Quality Filtering:** Fact-check sites (hoax query) are particularly valuable for misinformation

---

## Performance

**Typical Latency:** 0.5-1 second

**Factors:**
- LLM model speed (`temperature=0.3` for creativity)
- Complexity of claim
- Context length

**Recommended Model:** Use `openai/gpt-4o-mini` for reliable structured output. Free-tier models may return plain text instead of JSON, triggering fallback to single-query mode.

---

## Quality Metrics

**Good Queries:**
- Specific enough to find relevant sources (5-10 words)
- Broad enough to avoid zero results
- Include key entities and concepts
- Targeted to appropriate source types

**Bad Queries:**
- Too vague ("is this true?")
- Too specific ("exact quote from X on date Y")
- Missing key entities
- Redundant across all 3 queries

---

## Code Pointers

- Service: `backend/app/features/analyze/service.py` (`_generate_multi_queries()`)  
- Prompts: `backend/app/features/analyze/prompts.py` (`QUERY_GENERATION_SYSTEM`, `QUERY_GENERATION_HUMAN`)
- Model: `backend/app/features/analyze/prompts.py` (`MultiAngleQueries`)

---

See also:
- [Pipeline Overview](00-overview.md)
- [Phase 2: Parallel Search](03-search.md)
- [Source Filtering](../05-features/source-filtering.md)
