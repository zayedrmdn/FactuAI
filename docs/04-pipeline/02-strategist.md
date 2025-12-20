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
**Example:** "Earth shape scientific consensus NASA evidence"

### 2. Hoax Query
**Target:** Fact-check sites, debunking articles, exposés  
**Approach:** Actively search for debunking content  
**Example:** "flat Earth myth debunked fact-check"

### 3. Scientific Query
**Target:** Academic papers, research studies, expert analysis  
**Approach:** Scientific/technical angle  
**Example:** "Earth spherical shape satellite imagery physics"

---

## Rationale

**Why 3 queries instead of 1?**

1. **Source Diversity:** Different queries surface different types of sources
2. **Bias Reduction:** Approaching from multiple angles reduces confirmation bias
3. **Comprehensive Coverage:** Captures both supporting and contradicting evidence
4. **Quality Filtering:** Fact-check sites (hoax query) are particularly valuable for misinformation

---

## Implementation

### LLM Prompt Engineering

```python
# Simplified prompt structure
prompt = f"""
Given this claim: "{claim_text}"

Generate 3 search queries:
1. Factual query: Direct evidence from primary sources
2. Hoax query: Debunking/fact-check focused
3. Scientific query: Academic/research angle

Return as JSON:
{{
  "factual_query": "...",
  "hoax_query": "...",
  "scientific_query": "..."
}}
"""
```

---

## Example: Full Execution

**Claim:** "Coffee cures cancer"

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

## Query Optimization Techniques

### 1. Keyword Selection
- Include domain-specific terms
- Avoid overly broad keywords
- Balance specificity with recall

### 2. Temporal Awareness
- Add year/date if claim is time-sensitive
- Example: "COVID-19 vaccine safety 2023 studies"

### 3. Entity Extraction
- Identify key entities (people, products, events)
- Ensure entity names are in queries

---

## Performance

**Typical Latency:** 0.5-1 second

**Factors:**
- LLM model speed
- Complexity of claim

---

## Quality Metrics

**Good Queries:**
- Specific enough to find relevant sources
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

- Implementation: `backend/app/features/analyze/service.py` (Strategist phase)
- Uses same LLM as verification (`OPENROUTER_MODEL`)

---

See also:
- [Pipeline Overview](00-overview.md)
- [Phase 2: Parallel Search](03-search.md)
- [Source Filtering](../05-features/source-filtering.md)
