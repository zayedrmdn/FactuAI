# The 4-Phase Analysis Pipeline

Comprehensive overview of Fact uAI's core analysis pipeline.

---

## Overview

FactuAI uses a **4-phase analysis pipeline** with two modes for robust claim verification:

### Analysis Modes

**Quick Mode:**
- Intent Extraction → Single Direct Search → Verification
- ~7-11s total latency
- Best for simple, straightforward claims

**Deep Mode (Default):**
1. **Phase 1: Intent Extraction + Strategist** - Parse text into claims & generate 3 multi-angle queries
2. **Phase 2: Parallel Search** - Gather evidence from Tavily + internal RAG memory
3. **Phase 3: Pivot Loop** - Detect new concepts → execute follow-up research  
4. **Phase 4: Verification** - Synthesize evidence into verdict with confidence
- ~10-16s total latency (with pivot)
- Comprehensive multi-angle evidence gathering

---

## Pipeline Flow Diagram

### Deep Mode (Default)

```mermaid
graph TD
    A[User Input] --> B[Phase 1A: Intent Extraction<br/>Parse Claims + Global Context]
    B --> C[Phase 1B: STRATEGIST<br/>Multi-Angle Query Generation]
    C --> D[Phase 2: PARALLEL SEARCH<br/>3 Queries + RAG Memory]
    D --> E[Phase 3: PIVOT LOOP<br/>Detect New Concepts]
    E -->|Pivot Needed| F[Follow-up Search]
    F --> G[Merge Evidence]
    E -->|No Pivot| G
    G --> H[Phase 4: VERIFICATION<br/>LLM Synthesis]
    H --> I[Verdict + Confidence]
    I --> J[Persistence + Learning]
```

### Quick Mode

```mermaid
graph TD
    A[User Input] --> B[Phase 1: Intent Extraction<br/>Parse Claims]
    B --> C[Phase 2: Direct Search<br/>15 Results]
    C --> D[Phase 4: VERIFICATION<br/>LLM Synthesis]
    D --> E[Verdict + Confidence]
    E --> F[Persistence]
```

---

## Phase Summaries

### Phase 1: Intent Extraction + Strategist

**Part A: Intent Extraction (LLM-Based)**

**Purpose:** Extract structured, verifiable claims from raw user input + global context.

**Input:** Raw text (e.g., "The Earth is flat and vaccines cause autism")  
**Output:** `IntentResult` with global context and claims

```json
{
  "global_context": "Earth, vaccines, scientific consensus, medical research",
  "claims": [
    {
      "claim_text": "The Earth is flat",
      "search_query": "Earth shape flat evidence",
      "verification_question": "Is the Earth flat?"
    },
    {
      "claim_text": "Vaccines cause autism",
      "search_query": "vaccines autism link scientific studies",
      "verification_question": "Do vaccines cause autism?"
    }
  ]
}
```

**Implementation:**
- Uses `LLMIntentAdapter` (LangChain + structured output)
- Configured via `INTENT_LLM_MODEL` or falls back to `OPENROUTER_MODEL`
- **Key Feature:** Extracts `global_context` (entities, locations, events) to improve query generation

**See:** [01-intent.md](01-intent.md)

---

**Part B: Strategist - Multi-Angle Query Generation (Deep Mode Only)**

**Purpose:** Generate 3 strategic search queries per claim to maximize evidence quality.

**Input:** Single claim  
**Output:** 3 distinct queries

```json
{
  "factual_query": "shape of Earth scientific consensus",
  "hoax_query": "flat Earth debunked fact-check",
  "scientific_query": "Earth spherical evidence NASA images"
}
```

**Rationale:** Approaching from multiple angles (factual, debunking, scientific) surfaces diverse evidence sources and reduces bias.

**Implementation:**
- LLM prompt engineering
- Each query targets different source types

**See:** [02-strategist.md](02-strategist.md)

---

### Phase 2: Parallel Search - Hybrid External + Internal

**Purpose:** Gather evidence from external APIs (Tavily) and internal knowledge base (pgvector).

**Execution:**
1. Run 3 Tavily searches (factual, hoax, scientific) in parallel via `asyncio.gather`
2. Simultaneously query internal pgvector for similar past claims/evidence
3. Merge results, deduplicate by URL

**External Search (Tavily):**
- **Strict filtering:** ` exclude_domains` blocks 19 social media domains
- **Rich data:** Returns `ai_overview`, `content` (full article), `text` (snippets)
- **Configuration:** `auto_parameters=True`, `include_answer=True`, `include_raw_content=True`

**Internal RAG Search:**
- Query `claims` and `evidence` tables via cosine similarity
- **Threshold:** Only returns results with similarity > 0.80 (distance < 0.20)
- Results tagged with `[INTERNAL MEMORY]`

**Latency:** ~Same as single query (parallel execution)

**See:** [03-search.md](03-search.md)

---

### Phase 3: Pivot Loop - Iterative Research

**Purpose:** Detect if initial evidence reveals a **new specific entity** (product, event, concept) requiring follow-up research.

**Decision Process:**
1. LLM analyzes claim + initial search results
2. Generates `PivotDecision`:
   - `needs_pivot`: boolean
   - `pivot_query`: specific follow-up query
   - `reason`: explanation

**Example:**
```
Claim: "Air Wi-Fi technology is used in the Tesla Pi Phone"
Initial Search: Reveals "Tesla Pi Phone" is a hoax
Pivot Decision: {
  "needs_pivot": true,
  "pivot_query": "Tesla Pi Phone hoax fact-check",
  "reason": "Initial search mentions 'Tesla Pi Phone' as a specific entity requiring dedicated research"
}
```

**Safety:** Hard limit of **1 pivot** per claim (no infinite loops).

**See:** [04-pivot.md](04-pivot.md)

---

### Phase 4: Verification - LLM Synthesis

**Purpose:** Synthesize all evidence into a structured verdict with confidence and reasoning.

**Input:** Claim + merged evidence (from all phases)  
**Output:** Verdict object

```json
{
  "verdict": "false",
  "confidence": 0.95,
  "reasoning": "Multiple scientific sources confirm Earth is an oblate spheroid. No credible evidence supports flat Earth claim. Fact-check sites explicitly debunk this as a common misconception.",
  "evidence": [
    {
      "snippet": "NASA satellite images show...",
      "source_url": "https://nasa.gov/...",
      "relevance_score": 0.98
    }
  ]
}
```

**Implementation:**
- LangChain-based LLM call with structured output
- Prioritizes `ai_overview` and `content` over snippets
- Configurable via `OPENROUTER_MODEL`

**See:** [05-verification.md](05-verification.md)

---

## Post-Processing: Persistence + Learning

After verification completes:

1. **Persistence:** Store verification, claims, sources, evidence in Postgres
2. **Continuous Learning:** If confidence ≥ 0.85:
   - Asynchronously generate embeddings for claim and evidence
   - Store in pgvector columns for future RAG retrieval

See [../05-features/continuous-learning.md](../05-features/continuous-learning.md).

---

## Phase Timing (Typical)

| Phase | Quick Mode | Deep Mode (No Pivot) | Deep Mode (With Pivot) |
|-------|-----------|---------------------|----------------------|
| Phase 1: Intent | 1-2s | 1-2s | 1-2s |
| Phase 1B: Strategist | N/A | 0.5-1s | 0.5-1s |
| Phase 2: Search | 3-5s (single) | 3-5s (parallel) | 3-5s (parallel) |
| Phase 3: Pivot | N/A | N/A | +3-5s |
| Phase 4: Verification | 2-3s | 2-3s | 2-3s |
| **Total Latency** | **6-10s** | **7-11s** | **10-16s** |

---

## Example: Full Pipeline Execution (Deep Mode)

**Input:** "The Great Wall of China is visible from space with the naked eye"

**Phase 1A → Intent Extraction:**
```json
{
  "global_context": "Great Wall of China, space, visibility, astronomy",
  "claims": [
    {
      "claim_text": "The Great Wall of China is visible from space with the naked eye",
      "search_query": "Great Wall China visible space naked eye",
      "verification_question": "Is the Great Wall of China visible from space with the naked eye?"
    }
  ]
}
```

**Phase 1B → Strategist:**
- Factual: "Great Wall China visible from space scientific evidence"
- Hoax: "Great Wall space myth debunked fact-check"
- Scientific: "Great Wall China satellite imagery visibility astronauts"

**Phase 2 → Parallel Search:**
- Query 1 (Factual) → 5 results
- Query 2 (Hoax) → 5 results  
- Query 3 (Scientific) → 5 results
- RAG (internal memory) → 2 results tagged `[INTERNAL MEMORY]`
- **Merged & Deduplicated:** 15 unique sources

**Phase 3 → Pivot Decision:**
```json
{
  "needs_pivot": false,
  "reason": "No new specific entities detected; sufficient evidence from fact-check sites and NASA sources"
}
```

**Phase 4 → Verification:**
```json
{
  "verdict": "false",
  "confidence": 0.92,
  "reasoning": "Multiple authoritative sources (NASA, fact-check sites) confirm the Great Wall is NOT visible from space with the naked eye. This is a common myth. Low Earth orbit astronauts cannot see it without optical aid."
}
```

**Persistence + Learning:**
- Stored in DB with all evidence
- Confidence 0.92 ≥ 0.85 threshold → embeddings generated
- Available for future RAG retrieval

---

## Phase Deep Dives

For detailed documentation on each phase:

- [Phase 1: Intent Extraction + Strategist](01-intent.md)
- [Phase 1B: Strategist (Multi-Angle Queries)](02-strategist.md)
- [Phase 2: Parallel Search (Hybrid External + Internal)](03-search.md)
- [Phase 3: Pivot Loop (Deep Mode)](04-pivot.md)
- [Phase 4: Verification (LLM Synthesis)](05-verification.md)

---

See also:
- [Architecture Overview](../03-architecture/overview.md)
- [Features](../05-features/)
- [Testing with Benchmark Claims](../06-testing/test-claims.md)
