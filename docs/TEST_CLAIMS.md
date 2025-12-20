# FactuAI Claim Testing Benchmark

**Purpose:** Standardized test claims for internal QA before release.  
**Last Updated:** 2025-12-20

---

## How to Use

1. Run the backend: `uvicorn app.main:app --reload`
2. Send each claim to `POST /api/analyze` with `{"text": "<claim>"}`
3. Verify the response matches expected behavior
4. Check logs for pipeline phase execution

---

## Quick Smoke Test (5 mins)

Run these 3 claims first to verify the system is working:

| # | Claim | Expected |
|---|-------|----------|
| 1 | "The Earth is flat" | FALSE, high confidence |
| 2 | "Water boils at 100°C at sea level" | TRUE, high confidence |
| 3 | "Barack Obama was the 44th President of the United States" | TRUE, high confidence |

✅ If all 3 return reasonable verdicts with evidence, proceed to full testing.

---

## Level 1: Basic Claims (Factual Recall)

Simple, well-documented facts that should return clear verdicts.

| ID | Claim | Expected Verdict | Tests |
|----|-------|------------------|-------|
| B1 | "The Great Wall of China is visible from space with the naked eye" | FALSE | Common misconception detection |
| B2 | "Humans have 206 bones in their adult bodies" | TRUE | Basic scientific fact |
| B3 | "The capital of Australia is Sydney" | FALSE | Geographic fact |
| B4 | "Einstein won the Nobel Prize for the theory of relativity" | FALSE | Historical nuance (won for photoelectric effect) |
| B5 | "Goldfish have a 3-second memory" | FALSE | Popular myth debunking |

**What to verify:**
- [ ] Verdicts are correct
- [ ] Confidence scores are high (>0.80)
- [ ] Evidence sources are credible (no social media)
- [ ] Reasoning explains the verdict

---

## Level 2: Intermediate Claims (Context-Dependent)

Claims requiring nuance, temporal context, or multi-source verification.

| ID | Claim | Expected Verdict | Tests |
|----|-------|------------------|-------|
| I1 | "COVID-19 vaccines contain microchips" | FALSE | Conspiracy theory rejection |
| I2 | "Electric cars produce zero emissions" | MIXED/MOSTLY_FALSE | Nuanced (manufacturing vs. operation) |
| I3 | "The Amazon rainforest produces 20% of the world's oxygen" | MOSTLY_FALSE | Common exaggeration |
| I4 | "Drinking 8 glasses of water a day is medically necessary" | FALSE/MIXED | Health myth with nuance |
| I5 | "5G networks cause cancer" | FALSE | Tech misinformation |

**What to verify:**
- [ ] System handles nuanced verdicts (MIXED, MOSTLY_TRUE/FALSE)
- [ ] Reasoning acknowledges complexity
- [ ] Multiple evidence sources cited
- [ ] No false equivalence (conspiracy vs. science)

---

## Level 3: Advanced Claims (Multi-Hop & Pivot Required)

Complex claims that should trigger the **Pivot Loop** for follow-up research.

| ID | Claim | Expected Behavior | Tests |
|----|-------|-------------------|-------|
| A1 | "The Tesla Pi Phone uses Air Wi-Fi technology" | FALSE + PIVOT | Should pivot to research "Tesla Pi Phone" hoax |
| A2 | "Ivermectin is FDA-approved for treating COVID-19 in humans" | FALSE | Requires distinguishing veterinary vs. human use |
| A3 | "NASA discovered a parallel universe where time runs backwards" | FALSE | Clickbait/misinterpretation detection |
| A4 | "mRNA vaccines alter your DNA permanently" | FALSE | Technical claim requiring scientific sources |
| A5 | "The Wayfair conspiracy proves child trafficking through cabinets" | FALSE | Named conspiracy debunking |

**What to verify:**
- [ ] Pivot Loop triggers when new entities discovered
- [ ] Follow-up searches are specific and targeted
- [ ] Final verdict synthesizes all evidence
- [ ] Conspiracy theories rejected with authoritative sources

---

## Level 4: Edge Cases & Adversarial

Claims designed to stress-test the system.

| ID | Claim | Expected Behavior | Tests |
|----|-------|-------------------|-------|
| E1 | "I heard that maybe vaccines could be bad" | Should extract concrete claim or return UNVERIFIABLE | Vague language handling |
| E2 | "" (empty string) | Error or no claims extracted | Empty input handling |
| E3 | "ajskdfhaksjdfh random gibberish" | No verifiable claims | Gibberish rejection |
| E4 | "The sky is blue because of quantum entanglement with oceanic frequencies" | FALSE | Plausible-sounding nonsense |
| E5 | "According to my friend, Bill Gates said..." | UNVERIFIABLE or FALSE | Unverifiable source chain |
| E6 | "Breaking: Scientists just discovered immortality drug!" | UNVERIFIABLE/FALSE | Recency + sensationalism |

**What to verify:**
- [ ] System doesn't crash on edge inputs
- [ ] Returns appropriate UNVERIFIABLE for vague claims
- [ ] Doesn't hallucinate evidence
- [ ] Handles sensationalist language appropriately

---

## Level 5: Multi-Claim Extraction

Paragraphs containing multiple claims to test intent extraction.

### Test M1: Mixed Bag (3 claims)
```
Claim: "The Eiffel Tower is the tallest building in the world, 
it was built in 1889, and it's located in London."
```

**Expected:**
- Claim 1: "Eiffel Tower is tallest building" → FALSE
- Claim 2: "Built in 1889" → TRUE
- Claim 3: "Located in London" → FALSE

### Test M2: Compound Claim (linked facts)
```
Claim: "Since the moon landing was faked in 1969, 
NASA has continued to deceive the public about space exploration."
```

**Expected:**
- Should identify the false premise (moon landing faked)
- Should NOT accept the consequent claim as valid

### Test M3: Real-world paragraph
```
Claim: "The World Health Organization declared COVID-19 a pandemic in March 2020. 
The first vaccine was approved in December 2020, and by 2021 over 50% of 
Americans had been vaccinated."
```

**Expected:**
- 3 claims extracted
- All should be TRUE or MOSTLY_TRUE
- Specific dates/percentages verified

---

## Regression Checklist

After running all tests, verify:

### Pipeline Phases
- [ ] **Phase 0 (Intent):** Claims extracted correctly from all inputs
- [ ] **Phase 1 (Strategist):** 3 queries generated per claim (factual, hoax, scientific)
- [ ] **Phase 2 (Search):** Tavily returns results, no social media sources
- [ ] **Phase 3 (Pivot):** Triggers appropriately on A1-A5 claims
- [ ] **Phase 4 (Verify):** LLM synthesizes evidence into coherent verdict

### Data Quality
- [ ] No Facebook, TikTok, Reddit, Twitter URLs in evidence
- [ ] Source titles and URLs are populated
- [ ] Relevance scores are reasonable (0.5-1.0 range)
- [ ] Confidence scores correlate with evidence strength

### Performance
- [ ] Basic claims < 10s latency
- [ ] Complex claims < 30s latency
- [ ] No timeouts or 500 errors

### RAG Memory (if populated)
- [ ] Previously verified claims show `[INTERNAL MEMORY]` tagged results
- [ ] Similarity threshold (0.80) filtering works correctly

---

## Quick Reference: Expected Verdicts

| Verdict | Meaning | Typical Confidence |
|---------|---------|-------------------|
| TRUE | Claim is accurate | 0.85-1.0 |
| MOSTLY_TRUE | Largely accurate with minor caveats | 0.70-0.90 |
| MIXED | Contains both true and false elements | 0.50-0.75 |
| MOSTLY_FALSE | Largely inaccurate but has kernel of truth | 0.70-0.90 |
| FALSE | Claim is inaccurate | 0.85-1.0 |
| UNVERIFIABLE | Insufficient evidence to determine | 0.30-0.60 |

---

## Automated Testing (Future)

These claims can be converted to pytest fixtures:

```python
# backend/tests/test_claims_benchmark.py (future)
BENCHMARK_CLAIMS = [
    ("The Earth is flat", "false"),
    ("Water boils at 100°C at sea level", "true"),
    # ... etc
]
```

---

**Maintainer Notes:**
- Update this file when adding new pipeline features
- Add failing edge cases discovered in production
- Keep total test count manageable (< 30 claims for manual testing)
