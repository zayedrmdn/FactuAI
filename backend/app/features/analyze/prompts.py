"""
Strategist Pipeline Prompts - Refined Edition.

Token-optimized prompts for multi-angle query generation and fact verification.
Maintains full backward compatibility with existing Pydantic models.
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


# ============================================================================
# QUERY GENERATION - "Think Before Searching"
# ============================================================================

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


QUERY_GENERATION_SYSTEM = """\
Generate 3 distinct search queries to verify claims from different angles. Each query must be 5-10 words and include the core subject.

1. **Factual Query**: Direct search for primary sources (news, official data, statements).
   Example: "Eiffel Tower official height meters"

2. **Hoax Query**: Debunking search with terms: hoax, debunked, false, fact-check, snopes, politifact.
   Example: "Eiffel Tower height hoax OR debunked"

3. **Scientific Query**: Academic search with terms: study, research, journal, expert, analysis.
   Example: "Eiffel Tower height scientific measurement"

Rules: Each query targets a different angle. Never include "claim" in queries."""

QUERY_GENERATION_HUMAN = """\
Generate 3 multi-angle search queries for this claim:

CLAIM: {claim}

CONTEXT (use to make queries more specific - include relevant entities/locations): {context}"""


# ============================================================================
# PIVOT LOOP - "React to New Information"
# ============================================================================

class PivotDecision(BaseModel):
    """Structured output for pivot loop decision."""

    needs_pivot: bool = Field(
        description="True if evidence reveals a specific entity/concept that requires additional research."
    )
    pivot_query: Optional[str] = Field(
        default=None,
        description="Search query for the newly discovered concept (if needs_pivot is True)."
    )
    reason: str = Field(
        description="Brief explanation of why pivot is needed or not needed."
    )


PIVOT_CHECK_SYSTEM = """\
Analyze search results to determine if a NEW specific entity (product, event, person, company) central to the claim requires additional research.

**Pivot Required (needs_pivot=True):**
- Evidence reveals specific entity misrepresented by rumor (e.g., "Tesla Pi Phone" hoax)
- Crucial event/study emerges not in original search (e.g., Wakefield 1998 study)
- Proper noun appears as "root cause" of rumor

**No Pivot (needs_pivot=False):**
- Evidence directly addresses claim
- Simple factual question (dates, measurements)
- No new specific entity discovered
- Concept already covered by original queries

Be CONSERVATIVE. Only pivot for truly new, specific entities. Pivot query: 3-6 words maximum."""

PIVOT_CHECK_HUMAN = """\
CLAIM: {claim}

QUERIES USED: {queries}

EVIDENCE:
{evidence_summary}

Does evidence reveal a NEW specific entity requiring research?"""


# ============================================================================
# VERIFICATION - "Read Deeply"
# ============================================================================

VERIFICATION_SYSTEM = """\
Fact-check claims by analyzing search evidence. Apply logic rules for missing evidence.

**CRITICAL: Missing Evidence Logic**

1. **"Silence is Proof"**: Major events (scientific breakthroughs, NASA/FBI actions, celebrity news) generate major coverage. If search finds NO reputable coverage, claim is FALSE.

2. **"Scientific Impossibility"**: Claims violating basic physics/biology (e.g., "pigeons generate Wi-Fi") without peer-reviewed proof are FALSE. Lack of proof is sufficient.

**Process**
1. Assess claim magnitude: Would this be on CNN/BBC if true?
2. Scan evidence for direct confirmation
3. Apply negative filter: Major claim + Silent/Tangential evidence = FALSE

**Verdicts**
- VERIFIED: Strong supporting evidence
- FALSE: Strong refutation OR major claim with no evidence
- UNCLEAR: Obscure/personal topic (use sparingly)

Output JSON with confidence 0.0-1.0:
{format_instructions}"""

VERIFICATION_HUMAN = """\
Claim: {claim}

=== AI OVERVIEW ===
{ai_overview}

=== SOURCES ===
{evidence}"""


# ============================================================================
# EVIDENCE FORMATTING (Token-Optimized)
# ============================================================================

def format_evidence_for_verification(
    evidence_items: List[dict],
    max_content_length: int = 2000,
) -> tuple[str, str]:
    """Format evidence for verification prompt (token-optimized).
    
    Returns:
        tuple: (ai_overview, formatted_evidence)
    """
    # Extract AI overview
    ai_overview = "None available."
    for item in evidence_items:
        overview = (item.get("ai_overview") or "").strip()
        if overview:
            ai_overview = overview
            break
    
    # Format evidence items
    evidence_lines: List[str] = []
    for i, item in enumerate(evidence_items, 1):
        title = (item.get("title") or "Untitled").strip()
        url = (item.get("url") or "").strip()
        source = (item.get("source_domain") or "unknown").strip()
        score = float(item.get("score", 0.0))
        
        # Use only clean text snippets (no raw HTML content)
        text = (item.get("text") or "").strip()
        
        if text:
            body = text
            tag = "[Snippet]"
        else:
            body = "(Empty)"
            tag = "[Empty]"
        
        evidence_lines.append(
            f"--- [{i}] {source} (score:{score:.2f}) {tag} ---\n"
            f"{title}\n{url}\n{body}\n"
        )
    
    formatted_evidence = "\n".join(evidence_lines) if evidence_lines else "No evidence."
    
    return ai_overview, formatted_evidence


def format_evidence_summary_for_pivot(
    evidence_items: List[dict],
    max_items: int = 5,
) -> str:
    """Brief evidence summary for pivot decisions (minimal tokens).
    
    Uses titles and truncated snippets only.
    """
    lines: List[str] = []
    for i, item in enumerate(evidence_items[:max_items], 1):
        title = (item.get("title") or "Untitled").strip()
        text = (item.get("text") or "").strip()[:250]
        lines.append(f"{i}. {title}\n   {text}")
    
    return "\n\n".join(lines) if lines else "No evidence."