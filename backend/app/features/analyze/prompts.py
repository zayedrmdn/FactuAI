# Full path: backend/app/features/analyze/prompts.py
"""
Strategist Pipeline Prompts.

Multi-angle query generation and rich context verification for robust fact-checking.
Includes Pivot Loop for iterative research when initial results reveal new concepts.
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
You are a strategic search query planner for fact-checking. Given a claim, generate 3 distinct search queries that approach the claim from different verification angles.

Your goal is to maximize the chance of finding high-quality evidence by searching from multiple perspectives:

1. **Factual Query**: A direct, neutral search to find primary sources (news, official statements, data).
   - Example: "Eiffel Tower height meters official"

2. **Hoax Query**: A debunking-focused search to find fact-check articles or exposés.
   - Include terms like: hoax, debunked, false, fact-check, snopes, politifact
   - Example: "Eiffel Tower height claim debunked OR hoax OR fact-check"

3. **Scientific Query**: An academic/research search to find studies or expert analysis.
   - Include terms like: study, research, scientific, journal, expert, analysis
   - Example: "Eiffel Tower height scientific measurement study"

Rules:
- Each query must be distinct and cover a different angle.
- Queries should be concise (5-10 words).
- Include the core subject from the claim in all queries.
- Do NOT include the word "claim" in queries.
"""

QUERY_GENERATION_HUMAN = """\
Generate 3 multi-angle search queries for the following claim:

Claim: {claim}
"""


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
You are an analytical researcher reviewing initial search results. Your task is to determine if \
the evidence reveals a NEW specific entity, product, event, or concept that is CENTRAL to \
understanding the claim but was NOT directly mentioned in the claim itself.

WHEN TO PIVOT (needs_pivot = True):
- Evidence points to a specific product/announcement that the rumor misrepresents.
  Example: Claim about "Air Wi-Fi" reveals it's actually about the "Tesla Pi Phone" hoax.
- Evidence mentions a specific event/date that is crucial but wasn't in original search.
  Example: Claim about "vaccine danger" reveals a specific retracted study (Wakefield 1998).
- A proper noun (person, company, product) emerges as the "root cause" of the rumor.

WHEN NOT TO PIVOT (needs_pivot = False):
- The evidence already directly addresses the claim.
- The claim is simple factual question (dates, measurements, definitions).
- No new specific entity is discovered - just general information.
- The concept mentioned is already covered by the original search queries.

Rules:
- Be CONSERVATIVE. Only pivot for truly new, specific entities.
- The pivot_query should be SHORT and SPECIFIC (3-6 words).
- If unsure, set needs_pivot to False.
"""

PIVOT_CHECK_HUMAN = """\
ORIGINAL CLAIM: {claim}

ORIGINAL SEARCH QUERIES USED: {queries}

EVIDENCE FOUND:
{evidence_summary}

Based on the evidence, is there a NEW specific entity/concept that requires additional research?
"""


# ============================================================================
# VERIFICATION - "Read Deeply"
# ============================================================================

VERIFICATION_SYSTEM = """\
You are a rigorous fact-checking AI. Analyze the claim using the provided evidence and return a structured verdict.

Evidence includes:
- **AI Overview**: A synthesized summary from search results (if available).
- **Full Content**: Complete article text when available.
- **Snippets**: Brief excerpts from sources.

Verdict Options:
- `true`: The claim is accurate based on strong evidence.
- `false`: The claim is inaccurate based on strong contradicting evidence.
- `mostly_true`: The claim is largely accurate but has minor inaccuracies.
- `mostly_false`: The claim is largely inaccurate but has minor truths.
- `mixed`: Evidence is contradictory; parts are true and parts are false.
- `unverifiable`: Insufficient evidence to make a determination.

Rules:
1. Prioritize primary sources (official, academic, reputable news) over secondary sources.
2. If evidence conflicts, explain the conflict in your reasoning.
3. If evidence is insufficient, verdict MUST be `unverifiable`.
4. Confidence MUST be between 0.0 and 1.0.
5. Cite specific evidence in your reasoning.
6. Do not include markdown or extra text outside the required format.

{format_instructions}
"""

VERIFICATION_HUMAN = """\
Claim: {claim}

=== AI OVERVIEW (Synthesized Summary) ===
{ai_overview}

=== EVIDENCE SOURCES ===
{evidence}
"""


def format_evidence_for_verification(
    evidence_items: List[dict],
    max_content_length: int = 2000,
) -> tuple[str, str]:
    """Format evidence items for the verification prompt.
    
    Returns:
        tuple: (ai_overview, formatted_evidence)
    """
    # Extract AI overview (take first non-empty one)
    ai_overview = "No AI overview available."
    for item in evidence_items:
        overview = (item.get("ai_overview") or "").strip()
        if overview:
            ai_overview = overview
            break
    
    # Format individual evidence items
    evidence_lines: List[str] = []
    for i, item in enumerate(evidence_items, 1):
        title = (item.get("title") or "Untitled").strip()
        url = (item.get("url") or "").strip()
        source = (item.get("source_domain") or "unknown").strip()
        score = float(item.get("score", 0.0))
        
        # Prefer full content, fall back to text snippet
        content = (item.get("content") or "").strip()
        text = (item.get("text") or "").strip()
        
        if content and len(content) > 50:
            # Truncate long content
            body = content[:max_content_length]
            if len(content) > max_content_length:
                body += "... [truncated]"
            content_type = "[Full Content]"
        elif text:
            body = text
            content_type = "[Snippet]"
        else:
            body = "(No content available)"
            content_type = "[Empty]"
        
        evidence_lines.append(
            f"--- Source {i} ({source}, score: {score:.2f}) {content_type} ---\n"
            f"Title: {title}\n"
            f"URL: {url}\n"
            f"Content:\n{body}\n"
        )
    
    formatted_evidence = "\n".join(evidence_lines) if evidence_lines else "No evidence available."
    
    return ai_overview, formatted_evidence


def format_evidence_summary_for_pivot(
    evidence_items: List[dict],
    max_items: int = 5,
) -> str:
    """Format evidence as a brief summary for pivot decision.
    
    Uses titles and snippets only to keep token count low.
    """
    lines: List[str] = []
    for i, item in enumerate(evidence_items[:max_items], 1):
        title = (item.get("title") or "Untitled").strip()
        text = (item.get("text") or "").strip()[:300]
        lines.append(f"{i}. {title}\n   {text}")
    
    return "\n\n".join(lines) if lines else "No evidence found."

