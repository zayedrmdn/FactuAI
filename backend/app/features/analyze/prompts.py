# Full path: backend/app/features/analyze/prompts.py
"""
Strategist Pipeline Prompts.

Multi-angle query generation and rich context verification for robust fact-checking.
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
