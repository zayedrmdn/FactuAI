from __future__ import annotations

from typing import List, Optional, TypedDict


class EvidenceSnippet(TypedDict, total=False):
    text: str
    url: str
    title: Optional[str]
    source_domain: str
    score: float
    ai_overview: Optional[str]  # Tavily's AI-generated summary
    content: Optional[str]  # Full raw content from source


class ClaimVerdict(TypedDict):
    verdict: str
    confidence: float
    reasoning: str
    evidence: List[EvidenceSnippet]


class IntentClaim(TypedDict):
    claim_text: str
    search_query: str
    verification_question: Optional[str]
