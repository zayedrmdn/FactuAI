from __future__ import annotations

from typing import List, Optional, TypedDict


class EvidenceSnippet(TypedDict):
    text: str
    url: str
    title: Optional[str]
    source_domain: str
    score: float


class ClaimVerdict(TypedDict):
    verdict: str
    confidence: float
    reasoning: str
    evidence: List[EvidenceSnippet]


class IntentClaim(TypedDict):
    claim_text: str
    search_query: str
    verification_question: Optional[str]
