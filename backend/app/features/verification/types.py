from __future__ import annotations

from typing import List, TypedDict

from app.features.search.types import EvidenceSnippet


class ClaimVerdict(TypedDict):
    verdict: str
    confidence: float
    reasoning: str
    evidence: List[EvidenceSnippet]
