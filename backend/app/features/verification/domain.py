from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Evidence:
    snippet: str
    source_url: str
    source_title: Optional[str]
    source_domain: str
    relevance_score: float


@dataclass(frozen=True)
class ClaimAnalysis:
    claim_text: str
    verdict: str
    confidence: float
    reasoning: str
    evidence: list[Evidence]
