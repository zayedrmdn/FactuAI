from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, HttpUrl, UUID4, conint, constr


class AnalyzeRequest(BaseModel):
    text: constr(min_length=5, max_length=5000)
    provider: Optional[Literal["openrouter", "nvidia"]] = "nvidia"
    max_claims: conint(ge=1, le=8) = 3
    enable_web_search: bool = True
    enable_kb: bool = True


class EvidenceItem(BaseModel):
    snippet: str
    source_url: HttpUrl
    source_title: Optional[str]
    source_domain: str
    relevance_score: float


class ClaimResult(BaseModel):
    claim_text: str
    verdict: Literal["true", "false", "mostly_true", "mostly_false", "mixed", "unverifiable"]
    confidence: float
    reasoning: str
    evidence: List[EvidenceItem]


class AnalyzeResponse(BaseModel):
    request_id: UUID4
    model_used: str
    latency_ms: int
    claims: List[ClaimResult]


class ErrorResponse(BaseModel):
    detail: str
