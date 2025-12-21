from __future__ import annotations

from typing import List, Literal, Optional

from pydantic import BaseModel, HttpUrl, UUID4, conint, constr


class PipelineModelConfig(BaseModel):
    """Configuration for a single pipeline stage model."""
    provider: Optional[str] = None
    model_id: Optional[str] = None
    model_display_name: Optional[str] = None


class PipelineModels(BaseModel):
    """Per-stage model configuration for the analysis pipeline."""
    intent: Optional[PipelineModelConfig] = None
    extraction: Optional[PipelineModelConfig] = None
    summary: Optional[PipelineModelConfig] = None
    reasoning: Optional[PipelineModelConfig] = None


class AnalyzeRequest(BaseModel):
    text: constr(min_length=5, max_length=5000)
    provider: Optional[Literal["openrouter"]] = None  # Uses settings.llm_provider
    model_id: Optional[str] = None  # Frontend model selection (e.g., "tngtech/deepseek-r1t2-chimera:free")
    max_claims: conint(ge=1, le=8) = 3
    enable_web_search: bool = True
    enable_kb: bool = True
    analysis_mode: Literal["quick", "deep"] = "deep"  # Quick: 1 search, no pivot. Deep: full pipeline.
    pipeline_models: Optional[PipelineModels] = None  # Per-stage model configuration


class EvidenceItem(BaseModel):
    snippet: str
    source_url: str  # Changed from HttpUrl to avoid strict validation issues
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
