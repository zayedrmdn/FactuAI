# Full path: backend/app/features/verification/adapters/openai_compatible.py
"""
OpenAI-compatible claim verifier with circuit breaker protection.

This adapter uses LangChain's ChatOpenAI for structured LLM calls.
Circuit breaker protects against cascading failures when the LLM API is down.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.contracts.types import ClaimVerdict, EvidenceSnippet
from app.core.logging import get_logger
from app.core.settings import Settings
from app.core.circuit_breaker import (
    circuit_breaker,
    CircuitOpenError,
    LLM_CIRCUIT_CONFIG,
)
from app.features.verification.ports import ClaimVerifierPort

logger = get_logger(__name__)


class _LLMClaimVerdict(BaseModel):
    verdict: str = Field(
        description="One of: true, false, mostly_true, mostly_false, mixed, unverifiable.",
        pattern=r"^(true|false|mostly_true|mostly_false|mixed|unverifiable)$",
    )
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(min_length=1)


_SYSTEM = (
    "You are a fact-checking AI. Given a claim and evidence snippets, return a structured response. "
    "Do not include markdown or any extra text outside the required format.\n\n"
    "Rules:\n"
    "- If evidence is insufficient, verdict MUST be 'unverifiable'.\n"
    "- Confidence MUST be between 0.0 and 1.0.\n\n"
    "{format_instructions}"
)


class OpenAICompatibleClaimVerifier(ClaimVerifierPort):
    """Native async verifier using an OpenAI-compatible chat endpoint.
    
    Features:
    - Circuit breaker protection against LLM API failures
    - Automatic retries with exponential backoff for transient errors
    - Graceful degradation when circuit is open or LLM fails
    """

    def __init__(self, *, settings: Settings):
        self._settings = settings

    async def verify_claim(
        self,
        *,
        claim: str,
        evidence: List[EvidenceSnippet],
        provider: str,
        model: str,
    ) -> ClaimVerdict:
        claim_clean = (claim or "").strip()
        if not claim_clean:
            return ClaimVerdict(
                verdict="unverifiable",
                confidence=0.0,
                reasoning="Empty claim.",
                evidence=evidence,
            )

        if not evidence:
            return ClaimVerdict(
                verdict="unverifiable",
                confidence=0.0,
                reasoning="No evidence available to verify this claim.",
                evidence=[],
            )

        api_key = (self._settings.llm_api_key or "").strip()
        base_url = (self._settings.llm_api_base_url or "").strip()

        if not api_key:
            logger.info("[VERIFY] LLM key missing; returning unverifiable")
            return ClaimVerdict(
                verdict="unverifiable",
                confidence=0.0,
                reasoning="LLM is not configured (missing LLM_API_KEY / OPENROUTER_API_KEY).",
                evidence=evidence,
            )

        try:
            return await self._verify_with_circuit_breaker(
                claim_clean=claim_clean,
                evidence=evidence,
                model=model,
                api_key=api_key,
                base_url=base_url,
            )
        except CircuitOpenError as exc:
            logger.error(f"[VERIFY] Circuit breaker open: {exc}")
            return ClaimVerdict(
                verdict="unverifiable",
                confidence=0.0,
                reasoning=f"LLM service temporarily unavailable. Please try again in {int(exc.retry_after)} seconds.",
                evidence=evidence,
            )
        except Exception as exc:
            # Catch all other exceptions from the circuit breaker wrapper
            # (these are errors that exhausted retries)
            logger.warning(f"[VERIFY] LLM call failed after retries: {exc}")
            return ClaimVerdict(
                verdict="unverifiable",
                confidence=0.0,
                reasoning="LLM call failed. Please try again later.",
                evidence=evidence,
            )

    @circuit_breaker("llm_verifier", LLM_CIRCUIT_CONFIG)
    async def _verify_with_circuit_breaker(
        self,
        *,
        claim_clean: str,
        evidence: List[EvidenceSnippet],
        model: str,
        api_key: str,
        base_url: str,
    ) -> ClaimVerdict:
        """Internal method wrapped with circuit breaker for LLM calls."""
        # Format evidence for LLM input (token-optimized: exclude URLs)
        ev_lines: list[str] = []
        for item in evidence:
            title = (item.get("title") or "Untitled").strip()
            txt = (item.get("text") or "").strip()
            # Only include title + text, not URL (saves ~20-30 tokens per item)
            ev_lines.append(f"- {title}\n  {txt}".strip())

        evidence_text = "\n".join(ev_lines)

        parser = PydanticOutputParser(pydantic_object=_LLMClaimVerdict)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM),
                (
                    "human",
                    "Claim:\n{claim}\n\nEvidence:\n{evidence}\n",
                ),
            ]
        )

        try:
            llm = ChatOpenAI(
                model=model,
                temperature=0.2,
                api_key=api_key,
                base_url=base_url or None,
            )
        except TypeError:
            llm = ChatOpenAI(
                model=model,
                temperature=0.2,
                openai_api_key=api_key,
                openai_api_base=base_url or None,
            )

        chain = prompt | llm | parser

        # Execute the LLM call - exceptions will be caught by circuit breaker
        result: _LLMClaimVerdict = await chain.ainvoke(
            {
                "claim": claim_clean,
                "evidence": evidence_text,
                "format_instructions": parser.get_format_instructions(),
            }
        )

        return ClaimVerdict(
            verdict=result.verdict,
            confidence=float(result.confidence),
            reasoning=result.reasoning,
            evidence=evidence,
        )
