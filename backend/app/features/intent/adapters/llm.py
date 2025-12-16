# Full Path: backend/app/features/intent/adapters/llm.py
"""
LLM-based Intent Adapter for Tiered Intelligence Architecture.

This adapter uses a "fast & cheap" LLM (Tier 1) to extract structured claims
from unstructured text. It replaces the regex-based NativeIntentAdapter for
improved robustness with real-world input.

Usage:
    Set INTENT_ADAPTER=app.features.intent.adapters.llm.LLMIntentAdapter
    Optionally configure INTENT_LLM_MODEL, INTENT_LLM_API_BASE_URL, INTENT_LLM_API_KEY
"""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from app.contracts.types import IntentClaim
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.intent.ports import ClaimParserPort

logger = get_logger(__name__)


class _ClaimOutput(BaseModel):
    """Structured output for a single extracted claim."""

    claim_text: str = Field(
        description="The exact factual claim to verify, stated clearly and concisely."
    )
    search_query: str = Field(
        description="A web search query to find evidence for or against this claim."
    )
    verification_question: Optional[str] = Field(
        default=None,
        description="A yes/no question to determine if the claim is true.",
    )


class _ClaimListOutput(BaseModel):
    """Structured output for the full list of extracted claims."""

    claims: List[_ClaimOutput] = Field(
        default_factory=list,
        description="List of distinct, verifiable factual claims extracted from the text.",
    )


_SYSTEM_PROMPT = """\
You are a claim extraction assistant. Your task is to analyze text and extract distinct, \
verifiable factual claims.

Rules:
1. Extract only FACTUAL claims that can be verified with evidence (true/false).
2. Ignore opinions, predictions, questions, and rhetorical statements.
3. Each claim should be self-contained and understandable without context.
4. Generate a concise web search query to find evidence for each claim.
5. Generate a verification question that can be answered with yes/no.
6. Do NOT extract duplicate or overlapping claims.
7. If no verifiable claims exist, return an empty list.

Examples of GOOD claims:
- "The Eiffel Tower is 330 meters tall."
- "Apple was founded in 1976."
- "Water boils at 100°C at sea level."

Examples of BAD claims (do NOT extract):
- "I think the weather will be nice." (opinion/prediction)
- "Is Python a good language?" (question)
- "Everyone knows about climate change." (vague/rhetorical)
"""


class LLMIntentAdapter(ClaimParserPort):
    """LLM-based intent parser using structured output.

    Uses LangChain with async invocation for claim extraction.
    Falls back to main LLM config if intent-specific env vars are not set.
    """

    def __init__(self, *, settings: Settings):
        self._settings = settings

    async def parse_and_route(
        self,
        *,
        text: str,
        max_claims: int,
        provider: str,
        model: str,
    ) -> List[IntentClaim]:
        text_clean = (text or "").strip()
        if not text_clean:
            return []

        # Determine which LLM configuration to use
        # Priority: Request model (frontend) > Intent-specific config > Main LLM config
        api_base = (self._settings.intent_llm_api_base_url or "").strip()
        api_key = (self._settings.intent_llm_api_key or "").strip()
        
        # *** KEY CHANGE: Frontend model takes priority over settings ***
        # If frontend sends a model, use it. Otherwise, fall back to intent-specific config.
        request_model = (model or "").strip()
        settings_model = (self._settings.intent_llm_model or "").strip()
        intent_model = request_model or settings_model

        # Fall back to main LLM config if intent-specific not provided
        if not api_base:
            api_base = (self._settings.llm_api_base_url or "").strip()
        if not api_key:
            api_key = (self._settings.llm_api_key or "").strip()

        if not api_key:
            logger.warning("[INTENT-LLM] No API key configured; returning empty claims")
            return []

        model_source = "frontend" if request_model else "settings"
        logger.info(f"[INTENT-LLM] Using model: {intent_model} (source: {model_source})")

        try:
            return await self._extract_claims(
                text=text_clean,
                max_claims=max_claims,
                model=intent_model,
                api_key=api_key,
                api_base=api_base,
            )
        except Exception as exc:
            logger.error(f"[INTENT-LLM] Extraction failed: {exc}")
            # Graceful degradation: return empty list on failure
            return []

    async def _extract_claims(
        self,
        *,
        text: str,
        max_claims: int,
        model: str,
        api_key: str,
        api_base: str,
    ) -> List[IntentClaim]:
        """Extract claims using LLM with structured output."""
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _SYSTEM_PROMPT),
                (
                    "human",
                    "Extract up to {max_claims} verifiable factual claims from the following text:\n\n{text}",
                ),
            ]
        )

        try:
            llm = ChatOpenAI(
                model=model,
                temperature=0.1,  # Low temperature for consistent extraction
                api_key=api_key,
                base_url=api_base or None,
            )
        except TypeError:
            # Fallback for older langchain-openai versions
            llm = ChatOpenAI(
                model=model,
                temperature=0.1,
                openai_api_key=api_key,
                openai_api_base=api_base or None,
            )

        # Use with_structured_output for guaranteed schema compliance
        structured_llm = llm.with_structured_output(_ClaimListOutput)
        chain = prompt | structured_llm

        result: _ClaimListOutput = await chain.ainvoke(
            {
                "text": text,
                "max_claims": max_claims,
            }
        )

        # Convert to IntentClaim format
        items: List[IntentClaim] = []
        for claim in result.claims[:max_claims]:  # Ensure we don't exceed max
            items.append(
                IntentClaim(
                    claim_text=claim.claim_text,
                    search_query=claim.search_query,
                    verification_question=claim.verification_question,
                )
            )

        logger.info(f"[INTENT-LLM] Extracted {len(items)} claim(s)")
        return items
