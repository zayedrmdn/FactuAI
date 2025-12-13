# Full path: backend/app/features/verification/adapters/openai_compatible.py
from __future__ import annotations

import json
from typing import List

from openai import AsyncOpenAI

from app.contracts.types import ClaimVerdict, EvidenceSnippet
from app.core.logging import get_logger
from app.core.settings import Settings
from app.features.verification.ports import ClaimVerifierPort

logger = get_logger(__name__)


_SYSTEM = (
    "You are a fact-checking AI. Given a claim and evidence snippets, respond ONLY with valid JSON. "
    "No markdown, no extra text.\n\n"
    "JSON schema:\n"
    "{\n"
    '  "verdict": "true|false|mostly_true|mostly_false|mixed|unverifiable",\n'
    '  "confidence": 0.0,\n'
    '  "reasoning": "..."\n'
    "}\n\n"
    "Rules:\n"
    "- If evidence is insufficient, verdict must be 'unverifiable'.\n"
    "- confidence must be between 0.0 and 1.0.\n"
)


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _parse_json(text: str) -> dict:
    raw = (text or "").strip()
    if not raw:
        return {}

    # Try direct JSON first
    try:
        return json.loads(raw)
    except Exception:
        pass

    # Attempt to extract the first JSON object
    start = raw.find("{")
    end = raw.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except Exception:
            return {}

    return {}


class OpenAICompatibleClaimVerifier(ClaimVerifierPort):
    """Native async verifier using an OpenAI-compatible chat endpoint."""

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

        client = AsyncOpenAI(api_key=api_key, base_url=base_url or None)

        ev_lines = []
        for item in evidence:
            title = item.get("title") or ""
            url = item.get("url") or ""
            txt = item.get("text") or ""
            ev_lines.append(f"- {title} {url}\n  {txt}")

        user = f"Claim:\n{claim_clean}\n\nEvidence:\n" + "\n".join(ev_lines)

        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.2,
            )
            content = (resp.choices[0].message.content or "").strip()
        except Exception as exc:
            logger.warning(f"[VERIFY] LLM call failed: {exc}")
            return ClaimVerdict(
                verdict="unverifiable",
                confidence=0.0,
                reasoning="LLM call failed.",
                evidence=evidence,
            )

        payload = _parse_json(content)
        verdict = str(payload.get("verdict", "unverifiable")).strip().lower()
        mapping = {
            "true": "true",
            "false": "false",
            "mostly_true": "mostly_true",
            "mostly true": "mostly_true",
            "mostly_false": "mostly_false",
            "mostly false": "mostly_false",
            "mixed": "mixed",
            "unverifiable": "unverifiable",
            "unknown": "unverifiable",
        }
        verdict = mapping.get(verdict, "unverifiable")

        try:
            confidence = _clamp01(float(payload.get("confidence", 0.0)))
        except Exception:
            confidence = 0.0

        reasoning = str(payload.get("reasoning", "")).strip() or "No reasoning provided."

        return ClaimVerdict(
            verdict=verdict,
            confidence=confidence,
            reasoning=reasoning,
            evidence=evidence,
        )
