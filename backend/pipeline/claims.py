"""
Claim extraction module.

Extracts factual claims from text. Handles both single and multi-claim inputs.
"""

import re
from typing import List

from utils.logging import get_logger
from services.llm import chat

logger = get_logger(__name__)


def extract_claims(text: str, max_claims: int = 5, llm: str = None, model_id: str = None) -> List[str]:
    """
    Extract up to max_claims factual claims from text.

    Args:
        text: Input text
        max_claims: Maximum number of claims to extract
        llm: Optional LLM provider
        model_id: Optional model identifier

    Returns:
        List of extracted claim strings
    """
    logger.debug(f"[EXTRACTION] Extracting up to {max_claims} claims from {len(text)} chars")

    # Text length based extraction scaling
    # At least 2 claims for any text longer than 40 words
    word_count = len(text.split())
    if word_count < 20:
        dynamic_max = 1
    elif word_count < 80:
        dynamic_max = min(max_claims, 2)
    else:
        dynamic_max = min(max_claims, max(2, word_count // 60))

    system = f"""
Extract up to {dynamic_max} factual claims from the following text.
List them one per line, starting each with a dash (-).

Rules:
- Extract only what the text actually states
- Do not add commentary or evaluations
- Do not say that there are no claims
- Keep each claim in one sentence

Now extract from:
""".strip()

    try:
        # Token budget relative to text length
        dynamic_max_tokens = min(2048, max(512, len(text) // 2 + 300))

        response = chat(
            system,
            text,
            provider=llm,
            model_id=model_id,
            max_tokens=dynamic_max_tokens,
        )

        claims = []
        for raw in response.split("\n"):
            line = raw.strip()
            if not line:
                continue

            # Remove common bullet formats
            line = re.sub(r'^(\s*[-*•>\[]+\s*|\s*\d+\.\s+)', '', line).strip()

            # Reject useless lines
            lower = line.lower()
            if lower.startswith("source:") or lower.startswith("note:") or lower.startswith("disclaimer:"):
                continue
            if "no factual claims" in lower or "no claims found" in lower:
                continue

            # Minimum validity length
            if len(line.split()) < 3:
                continue

            claims.append(line)
            if len(claims) >= dynamic_max:
                break

        logger.debug(f"[EXTRACTION] Extracted {len(claims)} claims")

        # Fallback retry for empty extraction
        if len(claims) == 0:
            logger.warning("[EXTRACTION] No claims extracted, retrying with simpler prompt")

            simple_system = f"""
Extract up to {dynamic_max} factual claims from the text. Keep them brief. List with dashes.
""".strip()

            retry = chat(
                simple_system,
                text,
                provider=llm,
                model_id=model_id,
                max_tokens=dynamic_max_tokens,
            )

            for raw in retry.split("\n"):
                line = raw.strip()
                if not line:
                    continue

                line = re.sub(r'^(\s*[-*•>\[]+\s*|\s*\d+\.\s+)', '', line).strip()
                if len(line.split()) < 3:
                    continue

                claims.append(line)
                if len(claims) >= dynamic_max:
                    break

        return claims

    except Exception as e:
        logger.error(f"[EXTRACTION] Claim extraction failed: {e}")
        raise


__all__ = ["extract_claims"]