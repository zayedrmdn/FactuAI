from __future__ import annotations

import re
from typing import List

from app.contracts.types import IntentClaim
from app.core.logging import get_logger
from app.features.intent.ports import ClaimParserPort

logger = get_logger(__name__)


_SPLIT_RE = re.compile(r"\n+|[•\-\*]\s+")


class NativeIntentAdapter(ClaimParserPort):
    """Native intent parser.

    Produces claims plus a conservative search query and optional verification question.
    This is intentionally deterministic (no legacy scripts, no sync I/O).
    """

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

        parts = [p.strip() for p in _SPLIT_RE.split(text_clean) if p and p.strip()]
        if not parts:
            parts = [text_clean]

        claims: list[str] = []
        for p in parts:
            if p not in claims:
                claims.append(p)
            if len(claims) >= max(1, int(max_claims)):
                break

        items: List[IntentClaim] = []
        for claim in claims:
            items.append(
                IntentClaim(
                    claim_text=claim,
                    search_query=claim,
                    verification_question=f"Is the following claim true? {claim}",
                )
            )

        logger.info(f"[INTENT] Extracted {len(items)} claim(s)")
        return items
