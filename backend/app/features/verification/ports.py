from __future__ import annotations

from typing import List, Protocol

from app.contracts.types import ClaimVerdict, EvidenceSnippet


class ClaimVerifierPort(Protocol):
    async def verify_claim(
        self,
        *,
        claim: str,
        evidence: List[EvidenceSnippet],
        provider: str,
        model: str,
    ) -> ClaimVerdict:
        ...
