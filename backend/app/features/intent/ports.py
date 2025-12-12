from __future__ import annotations

from typing import List, Protocol

from app.contracts.types import IntentClaim


class ClaimParserPort(Protocol):
    async def parse_and_route(
        self,
        *,
        text: str,
        max_claims: int,
        provider: str,
        model: str,
    ) -> List[IntentClaim]:
        ...
