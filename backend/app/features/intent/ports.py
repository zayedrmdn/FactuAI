# Full Path: backend/app/features/intent/ports.py
from __future__ import annotations

from typing import Protocol

from app.contracts.types import IntentResult


class ClaimParserPort(Protocol):
    async def parse_and_route(
        self,
        *,
        text: str,
        max_claims: int,
        provider: str,
        model: str,
    ) -> IntentResult:
        ...
