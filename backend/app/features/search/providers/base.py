from __future__ import annotations

from typing import List, Optional, Protocol

from app.contracts.types import EvidenceSnippet


class SearchProvider(Protocol):
    name: str

    async def search(
        self,
        *,
        query: str,
        max_results: int,
        verification_question: Optional[str] = None,
    ) -> List[EvidenceSnippet]:
        ...
