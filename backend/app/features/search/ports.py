from __future__ import annotations

from typing import List, Optional, Protocol

from app.contracts.types import EvidenceSnippet


class SearchPort(Protocol):
    async def hybrid_search(
        self,
        *,
        query: str,
        max_results: int = 8,
        providers: Optional[List[str]] = None,
        verification_question: Optional[str] = None,
    ) -> List[EvidenceSnippet]:
        ...
