from typing import Optional, TypedDict


class EvidenceSnippet(TypedDict):
    text: str
    url: str
    title: Optional[str]
    source_domain: str
    score: float
