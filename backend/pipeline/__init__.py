"""
FactuAI Pipeline Module

Flattened pipeline structure for the fact-checking system.
Migrated from modules/factcheck/claims/ for cleaner imports.
"""

from pipeline.orchestrator import build_evidence, summarise, summarise_input
from pipeline.extraction.extractor import extract_claims_llm
from pipeline.config import MAX_EVIDENCE_WORDS, SIMILARITY_THRESHOLD

__all__ = [
    "build_evidence",
    "summarise",
    "summarise_input",
    "extract_claims_llm",
    "MAX_EVIDENCE_WORDS",
    "SIMILARITY_THRESHOLD",
]
