"""
Pipeline module for FactuAI.

Provides complete fact-checking pipeline orchestration.
"""

from pipeline.orchestrator import (
    check_text,
    check_text_stream,
    PHASE_DETECTING_INTENT,
    PHASE_EXTRACTING_CLAIMS,
    PHASE_GENERATING_SUMMARY,
    PHASE_VERIFYING_CLAIM,
    PHASE_COLLECTING_EVIDENCE,
)
from pipeline.intent import detect_intent
from pipeline.claims import extract_claims
from pipeline.verification import verify_claim
from pipeline.summary import summarize_input

__all__ = [
    "check_text",
    "check_text_stream",
    "detect_intent",
    "extract_claims",
    "verify_claim",
    "summarize_input",
    "PHASE_DETECTING_INTENT",
    "PHASE_EXTRACTING_CLAIMS",
    "PHASE_GENERATING_SUMMARY",
    "PHASE_VERIFYING_CLAIM",
    "PHASE_COLLECTING_EVIDENCE",
]
