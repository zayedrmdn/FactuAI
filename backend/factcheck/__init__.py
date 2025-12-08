"""
FactuAI Fact-checking Module

Public API for fact-checking functionality.
"""

from factcheck.llm_client import initialize, generate, chat, is_available, get_available_providers
from factcheck.evidence import collect_evidence, search_google, search_newsapi, scrape_article
from factcheck.pipeline import check_text, check_text_stream, verify_claim, extract_claims, detect_intent
from factcheck.ocr import extract_text_from_image
from factcheck.video import extract_text_from_video

__all__ = [
    # LLM
    "initialize",
    "generate",
    "chat",
    "is_available",
    "get_available_providers",
    # Evidence
    "collect_evidence",
    "search_google",
    "search_newsapi",
    "scrape_article",
    # Pipeline
    "check_text",
    "check_text_stream",
    "verify_claim",
    "extract_claims",
    "detect_intent",
    # Processing
    "extract_text_from_image",
    "extract_text_from_video",
]
