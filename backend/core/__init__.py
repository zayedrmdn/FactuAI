"""
FactuAI Core Module

This module contains centralized configuration, logging, and exception handling
for the FactuAI backend application.
"""

from core.config import Config
from core.logging import logger, get_logger
from core.exceptions import (
    FactuAIException,
    LLMClientError,
    ClassifierError,
    SearchError,
    ScrapingError,
    EvidenceError,
    ExtractionError,
    PipelineError,
    ValidationError,
    DatabaseError,
    AuthenticationError,
)

__all__ = [
    "Config",
    "logger",
    "get_logger",
    "FactuAIException",
    "LLMClientError",
    "ClassifierError",
    "SearchError",
    "ScrapingError",
    "EvidenceError",
    "ExtractionError",
    "PipelineError",
    "ValidationError",
    "DatabaseError",
    "AuthenticationError",
]
