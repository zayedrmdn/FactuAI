"""
Shared Utilities for FactuAI Backend

Consolidated from:
- core/helpers.py
- core/exceptions.py

Contains utility functions and custom exceptions used throughout the application.
"""

import re
import hashlib
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse
from functools import wraps
from flask import jsonify
import logging


# ==========================================================================
# Custom Exceptions
# ==========================================================================

class FactuAIException(Exception):
    """Base exception class for all FactuAI-specific errors."""
    pass


class LLMClientError(FactuAIException):
    """Raised when LLM client operations fail."""
    pass


class ClassifierError(FactuAIException):
    """Raised when classification operations fail."""
    pass


class SearchError(FactuAIException):
    """Raised when search operations fail."""
    pass


class ScrapingError(FactuAIException):
    """Raised when web scraping operations fail."""
    pass


class EvidenceError(FactuAIException):
    """Raised when evidence filtering/ranking operations fail."""
    pass


class ExtractionError(FactuAIException):
    """Raised when claim extraction operations fail."""
    pass


class PipelineError(FactuAIException):
    """Raised when pipeline orchestration fails."""
    pass


class ValidationError(FactuAIException):
    """Raised when input validation fails."""
    pass


class DatabaseError(FactuAIException):
    """Raised when database operations fail."""
    pass


class AuthenticationError(FactuAIException):
    """Raised when authentication operations fail."""
    pass


# ==========================================================================
# Helper Functions
# ==========================================================================

# Junk detection patterns for filtering low-quality content
JUNK_PATTERNS = [
    r"^(share|tweet|email|print|comment|subscribe|sign up|log in|menu|navigation)",
    r"(cookie|privacy policy|terms of service|advertisement|sponsored)",
    r"^(related|recommended|popular|trending|most read)",
    r"(all rights reserved|copyright \d{4})",
    r"^\s*$",  # Empty or whitespace only
]


def is_junk(text: str) -> bool:
    """
    Detect if text is likely junk/boilerplate content.
    
    Args:
        text: Text to check
        
    Returns:
        True if text appears to be junk/boilerplate
    """
    if not text or len(text.strip()) < 10:
        return True
    
    text_lower = text.lower().strip()
    
    for pattern in JUNK_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return True
    
    return False


def attribution_tail(claim: str) -> str:
    """
    Extract attribution information from a claim (e.g., speaker name).
    
    Args:
        claim: The claim text to analyze
        
    Returns:
        Attribution string or empty string if none found
    """
    # Look for common attribution patterns
    patterns = [
        r"(?:said|says|stated|claimed|according to)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\s+(?:said|says|stated|claimed)",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, claim)
        if match:
            return match.group(1)
    
    return ""


def is_valid_url(url: str) -> bool:
    """
    Validate if a string is a properly formatted URL.
    
    Args:
        url: String to validate as URL
        
    Returns:
        True if valid URL, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def clean_text(text: str) -> str:
    """
    Clean and normalize text for processing.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text with normalized whitespace
    """
    if not text:
        return ""
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    
    # Remove special characters that might interfere with processing
    text = re.sub(r'[^\w\s\-.,!?;:()\'"@#$%]', '', text)
    
    return text


def validate_email(email: str) -> bool:
    """
    Validate email format.
    
    Args:
        email: Email address to validate
        
    Returns:
        True if valid email format
    """
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def format_sources(evidence_list: List[Dict]) -> List[Dict]:
    """
    Format evidence sources for display.
    
    Args:
        evidence_list: List of evidence dictionaries
        
    Returns:
        Formatted evidence list
    """
    formatted = []
    for evidence in evidence_list:
        formatted.append({
            "text": evidence.get("text", ""),
            "url": evidence.get("url", ""),
            "source": evidence.get("source", ""),
            "score": evidence.get("score", 0.0)
        })
    return formatted


def create_error_response(error_message: str, status_code: int = 400) -> tuple:
    """
    Create standardized error response.
    
    Args:
        error_message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (response, status_code)
    """
    return jsonify({"error": error_message}), status_code


# ==========================================================================
# Decorators
# ==========================================================================

def handle_errors(f):
    """
    Decorator to handle errors in route functions.
    
    Catches FactuAI exceptions and returns formatted error responses.
    """
    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValidationError as e:
            logging.error(f"Validation error: {e}")
            return create_error_response(str(e), 400)
        except AuthenticationError as e:
            logging.error(f"Authentication error: {e}")
            return create_error_response(str(e), 401)
        except FactuAIException as e:
            logging.error(f"FactuAI error: {e}")
            return create_error_response(str(e), 400)
        except Exception as e:
            logging.error(f"Unexpected error: {e}", exc_info=True)
            return create_error_response("Internal server error", 500)
    
    return wrapper


__all__ = [
    # Exceptions
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
    # Functions
    "is_junk",
    "attribution_tail",
    "is_valid_url",
    "clean_text",
    "validate_email",
    "format_sources",
    "create_error_response",
    "handle_errors",
]
