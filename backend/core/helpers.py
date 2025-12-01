"""
Stateless helper functions shared across services and modules.
Contains utility functions that don't belong to any specific feature.
"""
import re
import hashlib
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


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