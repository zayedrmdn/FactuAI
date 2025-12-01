"""
Stateless helper functions shared across services and modules.
Contains utility functions that don't belong to any specific feature.
"""
import re
import hashlib
import json
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


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