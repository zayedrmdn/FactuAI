from __future__ import annotations

from urllib.parse import urlparse


def select_model(provider: str, *, openrouter_model: str) -> str:
    """Select the appropriate model based on provider.
    
    Currently only openrouter is supported. This function exists to maintain
    extensibility for future providers while keeping the code simple.
    """
    return openrouter_model


def normalize_url(url: str, fallback: str) -> str:
    """Normalize URL, preserving both HTTP(S) and internal protocol URLs.
    
    Args:
        url: URL to normalize (may be http://, https://, or internal://)
        fallback: Fallback URL if the input is invalid
        
    Returns:
        The original URL if it's valid (http/https/internal protocol), otherwise fallback
    """
    if url and (url.startswith("http") or url.startswith("internal://")):
        return url
    return fallback


def extract_domain(url: str, default: str = "web") -> str:
    try:
        parsed = urlparse(url)
        if parsed.netloc:
            return parsed.netloc
    except Exception:
        pass
    return default


def map_verdict(verdict: str) -> str:
    mapping = {
        "true": "true",
        "false": "false",
        "mostly_true": "mostly_true",
        "mostly false": "mostly_false",
        "mostly_false": "mostly_false",
        "mixed": "mixed",
        "unverifiable": "unverifiable",
    }
    normalized = (verdict or "").lower()
    return mapping.get(normalized, "unverifiable")
