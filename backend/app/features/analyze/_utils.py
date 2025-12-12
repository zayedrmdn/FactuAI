from __future__ import annotations

from urllib.parse import urlparse


def select_model(provider: str, *, openrouter_model: str, nvidia_model: str) -> str:
    if provider == "openrouter":
        return openrouter_model
    return nvidia_model


def normalize_url(url: str, fallback: str) -> str:
    if url and url.startswith("http"):
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
