# Full path: backend/app/core/constants.py
"""
Domain constants for search quality control.

SOCIAL_MEDIA_DOMAINS: Excluded from external searches to reduce noise.
TRUSTED_NEWS_DOMAINS: High-quality sources (reference for filtering/prioritization).
"""
from typing import List

# Domains to exclude from search results (social media, user-generated content)
SOCIAL_MEDIA_DOMAINS: List[str] = [
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "twitter.com",
    "x.com",
    "reddit.com",
    "linkedin.com",
    "pinterest.com",
    "quora.com",
    "tumblr.com",
    "snapchat.com",
    "weibo.com",
    "telegram.org",
    "discord.com",
    "youtube.com",
    "medium.com",
    "substack.com",
    "vimeo.com",
    "wikipedia.org",
]

# Trusted news/fact-checking domains (for prioritization/filtering)
TRUSTED_NEWS_DOMAINS: List[str] = [
    "reuters.com",
    "apnews.com",
    "bloomberg.com",
    "bbc.com",
    "cnn.com",
    "npr.org",
    "pbs.org",
    "wsj.com",
    "nytimes.com",
    "washingtonpost.com",
    "ft.com",
    "economist.com",
    "nature.com",
    "science.org",
    "gov",
    "edu",
    "mil",
    "snopes.com",
    "politifact.com",
]
