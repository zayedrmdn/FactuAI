"""
Shared extraction utilities (web scraping, OCR, video/audio transcription).
These live under infrastructure to avoid cross-feature coupling.
"""

from .scraper import scrape_article
from .base import extract_sentences
from .ocr import extract_text_from_image
from .video import extract_text_from_video, extract_audio_from_video, audio_to_text

__all__ = [
    "scrape_article",
    "extract_sentences",
    "extract_text_from_image",
    "extract_text_from_video",
    "extract_audio_from_video",
    "audio_to_text",
]
