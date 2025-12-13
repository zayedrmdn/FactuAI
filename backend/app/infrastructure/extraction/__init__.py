"""
Shared extraction utilities (web scraping, OCR, video/audio transcription).
Lives under infrastructure to avoid cross-feature coupling per VSA rules.
"""

from .base import extract_sentences
from .ocr import extract_text_from_image
from .scraper import scrape_article
from .video import audio_to_text, extract_audio_from_video, extract_text_from_video

__all__ = [
    "extract_sentences",
    "extract_text_from_image",
    "scrape_article",
    "audio_to_text",
    "extract_audio_from_video",
    "extract_text_from_video",
]
