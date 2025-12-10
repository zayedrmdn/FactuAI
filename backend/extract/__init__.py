"""
Extraction module for FactuAI.

Provides content extraction functionality:
- Web scraping (articles, HTML)
- OCR (image text extraction)
- Video processing (transcripts)
"""

from extract.scraper import scrape_article
from extract.base import extract_sentences
from extract.ocr import extract_text_from_image
from extract.video import extract_text_from_video, extract_audio_from_video, audio_to_text

__all__ = [
    "scrape_article",
    "extract_sentences",
    "extract_text_from_image",
    "extract_text_from_video",
    "extract_audio_from_video",
    "audio_to_text",
]
