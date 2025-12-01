# backend/services/ocr.py
from PIL import Image
import pytesseract
from core.logging import logger

class OCRService:
    """Simple wrapper around PIL + pytesseract for image→text."""
    def __init__(self):
        # you could validate dependencies here
        logger.debug("[OCR] initialized")

    def extract_text(self, image_path: str) -> str:
        """Load image from disk and return OCR’d text."""
        logger.debug(f"[OCR] extracting text from {image_path}")
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        img.close()
        logger.debug(f"[OCR] extracted {len(text)} characters")
        return text.strip()