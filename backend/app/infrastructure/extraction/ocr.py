"""
OCR Service for FactuAI.

Simple wrapper around PIL + pytesseract for extracting text from images.
"""

from PIL import Image
import pytesseract
from utils.logging import get_logger

logger = get_logger(__name__)


def extract_text_from_image(image_path: str) -> str:
    """
    Extract text from an image file using OCR.
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text string
    """
    logger.debug(f"[OCR] Extracting text from: {image_path}")
    
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        img.close()
        
        text = text.strip()
        logger.debug(f"[OCR] Extracted {len(text)} characters")
        return text
        
    except Exception as e:
        logger.error(f"[OCR] Failed to extract text: {e}")
        return ""


__all__ = ["extract_text_from_image"]
