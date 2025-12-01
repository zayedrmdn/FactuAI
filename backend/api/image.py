# backend/routes/image.py
"""
Image‑to‑text (OCR) endpoint with AI detection.

POST /api/extract-text
Form‑data field name:  image   (PNG/JPG/…)
Returns:              { "text": "…extracted text…", "ai_score": 0.85, "ai_confidence": "high" }
"""

from flask import Blueprint, request, jsonify
from common.logging_config import logger

import pytesseract
from PIL import Image
import io
import requests
import os
from urllib.parse import urlparse

bp_image = Blueprint("image", __name__, url_prefix="/api")

# SightEngine API configuration
SIGHTENGINE_API_USER = os.getenv('SIGHTENGINE_API_USER', '567750877')
SIGHTENGINE_API_SECRET = os.getenv('SIGHTENGINE_API_SECRET', 'uPhUNBr79YarPbKrJKj3oUMZ9oaW6Ryq')

def detect_ai_image(image_bytes: bytes) -> dict:
    """Detect if image is AI-generated using SightEngine API"""
    try:
        logger.info("Making SINGLE SightEngine API call (file-based)")
        
        params = {
            'models': 'genai',
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }
        
        files = {'media': ('image.jpg', image_bytes, 'image/jpeg')}
        
        response = requests.post(
            'https://api.sightengine.com/1.0/check.json', 
            files=files, 
            data=params,
            timeout=30
        )
        
        logger.info(f"SightEngine API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"SightEngine response: {result}")
            
            if result.get('status') == 'success':
                operations_used = result.get('request', {}).get('operations', 'unknown')
                logger.info(f"SightEngine operations used: {operations_used} (this is normal for genai model)")
                
                ai_score = result.get('type', {}).get('ai_generated', 0.0)
                return {
                    'ai_score': ai_score,
                    'ai_percentage': round(ai_score * 100, 1),
                    'success': True,
                    'operations_used': operations_used
                }
        
        logger.warning(f"SightEngine API error: {response.text}")
        return {'success': False, 'error': 'AI detection service unavailable'}
        
    except Exception as e:
        logger.error(f"AI detection failed: {e}")
        return {'success': False, 'error': 'AI detection failed'}

def detect_ai_image_url(image_url: str) -> dict:
    """Detect if image is AI-generated using SightEngine API with URL"""
    try:
        logger.info(f"Making SINGLE SightEngine API call (URL-based) for: {image_url}")
        
        params = {
            'url': image_url,
            'models': 'genai',
            'api_user': SIGHTENGINE_API_USER,
            'api_secret': SIGHTENGINE_API_SECRET
        }
        
        response = requests.get(
            'https://api.sightengine.com/1.0/check.json', 
            params=params,
            timeout=30
        )
        
        logger.info(f"SightEngine API response status: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"SightEngine response: {result}")
            
            if result.get('status') == 'success':
                operations_used = result.get('request', {}).get('operations', 'unknown')
                logger.info(f"SightEngine operations used: {operations_used} (this is normal for genai model)")
                
                ai_score = result.get('type', {}).get('ai_generated', 0.0)
                return {
                    'ai_score': ai_score,
                    'ai_percentage': round(ai_score * 100, 1),
                    'success': True,
                    'operations_used': operations_used
                }
        
        logger.warning(f"SightEngine API error: {response.text}")
        return {'success': False, 'error': 'AI detection service unavailable'}
        
    except Exception as e:
        logger.error(f"AI detection failed: {e}")
        return {'success': False, 'error': 'AI detection failed'}

def download_image_from_url(image_url: str) -> bytes:
    """Download image from URL and return bytes"""
    try:
        # Validate URL
        parsed = urlparse(image_url)
        if not parsed.scheme or not parsed.netloc:
            raise ValueError("Invalid URL format")
        
        response = requests.get(image_url, timeout=30, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        response.raise_for_status()
        
        # Check if response is an image
        content_type = response.headers.get('content-type', '')
        if not content_type.startswith('image/'):
            raise ValueError("URL does not point to an image")
        
        return response.content
        
    except Exception as e:
        logger.error(f"Failed to download image from URL: {e}")
        raise

@bp_image.post("/extract-text")
def extract_text() -> tuple:
    """Run OCR on an uploaded image or image URL and detect if it's AI-generated."""
    logger.info("=== NEW API REQUEST: /api/extract-text ===")
    
    image_bytes = None
    image_url = None
    
    # ── 1  Check input type (file or URL) ──────────────────────────
    if request.content_type and 'application/json' in request.content_type:
        # URL input
        logger.info("Processing image URL request")
        data = request.get_json()
        image_url = data.get('url', '').strip()
        
        if not image_url:
            logger.warning("No image URL provided")
            return jsonify({"error": "No image URL provided"}), 400
        
        logger.info(f"Processing image URL: {image_url}")
        
        try:
            image_bytes = download_image_from_url(image_url)
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"Failed to process image URL: {e}")
            return jsonify({"error": f"Failed to process image URL: {str(e)}"}), 400
    else:
        # File upload
        logger.info("Processing file upload request")
        if "image" not in request.files:
            logger.warning("No image field in request")
            return jsonify({"error": "No image field named 'image'"}), 400

        file = request.files["image"]
        if file.filename == "":
            logger.warning("No file selected")
            return jsonify({"error": "No file selected"}), 400

        logger.info(f"Processing uploaded file: {file.filename}")

        # ── 2  Load the image in‑memory ─────────────────────────────
        try:
            # Read raw bytes → PIL.Image
            image_bytes = file.read()
            image = Image.open(io.BytesIO(image_bytes))
        except Exception as e:
            logger.error(f"OCR: could not read image – {e}")
            return jsonify({"error": "Unsupported image format"}), 400

    # ── 3  OCR with Tesseract ───────────────────────────────────
    logger.info("Starting OCR processing...")
    try:
        text = pytesseract.image_to_string(image).strip()
        logger.info(f"OCR completed - extracted {len(text)} characters")
    except Exception as e:
        logger.error(f"OCR failed: {e}", exc_info=True)
        return jsonify({"error": f"OCR processing failed: {e}"}), 500

    # ── 4  AI Detection ─────────────────────────────────────────
    logger.info("Starting AI detection...")
    if image_url:
        # Use URL-based AI detection for better accuracy
        logger.info("Using URL-based AI detection")
        ai_result = detect_ai_image_url(image_url)
    else:
        # Use bytes-based AI detection for uploaded files
        logger.info("Using bytes-based AI detection")
        ai_result = detect_ai_image(image_bytes)
    
    logger.info(f"AI detection completed: {ai_result}")
    
    # Prepare response
    response_data = {"text": text}
    
    if ai_result['success']:
        response_data.update({
            "ai_score": ai_result['ai_score'],
            "ai_percentage": ai_result['ai_percentage']
        })
        logger.info(f"AI detection successful: {ai_result['ai_percentage']}%")
    else:
        logger.warning(f"AI detection failed: {ai_result.get('error', 'Unknown error')}")
        # Don't fail the whole request if AI detection fails
        response_data.update({
            "ai_score": None,
            "ai_percentage": None,
            "ai_error": ai_result.get('error', 'AI detection unavailable')
        })

    if not text and not ai_result['success']:
        logger.error("Both OCR and AI detection failed")
        return jsonify({"error": "No text found and AI detection failed"}), 400

    logger.info(f"Request completed successfully - OCR: {len(text)} chars, AI: {ai_result.get('ai_percentage', 'N/A')}%")
    return jsonify(response_data)
