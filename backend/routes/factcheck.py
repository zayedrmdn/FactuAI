"""
Fact-checking API Routes

Consolidated routes for:
- Text fact-checking
- Image OCR + fact-checking
- Video speech-to-text + fact-checking
"""

from flask import Blueprint, request, jsonify
import tempfile
import os

from factcheck import pipeline, ocr, video
from utils.logging import get_logger
from utils.helpers import handle_errors, ValidationError

logger = get_logger(__name__)

bp = Blueprint("factcheck", __name__, url_prefix="/api")


@bp.post("/factcheck")
@handle_errors
def factcheck():
    """
    Fact-check text input.
    
    POST /api/factcheck
    Body: {
        "text": "claim to check",
        "max_claims": 5,  // optional
        "llm": "openrouter"  // optional provider
    }
    """
    data = request.get_json(silent=True) or {}
    
    text = data.get("text", data.get("claim", "")).strip()
    if not text:
        raise ValidationError("text or claim field is required")
    
    max_claims = data.get("max_claims", 5)
    llm = data.get("llm")
    
    logger.info(f"[API] Fact-check request: {len(text)} chars")
    
    result = pipeline.check_text(text, max_claims=max_claims, llm=llm)
    return jsonify(result)


@bp.post("/factcheck-image")
@handle_errors
def factcheck_image():
    """
    Extract text from image and fact-check it.
    
    POST /api/factcheck-image
    Form-data: image file
    """
    if 'image' not in request.files:
        raise ValidationError("No image file provided")
    
    file = request.files['image']
    if file.filename == '':
        raise ValidationError("No file selected")
    
    # Save to temp file
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        image_path = tmp.name
        file.save(image_path)
    
    try:
        # Extract text
        logger.info(f"[API] OCR extraction from: {file.filename}")
        text = ocr.extract_text_from_image(image_path)
        
        if not text:
            return jsonify({"error": "No text found in image"}), 400
        
        # Fact-check extracted text
        result = pipeline.check_text(text)
        result["extracted_text"] = text
        
        return jsonify(result)
        
    finally:
        # Cleanup
        if os.path.exists(image_path):
            os.remove(image_path)


@bp.post("/factcheck-video")
@handle_errors
def factcheck_video():
    """
    Extract speech from video and fact-check it.
    
    POST /api/factcheck-video
    Form-data: video file
    """
    if 'video' not in request.files:
        raise ValidationError("No video file provided")
    
    file = request.files['video']
    if file.filename == '':
        raise ValidationError("No file selected")
    
    # Save to temp file
    video_ext = os.path.splitext(file.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp_video:
        video_path = tmp_video.name
        file.save(video_path)
    
    try:
        # Extract text from video
        logger.info(f"[API] Video processing: {file.filename}")
        text = video.extract_text_from_video(video_path)
        
        if not text:
            return jsonify({"error": "No speech found in video"}), 400
        
        # Fact-check extracted text
        result = pipeline.check_text(text)
        result["extracted_text"] = text
        
        return jsonify(result)
        
    finally:
        # Cleanup
        if os.path.exists(video_path):
            os.remove(video_path)


@bp.post("/extract-image-text")
@handle_errors
def extract_image_text():
    """
    Extract text from image only (no fact-checking).
    
    POST /api/extract-image-text
    Form-data: image file
    """
    if 'image' not in request.files:
        raise ValidationError("No image file provided")
    
    file = request.files['image']
    if file.filename == '':
        raise ValidationError("No file selected")
    
    ext = os.path.splitext(file.filename)[1] or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        image_path = tmp.name
        file.save(image_path)
    
    try:
        text = ocr.extract_text_from_image(image_path)
        return jsonify({"text": text})
        
    finally:
        if os.path.exists(image_path):
            os.remove(image_path)


@bp.post("/extract-video-text")
@handle_errors
def extract_video_text():
    """
    Extract speech from video only (no fact-checking).
    
    POST /api/extract-video-text
    Form-data: video file
    """
    if 'video' not in request.files:
        raise ValidationError("No video file provided")
    
    file = request.files['video']
    if file.filename == '':
        raise ValidationError("No file selected")
    
    video_ext = os.path.splitext(file.filename)[1] or '.mp4'
    with tempfile.NamedTemporaryFile(delete=False, suffix=video_ext) as tmp:
        video_path = tmp.name
        file.save(video_path)
    
    try:
        text = video.extract_text_from_video(video_path)
        return jsonify({"text": text})
        
    finally:
        if os.path.exists(video_path):
            os.remove(video_path)
