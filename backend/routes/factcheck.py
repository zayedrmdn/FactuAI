"""
Fact-checking API Routes

Consolidated routes for:
- Text fact-checking
- Image OCR + fact-checking
- Video speech-to-text + fact-checking
"""

from flask import Blueprint, request, jsonify, Response, stream_with_context
import tempfile
import os
import json

from factcheck import pipeline, ocr, video
from utils.logging import get_logger
from utils.helpers import handle_errors, ValidationError, create_error_response

logger = get_logger(__name__)

bp = Blueprint("factcheck", __name__, url_prefix="/api")


def _normalize_verdict_to_label(verdict: str) -> str:
    """
    Map backend verdict (UPPERCASE) to frontend label (lowercase).
    
    Backend returns: TRUE, FALSE, MOSTLY_TRUE, MOSTLY_FALSE, MIXED, UNVERIFIABLE
    Frontend expects: true, false, mostly_true, mostly_false, half_true, unknown
    """
    verdict_upper = (verdict or "").upper().strip()
    
    mapping = {
        "TRUE": "true",
        "MOSTLY_TRUE": "mostly_true",
        "MIXED": "half_true",  # Map MIXED to half_true
        "MOSTLY_FALSE": "mostly_false",
        "FALSE": "false",
        "UNVERIFIABLE": "unknown",
        "UNKNOWN": "unknown",
    }
    
    return mapping.get(verdict_upper, "unknown")


def _normalize_results(results: list) -> list:
    """
    Normalize fact-check results to include 'label' field for frontend.
    """
    normalized = []
    for result in results:
        result_copy = dict(result)
        # Add 'label' field from 'verdict'
        if 'verdict' in result_copy:
            result_copy['label'] = _normalize_verdict_to_label(result_copy['verdict'])
        else:
            result_copy['label'] = 'unknown'
        normalized.append(result_copy)
    return normalized


def _build_stage_models(payload: dict, fallback_provider: str, fallback_model_id: str) -> dict:
    """Normalize pipeline model selections with sensible fallbacks."""
    payload = payload or {}
    def stage_cfg(key: str) -> dict:
        cfg = payload.get(key) or {}
        return {
            "provider": cfg.get("provider") or fallback_provider,
            "model_id": cfg.get("model_id") or fallback_model_id,
        }
    return {
        "intent": stage_cfg("intent"),
        "extraction": stage_cfg("extraction"),
        "reasoning": stage_cfg("reasoning"),
        "summary": stage_cfg("summary"),
    }


@bp.post("/validate")
def validate_content():
    """Basic validation before fact-checking (no LLM call - just checks length/format)."""
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    
    if not text:
        return jsonify({
            "isValid": False,
            "error": "Text is required for validation.",
            "suggestion": "Please enter a factual statement or question.",
        }), 400
    
    # Basic length check
    if len(text) < 5:
        return jsonify({
            "isValid": False,
            "error": "Input too short.",
            "suggestion": "Please provide a complete factual claim or question.",
        })
    
    if len(text) > 5000:
        return jsonify({
            "isValid": False,
            "error": "Input too long.",
            "suggestion": "Please limit input to 5000 characters.",
        })
    
    # Quick heuristic checks for obviously invalid input
    if text.lower() in ["test", "hello", "hi", "hey", "...", "???"]:
        return jsonify({
            "isValid": False,
            "error": "Input not meaningful.",
            "suggestion": "Please provide a factual claim or question to verify.",
        })
    
    # All basic checks passed
    return jsonify({"isValid": True, "error": "", "suggestion": ""})


@bp.post("/process")
def process_factcheck():
    """Progressive fact-check endpoint with SSE support."""
    data = request.get_json(silent=True) or {}

    text = (data.get("text") or "").strip()
    if not text:
        return create_error_response("text field is required", 400)

    max_claims = data.get("max_claims", 5)
    include_summary = bool(data.get("include_summary", True))
    progressive = bool(data.get("progressive", True))

    provider = data.get("provider")
    model_id = data.get("model_id")
    pipeline_models = _build_stage_models(data.get("pipeline_models") or {}, provider, model_id)

    logger.info(
        f"[API_REQUEST] POST /api/process | provider={provider} | model={model_id} | progressive={progressive} | include_summary={include_summary}"
    )

    if progressive:
        def event_stream():
            try:
                for event in pipeline.check_text_stream(
                    text,
                    max_claims=max_claims,
                    llm=provider,
                    pipeline_models=pipeline_models,
                ):
                    if not include_summary and event.get("type") == "summary":
                        continue
                    
                    # Normalize result events to include 'label' field
                    if event.get("type") == "result" and "result" in event:
                        result = event["result"]
                        if "verdict" in result:
                            result["label"] = _normalize_verdict_to_label(result["verdict"])
                    
                    yield f"data: {json.dumps(event)}\n\n"
            except Exception as e:
                logger.error(f"[API] Progressive processing failed: {e}", exc_info=True)
                error_event = {"type": "error", "message": "Internal server error"}
                yield f"data: {json.dumps(error_event)}\n\n"

        headers = {
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
        return Response(
            stream_with_context(event_stream()),
            mimetype="text/event-stream",
            headers=headers,
        )

    try:
        result = pipeline.check_text(
            text,
            max_claims=max_claims,
            llm=provider,
            pipeline_models=pipeline_models,
        )
        
        # Normalize results to include 'label' field
        if "results" in result:
            result["results"] = _normalize_results(result["results"])
        
        if not include_summary:
            result.pop("summary", None)
        return jsonify(result)
    except ValidationError as e:
        return create_error_response(str(e), 400)
    except Exception as e:
        logger.error(f"[API] Fact-check failed: {e}", exc_info=True)
        return create_error_response("Internal server error", 500)


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
