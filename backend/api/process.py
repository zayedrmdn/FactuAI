from flask import Blueprint, request, jsonify, Response
from services.factcheck_service import PipelineOrchestrator
import json
import time
from core.logging import logger

bp = Blueprint("process", __name__, url_prefix="/api")
orchestrator = PipelineOrchestrator()

@bp.post("/process")
def process():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    include_summary = data.get("include_summary", False)
    progressive = data.get("progressive", False)
    
    # Extract AI model parameters from request
    provider = data.get("provider")
    model_id = data.get("model_id")
    model_display_name = data.get("model_display_name", "default")
    temperature = data.get("temperature")
    max_tokens = data.get("max_tokens")
    top_p = data.get("top_p")
    system_prompt = data.get("system_prompt")

    # Log incoming request with model info
    logger.info("[API] Received fact-check request:")
    if provider:
        logger.info("[API] MODEL SWITCH DETECTED:")
        logger.info(f"  Provider: {provider}")
        logger.info(f"  Model ID: {model_id}")
        logger.info(f"  Display Name: {model_display_name}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Max Tokens: {max_tokens}")
        logger.info(f"  Top-P: {top_p}")
    else:
        logger.info("[API] Using default model configuration")
    logger.info(f"  Progressive Mode: {progressive}")
    logger.debug(f"[API] Processing request: include_summary={include_summary}, progressive={progressive}")

    if not text:
        return jsonify({"error": "text field required"}), 400
    
    # Prepare model config for orchestrator
    model_config = {
        "provider": provider,
        "model_id": model_id,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "system_prompt": system_prompt,
    } if provider else None

    if progressive:
        return Response(
            process_progressive(text, include_summary, model_config),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    # Single‑shot pipeline run (summary + claims)
    results_bundle = orchestrator.check_text(text, max_claims=5, model_config=model_config)

    # If validation failed propagate the error
    if "validation_error" in results_bundle:
        return jsonify(
            {
                "error": results_bundle["validation_error"],
                "suggestion": results_bundle["suggestion"],
            }
        ), 400

    response_data = {
        "results": results_bundle,
        "summary": results_bundle.get("summary", "") if include_summary else "",
    }

    return jsonify(response_data)

def process_progressive(text: str, include_summary: bool, model_config: dict = None):
    """SSE generator for progressive delivery."""

    def send_event(event_type: str, data: dict):
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

    try:
        # Log model being used
        if model_config:
            logger.info(f"[API] Initializing fact-check with model: {model_config.get('model_id', 'default')}")
        
        # Use the generator version of check_text
        for event in orchestrator.check_text_generator(text, max_claims=5, model_config=model_config):
            
            if event["type"] == "phase":
                yield send_event("phase", {
                    "message": event["message"],
                    "progress": event["progress"],
                    "claim_index": event.get("claim_index")
                })
                
            elif event["type"] == "summary":
                if include_summary:
                    yield send_event("summary", {"summary": event["summary"]})
                    
            elif event["type"] == "result":
                yield send_event("result", {"result": event["result"]})
                
            elif event["type"] == "error":
                yield send_event("error", {"message": event["message"]})
                yield send_event("complete", {})
                return

        # Phase 4: complete
        yield send_event("phase", {"message": "Complete!", "progress": 100})
        time.sleep(0.1)
        yield send_event("complete", {})

    except Exception as e:
        logger.exception(f"[API] Error in progressive processing: {e}")
        yield send_event("error", {"message": str(e)})
        yield send_event("complete", {})
