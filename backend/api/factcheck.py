from flask import Blueprint, request, jsonify
from services.factcheck_service import PipelineOrchestrator
from core.logging import logger

bp_fact = Blueprint("factcheck", __name__, url_prefix="/api")
orchestrator = PipelineOrchestrator()

@bp_fact.post("/factcheck")
def factcheck():
    data = request.get_json(silent=True) or {}
    
    # Extract claim (support both 'claim' and 'text' fields for compatibility)
    claim = data.get("claim", data.get("text", "")).strip()
    if not claim:
        return jsonify({"error": "claim or text field is required"}), 400
    
    # Extract model configuration from frontend
    model_config = data.get("model_config")
    
    # Log incoming request for verification
    if model_config:
        logger.info("[API] Received factcheck request with custom model:")
        logger.info(f"   Provider: {model_config.get('provider', 'default')}")
        logger.info(f"   Model: {model_config.get('model_id', 'default')}")
        logger.info(f"   Temperature: {model_config.get('temperature')}")
        logger.info(f"   Max Tokens: {model_config.get('max_tokens')}")
    else:
        logger.info("📥 [API] Received factcheck request (using default model)")
    
    logger.info(f"   Claim length: {len(claim)} chars")
    
    # Pass model_config to orchestrator
    result = orchestrator.check_text(claim, model_config=model_config)
    return jsonify(result)
