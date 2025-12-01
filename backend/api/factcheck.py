from flask import Blueprint, request, jsonify
from services.factcheck_service import PipelineOrchestrator

bp_fact = Blueprint("factcheck", __name__, url_prefix="/api")
orchestrator = PipelineOrchestrator()

@bp_fact.post("/factcheck")
def factcheck():
    data = request.get_json(silent=True) or {}
    claim = data.get("claim", "").strip()
    if not claim:
        return jsonify({"error": "claim field is required"}), 400

    result = orchestrator.check(claim)
    return jsonify(result)
