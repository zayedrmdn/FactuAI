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

    logger.debug(f"[API] Processing request: include_summary={include_summary}, progressive={progressive}")

    if not text:
        return jsonify({"error": "text field required"}), 400

    if progressive:
        return Response(
            process_progressive(text, include_summary),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Content-Type",
            },
        )

    # Single‑shot pipeline run (summary + claims)
    results_bundle = orchestrator.check_text(text, max_claims=5)

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

def process_progressive(text: str, include_summary: bool):
    """SSE generator for progressive delivery."""

    def send_event(event_type: str, data: dict):
        return f"data: {json.dumps({'type': event_type, **data})}\n\n"

    try:
        yield send_event("phase", {"message": "Extracting claims...", "progress": 10})
        time.sleep(0.1)

        bundle = orchestrator.check_text(text, max_claims=5)

        if "validation_error" in bundle:
            yield send_event("error", {"message": bundle["validation_error"]})
            yield send_event("complete", {})
            return

        claims_results = bundle.get("results", [])
        summary_text = bundle.get("summary", "")

        if not claims_results:
            yield send_event("phase", {"message": "No claims found", "progress": 100})
            yield send_event("complete", {})
            return

        # Phase 2: send summary if requested
        if include_summary and summary_text:
            yield send_event("phase", {"message": "Summary generated", "progress": 25})
            yield send_event("summary", {"summary": summary_text})
            time.sleep(0.1)

        # Phase 3: stream claim results
        progress_start = 25 if include_summary else 15
        progress_end = 95
        total = len(claims_results)

        for idx, claim_obj in enumerate(claims_results, start=1):
            claim_progress = progress_start + (idx * (progress_end - progress_start) / total)

            yield send_event(
                "phase",
                {
                    "message": f"Processing claim {idx}...",
                    "progress": claim_progress,
                    "claim_index": idx - 1,
                },
            )

            yield send_event("result", {"result": claim_obj})
            time.sleep(0.1)

        # Phase 4: complete
        yield send_event("phase", {"message": "Complete!", "progress": 100})
        time.sleep(0.1)
        yield send_event("complete", {})

    except Exception as e:
        logger.debug(f"[API] Error in progressive processing: {e}")
        yield send_event("error", {"message": str(e)})
