"""
Pipeline orchestrator for FactuAI.

Coordinates the complete fact-checking workflow.
"""

from typing import Dict, List, Any, Generator, Optional

from utils.logging import get_logger
from utils.helpers import ValidationError, PipelineError
from pipeline.intent import detect_intent
from pipeline.claims import extract_claims
from pipeline.verification import verify_claim
from pipeline.summary import summarize_input
from config import EVIDENCE_MULTI_CLAIM_COUNT, EVIDENCE_STREAMING_COUNT, EVIDENCE_DEFAULT_COUNT

logger = get_logger(__name__)

# Pipeline phase messages
PHASE_DETECTING_INTENT = "Detecting intent..."
PHASE_EXTRACTING_CLAIMS = "Extracting claims..."
PHASE_GENERATING_SUMMARY = "Generating summary..."
PHASE_VERIFYING_CLAIM = "Verifying claim..."
PHASE_COLLECTING_EVIDENCE = "Collecting evidence..."

# Evidence selection constants
TOP_K_MULTI_CLAIM = EVIDENCE_MULTI_CLAIM_COUNT  # Conservative for multi-claim processing
TOP_K_STREAMING = EVIDENCE_STREAMING_COUNT     # More comprehensive for streaming responses
TOP_K_DEFAULT = EVIDENCE_DEFAULT_COUNT          # Default for single claims


def _log_model_usage(stage: str, provider: Optional[str], model_id: Optional[str]) -> None:
    """Centralized trace for which model is used at each pipeline stage."""
    logger.info(f"[PIPELINE_MODEL] stage={stage} provider={provider or 'default'} model={model_id or 'env_default'}")


def _resolve_models(pipeline_models: Optional[Dict[str, Dict[str, str]]], fallback_provider: Optional[str], fallback_model: Optional[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """KISS helper: normalize per-stage provider/model with simple fallbacks."""
    pipeline_models = pipeline_models or {}
    def stage(key: str) -> Dict[str, Optional[str]]:
        cfg = pipeline_models.get(key) or {}
        return {
            "provider": cfg.get("provider") or fallback_provider,
            "model_id": cfg.get("model_id") or fallback_model,
        }
    return {
        "intent": stage("intent"),
        "extraction": stage("extraction"),
        "reasoning": stage("reasoning"),
        "summary": stage("summary"),
    }


def check_text(
    text: str,
    max_claims: int = 5,
    llm: str = None,
    pipeline_models: Optional[Dict[str, Dict[str, str]]] = None,
    enabled_search_providers: Optional[List[str]] = None,
    num_google: int = 5,
    num_news: int = 5,
) -> Dict[str, Any]:
    """
    Complete fact-checking pipeline for text input.
    
    Args:
        text: Input text to fact-check
        max_claims: Maximum claims to extract for multi-claim inputs
        llm: Optional LLM provider
        pipeline_models: Per-stage model configuration
        enabled_search_providers: List of enabled search providers ['google', 'newsapi']
        num_google: Number of Google results to fetch
        num_news: Number of NewsAPI results to fetch
        
    Returns:
        Dict with 'summary', 'results', optional 'validation_error'
    """
    logger.info(f"[PIPELINE] Starting fact-check for {len(text)} chars")
    
    try:
        models = _resolve_models(pipeline_models, fallback_provider=llm, fallback_model=None)
        intent_cfg = models["intent"]
        extraction_cfg = models["extraction"]
        reasoning_cfg = models["reasoning"]
        summary_cfg = models["summary"]

        # Detect intent and extract search queries
        _log_model_usage("intent_query_detection", intent_cfg.get("provider", llm), intent_cfg.get("model_id"))
        detection_result = detect_intent(
            text,
            llm=intent_cfg.get("provider", llm),
            model_id=intent_cfg.get("model_id"),
        )
        intent = detection_result["intent"]
        google_query = detection_result["google_query"]
        newsapi_query = detection_result["newsapi_query"]
        verification_question = detection_result.get("verification_question", "")
        logger.info(f"[PIPELINE] Detected intent: {intent}, Google: {google_query[:50]}..., NewsAPI: {newsapi_query[:50]}...")
        if verification_question:
            logger.info(f"[PIPELINE] Verification Question: {verification_question}")
        
        # Handle non-verifiable inputs
        if intent in ["opinion", "nonsense", "instructional"]:
            return {
                "summary": "",
                "results": [],
                "validation_error": "Input is not verifiable.",
                "suggestion": "Please enter a factual claim, question, or news paragraph."
            }
        
        # Handle different intent types
        if intent == "fact_question":
            # Treat question as a claim
            _log_model_usage("verify_single", reasoning_cfg.get("provider", llm), reasoning_cfg.get("model_id"))
            result = verify_claim(
                text,
                google_query,
                newsapi_query,
                llm=reasoning_cfg.get("provider", llm),
                model_id=reasoning_cfg.get("model_id"),
                enabled_search_providers=enabled_search_providers,
                num_google=num_google,
                num_news=num_news,
                verification_question=verification_question,
            )
            
            # Generate comprehensive summary with verification results
            _log_model_usage("summary", summary_cfg.get("provider", llm), summary_cfg.get("model_id"))
            summary = summarize_input(
                text,
                llm=summary_cfg.get("provider", llm),
                model_id=summary_cfg.get("model_id"),
                evidence_results=[result]
            )
            
            return {
                "summary": summary,
                "results": [{"claim": text, **result}]
            }
        
        elif intent == "multi_claim":
            # Extract and verify multiple claims
            _log_model_usage("claim_extraction", extraction_cfg.get("provider", llm), extraction_cfg.get("model_id"))
            claims = extract_claims(
                text,
                max_claims,
                llm=extraction_cfg.get("provider", llm),
                model_id=extraction_cfg.get("model_id"),
            )
            
            if not claims:
                return {
                    "summary": "",
                    "results": [],
                    "validation_error": "No factual claims found.",
                    "suggestion": "Try a different text."
                }
            
            results = []
            for claim in claims:
                _log_model_usage("verify_claim", reasoning_cfg.get("provider", llm), reasoning_cfg.get("model_id"))
                logger.info(f"[PIPELINE] Verifying claim with top_k={TOP_K_MULTI_CLAIM} (multi-claim mode)")
                result = verify_claim(
                    claim,
                    google_query,
                    newsapi_query,
                    llm=reasoning_cfg.get("provider", llm),
                    model_id=reasoning_cfg.get("model_id"),
                    num_google=num_google,
                    num_news=num_news,
                    top_k=TOP_K_MULTI_CLAIM,
                    enabled_search_providers=enabled_search_providers,
                    verification_question=verification_question,
                )
                results.append({"claim": claim, **result})
            
            # Generate comprehensive summary with all verification results
            _log_model_usage("summary", summary_cfg.get("provider", llm), summary_cfg.get("model_id"))
            summary = summarize_input(
                text,
                llm=summary_cfg.get("provider", llm),
                model_id=summary_cfg.get("model_id"),
                evidence_results=results
            )
            
            return {
                "summary": summary,
                "results": results
            }
        
        else:  # fact_claim
            _log_model_usage("verify_single", reasoning_cfg.get("provider", llm), reasoning_cfg.get("model_id"))
            result = verify_claim(
                text,
                google_query,
                newsapi_query,
                llm=reasoning_cfg.get("provider", llm),
                model_id=reasoning_cfg.get("model_id"),
                enabled_search_providers=enabled_search_providers,
                verification_question=verification_question,
            )
            
            # Generate comprehensive summary with verification results
            _log_model_usage("summary", summary_cfg.get("provider", llm), summary_cfg.get("model_id"))
            summary = summarize_input(
                text,
                llm=summary_cfg.get("provider", llm),
                model_id=summary_cfg.get("model_id"),
                evidence_results=[result]
            )
            
            return {
                "summary": summary,
                "results": [{"claim": text, **result}]
            }
        
    except Exception as e:
        logger.error(f"[PIPELINE] Unexpected error: {e}", exc_info=True)
        return {
            "summary": "",
            "results": [],
            "validation_error": f"Internal error: {str(e)}"
        }


def check_text_stream(
    text: str,
    max_claims: int = 5,
    llm: str = None,
    pipeline_models: Optional[Dict[str, Dict[str, str]]] = None,
    enabled_search_providers: Optional[List[str]] = None,
    num_google: int = 5,
    num_news: int = 5,
    num_tavily: int = 5,
) -> Generator[Dict[str, Any], None, None]:
    """
    Streaming version of check_text that yields progress events.
    
    Yields dicts with 'type' key:
    - {"type": "phase", "message": str, "progress": int}
    - {"type": "summary", "summary": str}
    - {"type": "result", "result": dict}
    - {"type": "error", "message": str, "suggestion": str}
    
    Args:
        text: Input text
        max_claims: Maximum claims to extract
        llm: Optional LLM provider
        pipeline_models: Per-stage model configuration
        enabled_search_providers: List of enabled search providers ['google', 'newsapi', 'tavily']
        num_google: Number of Google results to fetch
        num_news: Number of NewsAPI results to fetch
        num_tavily: Number of Tavily results to fetch
    """
    try:
        models = _resolve_models(pipeline_models, fallback_provider=llm, fallback_model=None)
        intent_cfg = models["intent"]
        extraction_cfg = models["extraction"]
        reasoning_cfg = models["reasoning"]
        summary_cfg = models["summary"]

        # Intent detection
        yield {"type": "phase", "message": PHASE_DETECTING_INTENT, "progress": 5}
        _log_model_usage("intent_detection", intent_cfg.get("provider", llm), intent_cfg.get("model_id"))
        detection_result = detect_intent(
            text,
            llm=intent_cfg.get("provider", llm),
            model_id=intent_cfg.get("model_id"),
        )
        intent = detection_result["intent"]
        google_query = detection_result["google_query"]
        newsapi_query = detection_result["newsapi_query"]
        verification_question = detection_result.get("verification_question")
        logger.info(f"[PIPELINE] Detected intent: {intent}, Google: {google_query[:50]}..., NewsAPI: {newsapi_query[:50]}...")
        if verification_question:
            logger.info(f"[PIPELINE] Verification Question: {verification_question}")
        
        if intent in ["opinion", "nonsense", "instructional"]:
            yield {
                "type": "error",
                "message": "Input is not verifiable.",
                "suggestion": "Please enter a factual claim, question, or news paragraph."
            }
            return
        
        # Handle different intents and collect results for summary
        all_results = []
        
        if intent == "fact_question":
            yield {"type": "phase", "message": PHASE_VERIFYING_CLAIM, "progress": 30}
            _log_model_usage("verify_single", reasoning_cfg.get("provider", llm), reasoning_cfg.get("model_id"))
            result = verify_claim(
                text,
                google_query,
                newsapi_query,
                llm=reasoning_cfg.get("provider", llm),
                model_id=reasoning_cfg.get("model_id"),
                num_google=num_google,
                num_news=num_news,
                num_tavily=num_tavily,
                enabled_search_providers=enabled_search_providers,
                verification_question=verification_question,
            )
            all_results.append(result)
            yield {"type": "result", "result": {"claim": text, **result}}
        
        elif intent == "multi_claim":
            yield {"type": "phase", "message": PHASE_EXTRACTING_CLAIMS, "progress": 15}
            _log_model_usage("claim_extraction", extraction_cfg.get("provider", llm), extraction_cfg.get("model_id"))
            claims = extract_claims(
                text,
                max_claims,
                llm=extraction_cfg.get("provider", llm),
                model_id=extraction_cfg.get("model_id"),
            )
            
            if not claims:
                yield {
                    "type": "error",
                    "message": "No factual claims found.",
                    "suggestion": "Try a different text."
                }
                return
            
            # Verify each claim with progress updates
            total = len(claims)
            for i, claim in enumerate(claims):
                progress = 25 + int((i / total) * 70)
                yield {
                    "type": "phase",
                    "message": f"Verifying claim {i+1}/{total}...",
                    "progress": progress,
                    "claim_index": i + 1,
                    "total_claims": total
                }
                
                _log_model_usage("verify_claim", reasoning_cfg.get("provider", llm), reasoning_cfg.get("model_id"))
                logger.info(f"[PIPELINE] Verifying claim with top_k={TOP_K_STREAMING} (streaming mode)")
                result = verify_claim(
                    claim,
                    google_query,
                    newsapi_query,
                    llm=reasoning_cfg.get("provider", llm),
                    model_id=reasoning_cfg.get("model_id"),
                    num_google=num_google,
                    num_news=num_news,
                    num_tavily=num_tavily,
                    top_k=TOP_K_STREAMING,
                    enabled_search_providers=enabled_search_providers,
                    verification_question=verification_question,
                )
                all_results.append(result)
                yield {"type": "result", "result": {"claim": claim, **result}, "claim_index": i + 1}
        
        else:  # fact_claim
            yield {"type": "phase", "message": PHASE_VERIFYING_CLAIM, "progress": 30}
            _log_model_usage("verify_single", reasoning_cfg.get("provider", llm), reasoning_cfg.get("model_id"))
            logger.info(f"[PIPELINE] Verifying single claim with top_k={TOP_K_DEFAULT} (default)")
            result = verify_claim(
                text,
                google_query,
                newsapi_query,
                llm=reasoning_cfg.get("provider", llm),
                model_id=reasoning_cfg.get("model_id"),
                num_google=num_google,
                num_news=num_news,
                num_tavily=num_tavily,
                enabled_search_providers=enabled_search_providers,
                verification_question=verification_question,
            )
            all_results.append(result)
            yield {"type": "result", "result": {"claim": text, **result}}

        # Summary generation (after all verification is complete)
        yield {"type": "phase", "message": PHASE_GENERATING_SUMMARY, "progress": 95}
        _log_model_usage("summary", summary_cfg.get("provider", llm), summary_cfg.get("model_id"))
        summary = summarize_input(
            text,
            llm=summary_cfg.get("provider", llm),
            model_id=summary_cfg.get("model_id"),
            evidence_results=all_results
        )
        yield {"type": "summary", "summary": summary}

        # Signal completion
        yield {"type": "complete", "progress": 100}
            
    except Exception as e:
        logger.error(f"[PIPELINE] Stream error: {e}", exc_info=True)
        yield {
            "type": "error",
            "message": f"Internal error: {str(e)}"
        }


__all__ = [
    "check_text",
    "check_text_stream",
    "PHASE_DETECTING_INTENT",
    "PHASE_EXTRACTING_CLAIMS",
    "PHASE_GENERATING_SUMMARY",
    "PHASE_VERIFYING_CLAIM",
    "PHASE_COLLECTING_EVIDENCE",
]
