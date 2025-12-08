# backend/services/factcheck_service.py

import time
from .service_manager import service_manager
from services.ocr import OCRService
from pipeline.orchestrator import build_evidence
from pipeline.summarization import summarise_input_text, summarise_evidence as summarise
from pipeline.extraction.extractor import extract_claims_llm
from services.classifier_intent.intent_parser import detect_intent
from core.logging import logger

# Constants for phase messages
PHASE_GENERATING_SUMMARY = "Generating summary..."
PHASE_VERIFYING_CLAIM = "Verifying claim..."
PHASE_EXTRACTING_CLAIMS = "Extracting claims..."


class PipelineOrchestrator:
    def __init__(self):
        # Use singleton services from service_manager
        self.llm           = service_manager.get_llm_client()
        self.clf           = service_manager.get_classifier()
        self.search_client = service_manager.get_search_client()
        self.ocr           = service_manager.get_ocr_service()
        logger.debug("[PIPELINE] Singleton instance created")

        # map intent strings to handler methods
        self._handlers = {
            "opinion":        self._handle_non_verifiable,
            "nonsense":       self._handle_non_verifiable,
            "instructional":  self._handle_non_verifiable,
            "fact_question":  self._handle_fact_question,
            "news_paragraph": self._handle_multi_claim,
            "multi_claim":    self._handle_multi_claim,
            "fact_claim":     self._handle_fact_claim,
        }

    def start_new_session(self):
        if hasattr(self.llm, "clear_cache"):
            self.llm.clear_cache()
        logger.debug("[PIPELINE] new session started")

    def check_image(self, image_path: str, max_claims: int = 5) -> list[dict]:
        logger.debug("[PIPELINE] check_image() called")
        text = self._extract_text_from_image(image_path)
        return self.check_text(text, max_claims) if text else []

    def _extract_text_from_image(self, image_path: str) -> str:
        logger.debug(f"[PIPELINE] extracting text from image: {image_path}")
        try:
            text = self.ocr.extract_text(image_path)
            logger.debug(f"[PIPELINE] OCR extracted {len(text)} chars")
            return text
        except Exception as e:
            logger.error(f"[PIPELINE] OCR failed: {e}")
            return ""

    def check_text(self, text: str, max_claims: int = 5, model_config: dict = None) -> dict:
        """Synchronous wrapper around the generator for backward compatibility."""
        logger.debug("[PIPELINE] check_text() called (sync wrapper)")
        
        results = []
        summary = ""
        validation_error = None
        suggestion = None

        for event in self.check_text_generator(text, max_claims, model_config):
            if event["type"] == "result":
                results.append(event["result"])
            elif event["type"] == "summary":
                summary = event["summary"]
            elif event["type"] == "error":
                validation_error = event["message"]
                suggestion = event.get("suggestion")
        
        if validation_error:
            return {
                "summary": summary,
                "results": results,
                "validation_error": validation_error,
                "suggestion": suggestion
            }
            
        return {"summary": summary, "results": results}

    def check_text_generator(self, text: str, max_claims: int = 5, model_config: dict = None):
        """Generator that yields progress events and results."""
        logger.debug("[PIPELINE] check_text_generator() started")
        
        # Extract pipeline-specific models if provided
        pipeline_models = model_config.get("pipeline_models", {}) if model_config else {}
        
        # Get task-specific LLM clients
        intent_config = pipeline_models.get("intent", {})
        extraction_config = pipeline_models.get("extraction", {})
        reasoning_config = pipeline_models.get("reasoning", {})
        
        # Intent LLM (lightweight)
        if intent_config.get("provider") and intent_config.get("model_id"):
            try:
                intent_llm = service_manager.get_or_create_llm(
                    intent_config["provider"], 
                    intent_config["model_id"]
                )
                logger.info(f"[PIPELINE] ⚡ Intent: {intent_config.get('model_display_name')} ({intent_config['provider']})")
            except Exception as e:
                logger.error(f"[PIPELINE] Failed to load intent model: {e}. Using tier default.")
                intent_llm = service_manager.get_tiered_llm("intent")
        else:
            logger.debug("[PIPELINE] Using default tier model for intent")
            intent_llm = service_manager.get_tiered_llm("intent")
        
        # Extraction LLM (medium)
        if extraction_config.get("provider") and extraction_config.get("model_id"):
            try:
                extraction_llm = service_manager.get_or_create_llm(
                    extraction_config["provider"],
                    extraction_config["model_id"]
                )
                logger.info(f"[PIPELINE] 📝 Extraction: {extraction_config.get('model_display_name')} ({extraction_config['provider']})")
            except Exception as e:
                logger.error(f"[PIPELINE] Failed to load extraction model: {e}. Using tier default.")
                extraction_llm = service_manager.get_tiered_llm("extraction")
        else:
            logger.debug("[PIPELINE] Using default tier model for extraction")
            extraction_llm = service_manager.get_tiered_llm("extraction")
        
        # Reasoning LLM (heavyweight) - fallback to main model config
        if reasoning_config.get("provider") and reasoning_config.get("model_id"):
            try:
                reasoning_llm = service_manager.get_or_create_llm(
                    reasoning_config["provider"],
                    reasoning_config["model_id"]
                )
                logger.info(f"[PIPELINE] 🧠 Reasoning: {reasoning_config.get('model_display_name')} ({reasoning_config['provider']})")
            except Exception as e:
                logger.error(f"[PIPELINE] Failed to load reasoning model: {e}. Using tier default.")
                reasoning_llm = service_manager.get_tiered_llm("reasoning")
        elif model_config and model_config.get("provider"):
            # Use provided main model config
            try:
                reasoning_llm = service_manager.get_or_create_llm(
                    model_config["provider"],
                    model_config["model_id"]
                )
            except Exception as e:
                logger.error(f"[PIPELINE] Failed to load reasoning model: {e}. Using tier default.")
                reasoning_llm = service_manager.get_tiered_llm("reasoning")
        else:
            reasoning_llm = service_manager.get_tiered_llm("reasoning")
        
        try:
            if hasattr(reasoning_llm, "clear_cache"):
                reasoning_llm.clear_cache()

            from core.logging import log_pipeline_stage
            
            yield {"type": "phase", "message": "Detecting intent...", "progress": 5}
            log_pipeline_stage(logger, "Intent Detection", text[:100], 5)
            intent = detect_intent(text, intent_llm)
            logger.debug(f"[PIPELINE] detected intent: {intent}")

            if intent in ["opinion", "nonsense", "instructional"]:
                yield {
                    "type": "error", 
                    "message": "Input is not verifiable.", 
                    "suggestion": "Please enter a factual claim, question, or news paragraph."
                }
                return

            # Dispatch to appropriate handler generator with task-specific LLMs
            if intent == "fact_question":
                yield from self._handle_fact_question_gen(text, reasoning_llm)
            elif intent in ["news_paragraph", "multi_claim"]:
                yield from self._handle_multi_claim_gen(text, max_claims, reasoning_llm, extraction_llm)
            else: # fact_claim
                yield from self._handle_fact_claim_gen(text, reasoning_llm)
        except Exception as e:
            logger.exception(f"[PIPELINE] Unexpected error in check_text_generator: {e}")
            yield {
                "type": "error",
                "message": f"Internal server error: {str(e)}"
            }

    def _handle_fact_question_gen(self, text: str, llm_client):
        from core.logging import log_pipeline_stage
        logger.info("[PIPELINE] handling fact_question (fallback to claim)")
        
        # 1. Summary
        yield {"type": "phase", "message": PHASE_GENERATING_SUMMARY, "progress": 10}
        log_pipeline_stage(logger, "Question Analysis", text, 10)
        summary = summarise_input_text(text, llm_client)
        yield {"type": "summary", "summary": summary}
        
        # 2. Check
        yield {"type": "phase", "message": PHASE_VERIFYING_CLAIM, "progress": 30}
        log_pipeline_stage(logger, "Question Verification", text, 30)
        result = self._run_check(text, llm_client) # Treating question as claim
        yield {"type": "result", "result": {"claim": text, **result}}

    def _handle_fact_claim_gen(self, text: str, llm_client):
        from core.logging import log_pipeline_stage
        logger.debug("[PIPELINE] handling fact_claim")
        
        # 1. Summary
        yield {"type": "phase", "message": PHASE_GENERATING_SUMMARY, "progress": 10}
        log_pipeline_stage(logger, "Summary Generation", text, 10)
        summary = summarise_input_text(text, llm_client)
        yield {"type": "summary", "summary": summary}
        
        # 2. Check
        yield {"type": "phase", "message": PHASE_VERIFYING_CLAIM, "progress": 30}
        log_pipeline_stage(logger, "Claim Verification", text, 30)
        result = self._run_check(text, llm_client)
        yield {"type": "result", "result": {"claim": text, **result}}

    def _handle_multi_claim_gen(self, text: str, max_claims: int, reasoning_llm, extraction_llm=None):
        from core.logging import log_pipeline_stage
        logger.debug("[PIPELINE] handling multi_claim/news_paragraph")
        
        # Use extraction LLM if provided, otherwise use reasoning LLM
        extract_llm = extraction_llm if extraction_llm else reasoning_llm
        
        # 1. Extract claims
        yield {"type": "phase", "message": PHASE_EXTRACTING_CLAIMS, "progress": 10}
        log_pipeline_stage(logger, "Claim Extraction", text, 10)
        claims = extract_claims_llm(text, max_claims, llm=extract_llm)
        
        if not claims:
             yield {
                "type": "error", 
                "message": "No factual claims found.", 
                "suggestion": "Try a different text."
            }
             return

        # 2. Summary
        yield {"type": "phase", "message": PHASE_GENERATING_SUMMARY, "progress": 20}
        summary = summarise_input_text(text, reasoning_llm)
        yield {"type": "summary", "summary": summary}

        # 3. Process each claim
        total = len(claims)
        for i, claim in enumerate(claims):
            progress = 25 + int((i / total) * 70) # 25% to 95%
            yield {
                "type": "phase", 
                "message": f"Verifying claim {i+1}/{total}...", 
                "progress": progress,
                "claim_index": i
            }
            
            result = self._run_check(
                claim,
                reasoning_llm,
                max_google=2,
                max_news=1
            )
            yield {"type": "result", "result": {"claim": claim, **result}}

    # --- Backwards-compatible synchronous handler wrappers ---
    def _handle_non_verifiable(self, text: str):
        """Synchronous handler for non-verifiable inputs.

        Returns a structure similar to check_text when validation fails.
        """
        return {
            "summary": "",
            "results": [],
            "validation_error": "Input is not verifiable.",
            "suggestion": "Please enter a factual claim, question, or news paragraph."
        }

    def _handle_fact_question(self, text: str):
        """Synchronous wrapper for handling a fact_question.

        Reuses the existing synchronous check_text wrapper for compatibility.
        """
        return self.check_text(text, max_claims=1)

    def _handle_fact_claim(self, text: str):
        """Synchronous wrapper for handling a fact_claim."""
        return self.check_text(text, max_claims=1)

    def _handle_multi_claim(self, text: str, max_claims: int = 5):
        """Synchronous wrapper for handling multi-claim inputs."""
        return self.check_text(text, max_claims=max_claims)

    def _run_check(
        self,
        claim:       str,
        llm_client,
        max_google:  int = None,
        max_news:    int = None
    ) -> dict:
        logger.debug(f"[PIPELINE] running check for: {claim!r}")

        try:
            # 1) Build search query
            query = self.search_client.build_query(claim)
            logger.debug(f"[PIPELINE] using query: {query}")

            # 2) Fetch raw results and normalize to dicts
            raw = self.search_client.search(query)
            normalized = []
            for entry in raw:
                if isinstance(entry, str):
                    normalized.append({"title": "", "link": entry})
                else:
                    url = entry.get("url") or entry.get("link") or ""
                    normalized.append({
                        "title": entry.get("title", ""),
                        "link":  url
                    })

            resp = {"items": normalized}

            # 3) Gather evidence, passing along any overrides
            be_kwargs = {"llm": llm_client}
            if max_google is not None:
                be_kwargs["max_google"] = max_google
            if max_news is not None:
                be_kwargs["max_news"] = max_news

            evidence, sources, quotes = build_evidence(resp, claim, **be_kwargs)
            if not evidence:
                logger.debug("[PIPELINE] no evidence found")
                return {
                    "label":         "unknown",
                    "confidence":    0.0,
                    "explanation":   "No credible evidence found",
                    "evidence":      "",
                    "sources":       sources,
                    "source_quotes": []
                }

            # 4) Classify
            label, conf = self.clf.classify_with_evidence(
                claim, evidence, return_conf=True
            )
            logger.debug(f"[PIPELINE] label={label} confidence={conf:.2f}")

            # 5) Justify
            try:
                explanation = self.clf.justify(label, claim, evidence, llm=llm_client)
            except Exception:
                explanation = f"The claim is {label} based on the available evidence."

            # 6) Summarize evidence
            ev_summary = summarise(evidence, llm=llm_client)

            return {
                "label":         label,
                "confidence":    conf,
                "explanation":   explanation,
                "evidence":      ev_summary,
                "sources":       sources,
                "source_quotes": quotes
            }
        except Exception as e:
            logger.exception(f"[PIPELINE] _run_check failed: {e}")
            return {
                "label":         "unknown",
                "confidence":    0.0,
                "explanation":   f"Error during fact-checking: {e}",
                "evidence":      "",
                "sources":       [],
                "source_quotes": []
            }
