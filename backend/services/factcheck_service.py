# backend/services/factcheck_service.py

import time
from .service_manager import service_manager
from services.ocr import OCRService
from pipeline.orchestrator import build_evidence
from pipeline.summarization import summarise_input_text, summarise_evidence as summarise
from pipeline.extraction.extractor import extract_claims_llm
from services.classifier_intent.intent_parser import detect_intent
from core.logging import logger


class PipelineOrchestrator:
    def __init__(self):
        self.llm           = service_manager.get_llm_client()
        self.clf           = service_manager.get_classifier()
        self.search_client = service_manager.get_search_client()
        self.ocr           = OCRService()
        logger.debug("[PIPELINE] instance created")

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

    def check_text(self, text: str, max_claims: int = 5) -> dict:
        logger.debug("[PIPELINE] check_text() called")
        if hasattr(self.llm, "clear_cache"):
            self.llm.clear_cache()

        intent = detect_intent(text, self.llm)
        logger.debug(f"[PIPELINE] detected intent: {intent}")

        handler = self._handlers.get(intent, self._handle_unknown)
        return handler(text, max_claims)

    def _handle_non_verifiable(self, *_):
        logger.debug("[PIPELINE] non-verifiable input")
        return {
            "summary": "",
            "results": [],
            "validation_error": "Input is not verifiable.",
            "suggestion":       "Please enter a factual claim, question, or news paragraph."
        }

    def _handle_unknown(self, *_):
        return {
            "summary": "",
            "results": [],
            "validation_error": "Intent could not be determined.",
            "suggestion":       "Please enter a verifiable statement or question."
        }

    def _handle_fact_question(self, text: str, max_claims: int):
        # fallback to claim path (enhanced QA not implemented)
        logger.info("[PIPELINE] fallback to claim path")
        claim = text  # treat question as claim for now
        summary = summarise_input_text(text, self.llm)
        result = self._run_check(claim)
        return {"summary": summary, "results": [{"claim": claim, **result}]}

    def _handle_multi_claim(self, text: str, max_claims: int):
        logger.debug("[PIPELINE] handling multi_claim/news_paragraph")
        claims  = extract_claims_llm(text, max_claims, llm=self.llm)
        summary = summarise_input_text(text, self.llm)

        results = [
            {
                "claim": c,
                **self._run_check(
                    c,
                    max_google=2,  # override defaults for multi‐claim
                    max_news=1
                )
            }
            for c in claims
        ]

        return {"summary": summary, "results": results}

    def _handle_fact_claim(self, text: str, max_claims: int):
        logger.debug("[PIPELINE] handling fact_claim")
        summary = summarise_input_text(text, self.llm)
        result  = self._run_check(text)
        return {"summary": summary, "results": [{"claim": text, **result}]}

    def _run_check(
        self,
        claim:       str,
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
            be_kwargs = {"llm": self.llm}
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
                explanation = self.clf.justify(label, claim, evidence, llm=self.llm)
            except Exception:
                explanation = f"The claim is {label} based on the available evidence."

            # 6) Summarize evidence
            ev_summary = summarise(evidence, llm=self.llm)

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
