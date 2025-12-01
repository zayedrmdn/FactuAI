# services/classifier_intent/client.py

from typing import Optional

from services.llm.llm_client import QwenClient
from core.logging import logger

from .heuristics import _fast_heuristics
from .parser import _parse_llm_label
from .prompt import build_detect_intent_system, build_detect_intent_prompt


def detect_intent(text: str, llm: Optional[QwenClient] = None) -> str:
    if llm is None:
        llm = QwenClient()

    logger.debug(f"[INTENT] Detecting intent for: {repr(text)}")

    if hasattr(llm, "clear_cache"):
        llm.clear_cache()

    text_stripped = text.strip()

    if not text_stripped:
        logger.debug("[INTENT] Empty input -> nonsense")
        return "nonsense"
    if len(text_stripped) < 5:
        logger.debug("[INTENT] Too short (<5 chars) -> nonsense")
        return "nonsense"
    if text_stripped.endswith("?"):
        logger.debug("[INTENT] Ends with '?' -> fact_question")
        return "fact_question"

    heuristic_label = _fast_heuristics(text_stripped)
    if heuristic_label:
        logger.debug(f"[INTENT] Heuristic early decision: {heuristic_label}")
        return heuristic_label

    system = build_detect_intent_system()
    prompt = build_detect_intent_prompt(text_stripped)
    try:
        logger.debug("[INTENT] Calling LLM for intent classification")
        raw = llm.generate_response(f"{system}\n\n{prompt}", max_tokens=16)
        logger.debug(f"[INTENT] Raw LLM response: {repr(raw)}")
        label = _parse_llm_label(raw)
        if label:
            logger.debug(f"[INTENT] LLM returned valid label: {label}")
            return label
        logger.debug("[INTENT] LLM yielded no valid label, falling back")
    except Exception as e:
        logger.debug(f"[INTENT] LLM error: {e}, falling back")

    logger.debug("[INTENT] Default fallback -> fact_claim")
    return "fact_claim"
