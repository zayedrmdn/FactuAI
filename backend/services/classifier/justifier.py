# services/classifier/justify.py

from core.logging import logger

def justify(label: str, claim: str, evidence: str, llm) -> str:
    """
    Given a label, claim, and evidence, ask the LLM to produce a one‑sentence
    justification of why the claim is labeled as such.
    """
    logger.debug(f"[CLASSIFIER] justify called for label: {label}")
    if hasattr(llm, "clear_cache"):
        llm.clear_cache()

    try:
        prompt = (
            f"You are given a factual claim and supporting evidence.\n"
            f"Explain why the claim is labeled as {label.lower()} in a single, clear sentence.\n\n"
            f"Claim: {claim}\n"
            f"Evidence: {evidence}\n\n"
            "Explanation:"
        )
        explanation = llm.generate_response(prompt, max_tokens=100).strip()

        # remove any leading "Explanation:" and clean up
        explanation = explanation.replace("Explanation:", "").strip()

        # ensure it begins with "The claim is ..."
        if not explanation.lower().startswith("the claim is"):
            explanation = f"The claim is {label.lower()} because {explanation}"

        # ensure it ends with a period
        if not explanation.endswith("."):
            explanation += "."

        return explanation

    except Exception as e:
        logger.debug(f"[CLASSIFIER] Justify error: {e}")
        return f"The claim is {label.lower()} based on the available evidence."
    