"""
Summarization module.

Generates executive summaries of verification results.
"""

import re
from typing import List, Dict

from utils.logging import get_logger
from services.llm import chat

logger = get_logger(__name__)


def summarize_input(text: str, llm: str = None, model_id: str = None, max_tokens: int = 500, evidence_results: List[Dict] = None) -> str:
    """
    Generate an executive summary of the input text and verification results.
    
    Args:
        text: Original input text
        llm: Optional LLM provider
        model_id: Optional model identifier
        max_tokens: Maximum tokens for summary (default 500 for reasoning models)
        evidence_results: Optional list of verification results with evidence and reasoning
        
    Returns:
        Executive summary string
    """
    if not text or not text.strip():
        return ""
    
    # Build comprehensive context for summary
    context_parts = [f"Original Text:\n{text}"]
    
    if evidence_results:
        context_parts.append("\nVerification Results:")
        for i, result in enumerate(evidence_results, 1):
            claim = result.get("claim", f"Claim {i}")
            verdict = result.get("verdict", "UNKNOWN")
            confidence = result.get("confidence", 0.0)
            reasoning = result.get("reasoning", "")
            
            context_parts.append(f"\nClaim {i}: {claim}")
            context_parts.append(f"Verdict: {verdict} (Confidence: {confidence:.2f})")
            context_parts.append(f"Reasoning: {reasoning}")
            
            # Include key evidence snippets
            sources = result.get("sources", [])
            if sources:
                context_parts.append("Key Sources:")
                for source in sources[:3]:
                    title = source.get("title", "")
                    url = source.get("url", "")
                    if title:
                        context_parts.append(f"- {title}")
    
    full_context = "\n".join(context_parts)
    
    system = """
You are an expert fact checker. Produce a direct executive summary of the verification results. The summary must be 50 to 60 words. Never exceed 60 words.
Strict rules:
1. Start with the final verdict immediately. Use a clear statement such as “The claim is false”, “The claim is unverified”, or “The claim is supported”.
2. Explain the reason in one or two short sentences using only the evidence provided. Do not invent facts.
3. Reference the evidence as a group, for example “multiple independent reports” or “the cited articles”.
4. Use active voice only. No passive constructions.
5. Do not repeat the claim text. Do not quote the original claim.
6. Do not hedge. No vague phrases.
7. If evidence is conflicting, state that the evidence is mixed and explain which side is stronger.

If information is missing or unclear, state the uncertainty directly instead of guessing.
"""
    
    try:
        # Retry logic for LLM calls
        response = None
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = chat(
                    system,
                    f"Content to summarize:\n{full_context}",
                    provider=llm,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=0.5,
                )
                
                # Validate response quality
                if response and len(response.strip()) >= 30:
                    break
                else:
                    logger.warning(f"[SUMMARY] Attempt {attempt + 1}/{max_retries}: Response too short ({len(response) if response else 0} chars)")
                    if attempt < max_retries - 1:
                        continue
                        
            except Exception as e:
                logger.error(f"[SUMMARY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    response = None
        
        # Handle empty or invalid responses gracefully
        if not response or len(response.strip()) < 20:
            logger.warning("[SUMMARY] LLM returned invalid response after retries, using fallback")
            # Return first sentence as fallback
            for end in ['. ', '! ', '? ']:
                idx = text.find(end)
                if idx > 0:
                    return text[:idx + 1].strip()
            return text[:100].strip() + ('...' if len(text) > 100 else '')
        
        # Clean up response
        if response.startswith("Summary:"):
            response = response[8:].strip()
        
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Fallback if response looks bad
        if len(response) < 10:
            logger.warning(f"[SUMMARY] Response too short ({len(response)} chars), using fallback")
            for end in ['. ', '! ', '? ']:
                idx = text.find(end)
                if idx > 0:
                    return text[:idx + 1].strip()
            return text[:100].strip() + ('...' if len(text) > 100 else '')
        
        return response
        
    except Exception as e:
        logger.error(f"[SUMMARY] Input summary failed: {e}")
        # Fallback: first sentence
        for end in ['. ', '! ', '? ']:
            idx = text.find(end)
            if idx > 0:
                return text[:idx + 1].strip()
        return text[:100].strip() + ('...' if len(text) > 100 else '')


__all__ = ["summarize_input"]
