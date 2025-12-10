"""
Claim verification module.

Verifies claims by collecting evidence and generating verdicts.
"""

from typing import Dict, Any, Optional, List

from utils.logging import get_logger
from utils.helpers import PipelineError
from search.base import collect_evidence
from services.llm import chat
from config import (
    EVIDENCE_DEFAULT_COUNT,
    MAX_EVIDENCE_CHARS,
    TOKEN_ESTIMATE_RATIO,
    LLM_MAX_TOKENS_BASE,
    LLM_MAX_TOKENS_BUFFER,
    LLM_MAX_TOKENS_MAX,
    LLM_MAX_TOKENS_REASONING_BASE
)

logger = get_logger(__name__)


def verify_claim(
    claim: str,
    google_query: str,
    newsapi_query: str,
    llm: str = None,
    model_id: str = None,
    num_google: int = 5,
    num_news: int = 5,
    num_tavily: int = 5,
    top_k: int = EVIDENCE_DEFAULT_COUNT,
    enabled_search_providers: Optional[List[str]] = None,
    verification_question: Optional[str] = None
) -> Dict[str, Any]:
    """
    Verify a single claim by collecting evidence and generating verdict.
    
    Args:
        claim: The specific claim to verify
        google_query: Optimized search query for Google (from intent detection)
        newsapi_query: Optimized search query for NewsAPI (from intent detection)
        llm: Optional LLM provider
        model_id: Optional model identifier
        num_google: Number of Google results
        num_news: Number of NewsAPI results
        num_tavily: Number of Tavily results
        top_k: Number of top evidence items
        enabled_search_providers: List of enabled providers ['google', 'newsapi', 'tavily']
        verification_question: Optional natural language question for Tavily answer-seeking
        
    Returns:
        Dict with 'verdict', 'confidence', 'reasoning', 'evidence', 'sources'
    """
    logger.info(f"[VERIFY] Checking claim: {claim[:100]}...")
    
    try:
        # Collect evidence using the provider-specific queries
        evidence_items = collect_evidence(
            claim,
            google_query=google_query,
            newsapi_query=newsapi_query,
            num_google=num_google,
            num_news=num_news,
            num_tavily=num_tavily,
            top_k=top_k,
            enabled_providers=enabled_search_providers,
            verification_question=verification_question
        )
        
        logger.info(f"[VERIFY] Collected {len(evidence_items)} evidence items (top_k={top_k})")
        
        if not evidence_items:
            logger.warning("[VERIFY] No evidence found")
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "reasoning": "Could not find sufficient evidence to verify this claim.",
                "evidence": [],
                "sources": []
            }
        
        # Build evidence text with strict length limit
        evidence_text = '\n'.join([f"- {item['text']}" for item in evidence_items])
        
        logger.info(f"[VERIFY] Evidence text: {len(evidence_text)} chars, ~{len(evidence_text)//TOKEN_ESTIMATE_RATIO} tokens")
        
        if len(evidence_text) > MAX_EVIDENCE_CHARS:
            logger.warning(f"[VERIFY] Truncating evidence text from {len(evidence_text)} to {MAX_EVIDENCE_CHARS} chars")
            evidence_text = evidence_text[:MAX_EVIDENCE_CHARS] + "... (truncated)"
        
        # Generate verdict using LLM
        system = """You are a fact-checking AI. Analyze the claim and evidence, then provide:
1. VERDICT: One of [TRUE, FALSE, MOSTLY_TRUE, MOSTLY_FALSE, MIXED, UNVERIFIABLE]
2. CONFIDENCE: 0.0 to 1.0
3. REASONING: Brief explanation (2-3 sentences)

IMPORTANT:
- If the claim is a known conspiracy theory, myth, or explicitly debunked by the evidence, the verdict MUST be FALSE.
- Do NOT use "UNKNOWN" or "DEBUNKED" as verdict. Use FALSE.
- If evidence is insufficient, use UNVERIFIABLE.

Format:
VERDICT: <verdict>
CONFIDENCE: <score>
REASONING: <explanation>"""
        
        user_msg = f"""Claim: {claim}

Evidence:
{evidence_text}"""
        
        # Dynamic token limit based on evidence size
        evidence_token_estimate = len(evidence_text) // TOKEN_ESTIMATE_RATIO
        
        # Increase base tokens for reasoning models which tend to generate longer responses
        base_tokens = LLM_MAX_TOKENS_BASE
        if model_id and 'reasoning' in model_id.lower():
            base_tokens = max(base_tokens, LLM_MAX_TOKENS_REASONING_BASE)  # Use configured reasoning base
        
        dynamic_max_tokens = min(LLM_MAX_TOKENS_MAX, max(base_tokens, int(evidence_token_estimate + LLM_MAX_TOKENS_BUFFER)))
        
        logger.info(f"[VERIFY] Dynamic max_tokens: {dynamic_max_tokens} (evidence: ~{evidence_token_estimate} tokens, model: {model_id}, base: {base_tokens})")
        
        # Retry logic for LLM calls with error handling
        response = None
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = chat(
                    system,
                    user_msg,
                    provider=llm,
                    model_id=model_id,
                    max_tokens=dynamic_max_tokens,
                    temperature=0.3,
                )
                
                # Validate response quality
                if response and len(response.strip()) >= 50:
                    break
                else:
                    logger.warning(f"[VERIFY] Attempt {attempt + 1}/{max_retries}: Response too short ({len(response) if response else 0} chars)")
                    if attempt < max_retries - 1:
                        dynamic_max_tokens = min(2500, dynamic_max_tokens + 500)
                        continue
                    
            except Exception as e:
                error_msg = str(e).lower()
                logger.error(f"[VERIFY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                
                # Log error types to help users understand issues
                if "rate" in error_msg or "429" in error_msg:
                    logger.warning(f"[VERIFY] Model {model_id} is currently rate-limited by the provider")
                elif "developer instruction" in error_msg or "system message" in error_msg:
                    logger.warning(f"[VERIFY] Model {model_id} compatibility issue detected (retrying with adjusted format)")
                
                if attempt < max_retries - 1:
                    continue
                else:
                    response = None
        
        # Handle empty or invalid responses gracefully
        if not response or len(response.strip()) < 20:
            logger.warning("[VERIFY] LLM returned invalid response after retries, returning unverifiable")
            return {
                "verdict": "UNVERIFIABLE",
                "label": "unverifiable",
                "confidence": 0.3,
                "reasoning": "Unable to analyze evidence - model did not provide a valid response. Please try a different model.",
                "evidence": evidence_items,
                "sources": [],
                "source_quotes": []
            }
        
        # Parse response
        verdict = "UNVERIFIABLE"
        confidence = 0.5
        reasoning = "Unable to determine verdict."
        
        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('VERDICT:'):
                verdict = line.split(':', 1)[1].strip().upper()
                # Map common hallucinations to valid verdicts
                if verdict in ['UNKNOWN', 'DEBUNKED', 'MYTH']:
                    verdict = 'FALSE' if verdict != 'UNKNOWN' else 'UNVERIFIABLE'
            elif line.startswith('CONFIDENCE:'):
                try:
                    conf_str = line.split(':', 1)[1].strip()
                    confidence = float(conf_str)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, IndexError) as e:
                    logger.warning(f"[VERIFY] Failed to parse confidence: {e}")
            elif line.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()
        
        # Extract sources
        sources = [
            {"url": item["url"], "title": item["title"], "source": item["source"]}
            for item in evidence_items
        ]
        
        # Deduplicate sources by URL
        seen_urls = set()
        unique_sources = []
        for source in sources:
            if source["url"] not in seen_urls:
                seen_urls.add(source["url"])
                unique_sources.append(source)
        
        # Format evidence items as source_quotes for frontend
        source_quotes = [
            {
                "quote": item["text"],
                "source": item["source"],
                "url": item["url"]
            }
            for item in evidence_items
        ]
        
        return {
            "verdict": verdict,
            "label": verdict.lower(),
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence_items,
            "sources": unique_sources[:5],
            "source_quotes": source_quotes
        }
        
    except Exception as e:
        logger.error(f"[VERIFY] Verification failed: {e}")
        raise PipelineError(f"Verification failed: {e}")


__all__ = ["verify_claim"]
