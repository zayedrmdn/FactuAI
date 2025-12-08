"""
Fact-Checking Pipeline for FactuAI

Consolidated pipeline orchestration combining:
- Intent detection
- Claim extraction
- Evidence collection
- Summarization
- Verification

Simplified from orchestrator + factcheck_service + extractors into one module.
"""

import re
from typing import Dict, List, Any, Generator, Optional, Tuple

from utils.logging import get_logger
from utils.helpers import ValidationError, PipelineError
from factcheck import llm_client, evidence

logger = get_logger(__name__)

# Pipeline phase messages
PHASE_DETECTING_INTENT = "Detecting intent..."
PHASE_EXTRACTING_CLAIMS = "Extracting claims..."
PHASE_GENERATING_SUMMARY = "Generating summary..."
PHASE_VERIFYING_CLAIM = "Verifying claim..."
PHASE_COLLECTING_EVIDENCE = "Collecting evidence..."


# ==========================================================================
# Intent Detection
# ==========================================================================

def detect_intent(text: str, llm: str = None) -> str:
    """
    Detect the intent/type of input text.
    
    Categories:
    - fact_claim: Single factual claim
    - fact_question: Question about facts
    - news_paragraph: News article or paragraph with multiple claims
    - multi_claim: Multiple claims
    - opinion: Subjective opinion
    - nonsense: Invalid/nonsense input
    - instructional: How-to/instruction text
    
    Args:
        text: Input text to classify
        llm: Optional LLM provider (uses default if None)
        
    Returns:
        Intent category string
    """
    logger.debug(f"[INTENT] Detecting intent for: {text[:100]}...")
    
    # Quick heuristics
    text_stripped = text.strip()
    
    if not text_stripped or len(text_stripped) < 5:
        return "nonsense"
    
    if text_stripped.endswith("?"):
        return "fact_question"
    
    # Check word count for multi-claim detection
    word_count = len(text_stripped.split())
    if word_count > 100:
        return "news_paragraph"
    
    # Opinion markers
    opinion_markers = [
        "i think", "i believe", "in my opinion", "i feel", "should", "ought to"
    ]
    text_lower = text_stripped.lower()
    if any(marker in text_lower for marker in opinion_markers):
        return "opinion"
    
    # Instructional markers
    instructional_markers = ["how to", "step 1", "first,", "next,", "finally,"]
    if any(marker in text_lower for marker in instructional_markers):
        return "instructional"
    
    # Use LLM for complex cases
    try:
        system = """Classify the input text into ONE of these categories:
- fact_claim: A single factual claim that can be verified
- news_paragraph: A news article or paragraph with multiple claims
- multi_claim: Multiple distinct factual claims
- opinion: Subjective opinion or belief
- nonsense: Unclear, incoherent, or invalid input

Respond with ONLY the category name."""
        
        response = llm_client.chat(system, f"Text: {text_stripped}", provider=llm, max_tokens=16)
        response_clean = response.strip().lower()
        
        valid_intents = ["fact_claim", "fact_question", "news_paragraph", "multi_claim", "opinion", "nonsense", "instructional"]
        for intent in valid_intents:
            if intent in response_clean:
                logger.debug(f"[INTENT] Detected: {intent}")
                return intent
                
    except Exception as e:
        logger.warning(f"[INTENT] LLM classification failed: {e}")
    
    # Default fallback
    return "fact_claim"


# ==========================================================================
# Claim Extraction
# ==========================================================================

def extract_claims(text: str, max_claims: int = 5, llm: str = None) -> List[str]:
    """
    Extract factual claims from text.
    
    Args:
        text: Input text
        max_claims: Maximum number of claims to extract
        llm: Optional LLM provider
        
    Returns:
        List of extracted claim strings
    """
    logger.debug(f"[EXTRACTION] Extracting up to {max_claims} claims from {len(text)} chars")
    
    # Adjust max_claims based on text length
    word_count = len(text.split())
    dynamic_max = min(max_claims, max(1, word_count // 50))
    
    system = f"""Extract {dynamic_max} factual claims from the following text.
List them one per line.
Only include verifiable factual statements.
Do not include opinions or subjective statements."""
    
    try:
        response = llm_client.chat(system, text, provider=llm, max_tokens=512)
        
        # Parse response into claims
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Remove numbering (1., 2., etc.)
            if line[0].isdigit() and '. ' in line:
                line = line.split('. ', 1)[1]
            
            # Remove bullet points
            if line.startswith('- '):
                line = line[2:]
            
            claims.append(line)
            if len(claims) >= dynamic_max:
                break
        
        logger.debug(f"[EXTRACTION] Extracted {len(claims)} claims")
        return claims
        
    except Exception as e:
        logger.error(f"[EXTRACTION] Claim extraction failed: {e}")
        return []


# ==========================================================================
# Summarization
# ==========================================================================

def summarize_input(text: str, llm: str = None, max_tokens: int = 120) -> str:
    """
    Summarize input text.
    
    Args:
        text: Text to summarize
        llm: Optional LLM provider
        max_tokens: Maximum tokens for summary
        
    Returns:
        Summary string
    """
    if not text or not text.strip():
        return ""
    
    system = "Summarize the following text into a concise neutral overview. Do NOT classify; just summarize factual content."
    
    try:
        response = llm_client.chat(system, f"Text:\n{text}", provider=llm, max_tokens=max_tokens)
        
        # Clean up response
        if response.startswith("Summary:"):
            response = response[8:].strip()
        
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Fallback if response looks bad
        if len(response) < 10:
            # Return first sentence
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


def summarize_evidence(evidence_text: str, llm: str = None) -> str:
    """
    Summarize collected evidence.
    
    Args:
        evidence_text: Evidence text to summarize
        llm: Optional LLM provider
        
    Returns:
        Summary string
    """
    if not evidence_text or not evidence_text.strip():
        return ""
    
    words = evidence_text.split()
    target_words = max(20, min(50, len(words) // 4))
    
    system = f"""Summarize the following evidence in about {target_words} words.
Keep it factual, neutral, and concise. Avoid repetition."""
    
    try:
        response = llm_client.chat(system, f"Evidence:\n{evidence_text}", provider=llm, max_tokens=target_words + 30)
        
        # Clean up
        if response.startswith("Summary:"):
            response = response[8:].strip()
        
        response = re.sub(r'\s+', ' ', response).strip()
        
        # Truncate to target words
        response_words = response.split()
        if len(response_words) > target_words:
            response = ' '.join(response_words[:target_words]) + '...'
        
        return response
        
    except Exception as e:
        logger.error(f"[SUMMARY] Evidence summary failed: {e}")
        # Fallback: first N words
        return ' '.join(words[:target_words]) + '...'


# ==========================================================================
# Verification
# ==========================================================================

def verify_claim(
    claim: str,
    llm: str = None,
    num_google: int = 5,
    num_news: int = 5,
    top_k: int = 10
) -> Dict[str, Any]:
    """
    Verify a single claim by collecting evidence and generating verdict.
    
    Args:
        claim: The claim to verify
        llm: Optional LLM provider
        num_google: Number of Google results
        num_news: Number of NewsAPI results
        top_k: Number of top evidence items
        
    Returns:
        Dict with 'verdict', 'confidence', 'reasoning', 'evidence', 'sources'
    """
    logger.info(f"[VERIFY] Checking claim: {claim[:100]}...")
    
    try:
        # Collect evidence
        evidence_items = evidence.collect_evidence(
            claim,
            num_google=num_google,
            num_news=num_news,
            top_k=top_k
        )
        
        if not evidence_items:
            logger.warning("[VERIFY] No evidence found")
            return {
                "verdict": "UNVERIFIABLE",
                "confidence": 0.0,
                "reasoning": "Could not find sufficient evidence to verify this claim.",
                "evidence": [],
                "sources": []
            }
        
        # Build evidence text
        evidence_text = '\n'.join([f"- {item['text']}" for item in evidence_items])
        
        # Generate verdict using LLM
        system = """You are a fact-checking AI. Analyze the claim and evidence, then provide:
1. VERDICT: One of [TRUE, FALSE, MOSTLY_TRUE, MOSTLY_FALSE, MIXED, UNVERIFIABLE]
2. CONFIDENCE: 0.0 to 1.0
3. REASONING: Brief explanation (2-3 sentences)

Format:
VERDICT: <verdict>
CONFIDENCE: <score>
REASONING: <explanation>"""
        
        user_msg = f"""Claim: {claim}

Evidence:
{evidence_text}"""
        
        response = llm_client.chat(system, user_msg, provider=llm, max_tokens=300)
        
        # Parse response
        verdict = "UNVERIFIABLE"
        confidence = 0.5
        reasoning = "Unable to determine verdict."
        
        for line in response.split('\n'):
            if line.startswith('VERDICT:'):
                verdict = line.split(':', 1)[1].strip().upper()
            elif line.startswith('CONFIDENCE:'):
                try:
                    confidence = float(line.split(':', 1)[1].strip())
                except ValueError:
                    pass
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
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "evidence": evidence_items,
            "sources": unique_sources[:5]  # Limit to top 5 sources
        }
        
    except Exception as e:
        logger.error(f"[VERIFY] Verification failed: {e}")
        raise PipelineError(f"Verification failed: {e}")


# ==========================================================================
# High-Level Pipeline
# ==========================================================================

def check_text(
    text: str,
    max_claims: int = 5,
    llm: str = None
) -> Dict[str, Any]:
    """
    Complete fact-checking pipeline for text input.
    
    Args:
        text: Input text to fact-check
        max_claims: Maximum claims to extract for multi-claim inputs
        llm: Optional LLM provider
        
    Returns:
        Dict with 'summary', 'results', optional 'validation_error'
    """
    logger.info(f"[PIPELINE] Starting fact-check for {len(text)} chars")
    
    try:
        # Detect intent
        intent = detect_intent(text, llm)
        logger.info(f"[PIPELINE] Detected intent: {intent}")
        
        # Handle non-verifiable inputs
        if intent in ["opinion", "nonsense", "instructional"]:
            return {
                "summary": "",
                "results": [],
                "validation_error": "Input is not verifiable.",
                "suggestion": "Please enter a factual claim, question, or news paragraph."
            }
        
        # Generate summary
        summary = summarize_input(text, llm)
        
        # Handle different intent types
        if intent == "fact_question":
            # Treat question as a claim
            result = verify_claim(text, llm)
            return {
                "summary": summary,
                "results": [{"claim": text, **result}]
            }
        
        elif intent in ["news_paragraph", "multi_claim"]:
            # Extract and verify multiple claims
            claims = extract_claims(text, max_claims, llm)
            
            if not claims:
                return {
                    "summary": summary,
                    "results": [],
                    "validation_error": "No factual claims found.",
                    "suggestion": "Try a different text."
                }
            
            results = []
            for claim in claims:
                result = verify_claim(claim, llm, num_google=3, num_news=2, top_k=5)
                results.append({"claim": claim, **result})
            
            return {
                "summary": summary,
                "results": results
            }
        
        else:  # fact_claim
            result = verify_claim(text, llm)
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
    llm: str = None
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
    """
    try:
        # Intent detection
        yield {"type": "phase", "message": PHASE_DETECTING_INTENT, "progress": 5}
        intent = detect_intent(text, llm)
        logger.info(f"[PIPELINE] Detected intent: {intent}")
        
        if intent in ["opinion", "nonsense", "instructional"]:
            yield {
                "type": "error",
                "message": "Input is not verifiable.",
                "suggestion": "Please enter a factual claim, question, or news paragraph."
            }
            return
        
        # Summary generation
        yield {"type": "phase", "message": PHASE_GENERATING_SUMMARY, "progress": 10}
        summary = summarize_input(text, llm)
        yield {"type": "summary", "summary": summary}
        
        # Handle different intents
        if intent == "fact_question":
            yield {"type": "phase", "message": PHASE_VERIFYING_CLAIM, "progress": 30}
            result = verify_claim(text, llm)
            yield {"type": "result", "result": {"claim": text, **result}}
        
        elif intent in ["news_paragraph", "multi_claim"]:
            yield {"type": "phase", "message": PHASE_EXTRACTING_CLAIMS, "progress": 15}
            claims = extract_claims(text, max_claims, llm)
            
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
                    "progress": progress
                }
                
                result = verify_claim(claim, llm, num_google=3, num_news=2, top_k=5)
                yield {"type": "result", "result": {"claim": claim, **result}}
        
        else:  # fact_claim
            yield {"type": "phase", "message": PHASE_VERIFYING_CLAIM, "progress": 30}
            result = verify_claim(text, llm)
            yield {"type": "result", "result": {"claim": text, **result}}
            
    except Exception as e:
        logger.error(f"[PIPELINE] Stream error: {e}", exc_info=True)
        yield {
            "type": "error",
            "message": f"Internal error: {str(e)}"
        }


__all__ = [
    "detect_intent",
    "extract_claims",
    "summarize_input",
    "summarize_evidence",
    "verify_claim",
    "check_text",
    "check_text_stream",
]
