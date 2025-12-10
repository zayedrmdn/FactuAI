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
import json
from typing import Dict, List, Any, Generator, Optional, Tuple

from utils.logging import get_logger
from utils.helpers import ValidationError, PipelineError
from factcheck import llm_client, evidence

logger = get_logger(__name__)


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

# Pipeline phase messages
PHASE_DETECTING_INTENT = "Detecting intent..."
PHASE_EXTRACTING_CLAIMS = "Extracting claims..."
PHASE_GENERATING_SUMMARY = "Generating summary..."
PHASE_VERIFYING_CLAIM = "Verifying claim..."
PHASE_COLLECTING_EVIDENCE = "Collecting evidence..."


def extract_fallback_from_text(text: str) -> Optional[Dict[str, str]]:
    """
    Extract intent and queries from plain text response when JSON parsing fails.
    
    Args:
        text: Raw LLM response text
        
    Returns:
        Dict with intent, google_query, newsapi_query or None if extraction fails
    """
    try:
        text_lower = text.lower()
        
        # Extract intent
        intent = "fact_claim"  # default
        for intent_type in ["fact_claim", "fact_question", "news_paragraph", "multi_claim", "opinion", "nonsense", "instructional"]:
            if intent_type in text_lower:
                intent = intent_type
                break
        
        # Extract queries - look for patterns like "google_query:" or "newsapi_query:"
        google_query = ""
        newsapi_query = ""
        
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith('google_query:'):
                google_query = line.split(':', 1)[1].strip()
            elif line.lower().startswith('newsapi_query:'):
                newsapi_query = line.split(':', 1)[1].strip()
        
        # If no explicit queries found, try to extract from the text
        quotes = []
        if not google_query:
            # Look for quoted strings or comma-separated terms
            quotes = re.findall(r'"([^"]*)"', text)
            if quotes:
                google_query = quotes[0]
        
        if not newsapi_query and quotes and len(quotes) > 1:
            newsapi_query = quotes[1]
        
        # For reasoning models, try to find JSON at the end of the text
        if not google_query or not newsapi_query:
            # Look for JSON-like structure at the end
            json_match = re.search(r'\{.*\}', text, re.DOTALL)
            if json_match:
                try:
                    parsed = json.loads(json_match.group(0))
                    if 'google_query' in parsed:
                        google_query = parsed.get('google_query', google_query)
                    if 'newsapi_query' in parsed:
                        newsapi_query = parsed.get('newsapi_query', newsapi_query)
                    if 'intent' in parsed:
                        intent = parsed.get('intent', intent)
                except json.JSONDecodeError:
                    pass
        
        # Fallback: use first line after intent as query
        if not google_query:
            for line in lines:
                line = line.strip()
                if line and line not in [intent, "intent:", "query:", "queries:"]:
                    google_query = line
                    break
        
        return {
            "intent": intent,
            "google_query": google_query or text[:100],
            "newsapi_query": newsapi_query or google_query or text[:100],
            "verification_question": ""
        }
    except Exception as e:
        logger.warning(f"[PIPELINE] Fallback extraction failed: {e}")
        return None

def detect_intent(text: str, llm: str = None, model_id: str = None) -> Dict[str, str]:
    """
    Detect intent and generate optimized search queries in a single LLM call.
    
    Intent categories:
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
        model_id: Optional model identifier
        
    Returns:
        Dict with keys: 'intent', 'google_query', 'newsapi_query'
    """
    logger.debug(f"[INTENT] Detecting intent and building queries for: {text[:100]}...")
    
    # Quick heuristics for simple cases
    text_stripped = text.strip()
    
    if not text_stripped or len(text_stripped) < 5:
        fallback_query = text_stripped[:200] if text_stripped else ""
        return {
            "intent": "nonsense",
            "google_query": fallback_query,
            "newsapi_query": fallback_query,
            "verification_question": ""
        }
    
    text_lower = text_stripped.lower()
    
    # Quick opinion/instructional detection - skip LLM call
    opinion_markers = ["i think", "i believe", "in my opinion", "i feel", "should", "ought to"]
    if any(marker in text_lower for marker in opinion_markers):
        return {
            "intent": "opinion",
            "google_query": text_stripped[:200],
            "newsapi_query": text_stripped[:200],
            "verification_question": ""
        }
    
    instructional_markers = ["how to", "step 1", "first,", "next,", "finally,"]
    if any(marker in text_lower for marker in instructional_markers):
        return {
            "intent": "instructional",
            "google_query": text_stripped[:200],
            "newsapi_query": text_stripped[:200],
            "verification_question": ""
        }
    
    # Use LLM for intent + search queries extraction
    
    try:
        system = """You must output valid JSON only. No explanations, no markdown, no extra text.

IMPORTANT: For reasoning models, show your analysis in the reasoning field, then output ONLY the JSON at the very end.

Output format:
{
  "intent": "fact_claim",
  "google_query": "covid vaccine microchip tracking conspiracy debunk",
  "newsapi_query": "covid vaccine microchip",
  "verification_question": "Do COVID-19 vaccines contain microchips for tracking?"
}

Rules:
1. intent must be one of: fact_claim, fact_question, news_paragraph, multi_claim, opinion, nonsense, instructional

2. google_query: 4-8 optimized search terms for Google (multiword phrases allowed, lowercase, no punctuation)

3. newsapi_query: 3-6 shorter keywords for NewsAPI (minimal multiword, lowercase, no punctuation)

4. verification_question: ONLY for fact_claim or fact_question intents, generate a natural language question that directly verifies the claim. 
   Examples:
   - Claim: "Trump is president of Indonesia" → Question: "Who is the current president of Indonesia?"
   - Claim: "The moon landing was fake" → Question: "Did the Apollo moon landing actually happen?"
   - Question: "What causes climate change?" → Question: "What causes climate change?" (keep as is)
   For other intents (opinion, instructional, etc.), set verification_question to empty string.

5. Extract only essential concepts and entities from the input text.

Output JSON only at the end."""
        
        response = llm_client.chat(
            system,
            f"Text: {text_stripped}",
            provider=llm,
            model_id=model_id,
            max_tokens=500,  # Increased for reasoning models that need more tokens
            temperature=0.3,
        )
        
        # Parse JSON response with robust error handling for reasoning models
        response_clean = response.strip()
        
        # Handle various LLM response formats, especially reasoning models
        # 1. First, try to extract JSON from the end of reasoning text
        # Reasoning models often put the final answer at the end
        if not response_clean.startswith('{'):
            # Look for JSON at the end of the response
            last_brace = response_clean.rfind('{')
            if last_brace != -1:
                # Try to find the matching closing brace
                brace_count = 0
                end_pos = last_brace
                for i in range(last_brace, len(response_clean)):
                    if response_clean[i] == '{':
                        brace_count += 1
                    elif response_clean[i] == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end_pos = i + 1
                            break
                
                if end_pos > last_brace:
                    potential_json = response_clean[last_brace:end_pos]
                    try:
                        result = json.loads(potential_json)
                        logger.debug(f"[INTENT] Extracted JSON from end of reasoning text")
                        response_clean = potential_json  # Use the extracted JSON
                    except json.JSONDecodeError:
                        pass  # Continue with other parsing attempts
        
        # 2. Handle markdown code blocks
        if response_clean.startswith('```'):
            # Extract content between first and last ```
            parts = response_clean.split('```')
            if len(parts) >= 3:
                # Find JSON block (might be labeled as json)
                for part in parts[1:-1]:  # Skip first and last empty parts
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{') and part.endswith('}'):
                        response_clean = part
                        break
                else:
                    # If no labeled json block, take the first code block
                    response_clean = parts[1].strip()
                    if response_clean.startswith('json'):
                        response_clean = response_clean[4:].strip()
            else:
                # Malformed markdown, try to extract JSON anyway
                response_clean = response_clean.replace('```', '').strip()
        
        # 3. Try to find JSON object anywhere in the response
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            potential_json = response_clean[json_start:json_end]
            try:
                result = json.loads(potential_json)
                response_clean = potential_json  # Use the extracted JSON
            except json.JSONDecodeError:
                # If extraction fails, try the whole response
                pass
        
        # 4. Final JSON parsing
        try:
            result = json.loads(response_clean)
        except json.JSONDecodeError as e:
            logger.warning(f"[INTENT] Failed to parse JSON response: {e}")
            logger.debug(f"[INTENT] Raw response (first 500 chars): {response[:500]}...")
            logger.debug(f"[INTENT] Cleaned response (first 500 chars): {response_clean[:500]}...")
            # Try to extract key information from raw response as fallback
            fallback_result = extract_fallback_from_text(response)
            if fallback_result:
                logger.info(f"[INTENT] Using fallback extraction: {fallback_result}")
                result = fallback_result
            else:
                raise  # Re-raise to trigger outer fallback
        
        # Validate intent
        valid_intents = ["fact_claim", "fact_question", "news_paragraph", "multi_claim", "opinion", "nonsense", "instructional"]
        intent = result.get("intent", "fact_claim").lower()
        if intent not in valid_intents:
            intent = "fact_claim"
        
        google_query = result.get("google_query", "").strip()
        newsapi_query = result.get("newsapi_query", "").strip()
        verification_question = result.get("verification_question", "").strip()
        
        # Fallback to original text if queries are too short
        fallback_query = text_stripped[:200]
        if len(google_query.split()) < 2:
            google_query = fallback_query
        if len(newsapi_query.split()) < 2:
            newsapi_query = fallback_query
        
        # Generate verification question if missing for fact-checkable intents
        if not verification_question and intent in ["fact_claim", "fact_question"]:
            if intent == "fact_question":
                verification_question = text_stripped  # Use original question
            else:
                # Generate simple verification question from claim
                verification_question = f"Is this true: {text_stripped[:150]}?"
        
        logger.debug(f"[INTENT] Detected: {intent}, Google: {google_query[:50]}..., NewsAPI: {newsapi_query[:50]}..., VQ: {verification_question[:50] if verification_question else 'N/A'}...")
        return {
            "intent": intent,
            "google_query": google_query,
            "newsapi_query": newsapi_query,
            "verification_question": verification_question
        }
                
    except json.JSONDecodeError as e:
        logger.warning(f"[INTENT] Failed to parse JSON response: {e}")
        fallback_query = text_stripped[:200]
        return {
            "intent": "fact_claim",
            "google_query": fallback_query,
            "newsapi_query": fallback_query,
            "verification_question": f"Is this true: {text_stripped[:150]}?"
        }
    except Exception as e:
        logger.warning(f"[INTENT] LLM call failed: {e}")
        fallback_query = text_stripped[:200]
        return {
            "intent": "fact_claim",
            "google_query": fallback_query,
            "newsapi_query": fallback_query,
            "verification_question": f"Is this true: {text_stripped[:150]}?"
        }


# ==========================================================================
# Claim Extraction
# ==========================================================================

def extract_claims(text: str, max_claims: int = 5, llm: str = None, model_id: str = None) -> List[str]:
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
List them one per line, starting each with a dash (-).

IMPORTANT RULES:
- Extract EXACTLY what the text claims, even if false or conspiracy theories
- Do NOT add commentary like "this is false" or "debunked"
- Do NOT say "there are no claims" - extract what's stated
- Keep the core substance of each claim

Example:
Input: "COVID vaccines contain microchips."
Output: - COVID-19 vaccines contain microchips

Now extract from:"""
    
    try:
        # Dynamic: base on number of claims and text length
        dynamic_max_tokens = min(1024, max(256, dynamic_max * 80 + 400))
        
        response = llm_client.chat(
            system,
            text,
            provider=llm,
            model_id=model_id,
            max_tokens=dynamic_max_tokens,
        )
        
        # Parse response into claims
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Clean up formatting markers using regex
            # Matches:
            # ^\s*[-*]\s+  -> "- " or "* "
            # ^\s*\d+\.\s+ -> "1. "
            # ^\s*-\s*     -> "-" (without space)
            line = re.sub(r'^(\s*[-*]\s*|\s*\d+\.\s+)', '', line).strip()
            
            # Basic validation
            if len(line) < 10:
                continue
                
            # Skip obvious non-claims
            line_lower = line.lower()
            if line_lower.startswith(('source:', 'note:', 'disclaimer:')):
                continue
                
            # Skip "No claims found" type responses
            if "no factual claims" in line_lower or "no claims found" in line_lower:
                continue

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

def summarize_input(text: str, llm: str = None, model_id: str = None, max_tokens: int = 500, evidence_results: List[Dict] = None) -> str:
    """
    Generate an executive summary of the input text and verification results.
    
    Args:
        text: Original input text
        llm: Optional LLM provider
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
                for source in sources[:3]:  # Limit to top 3 sources
                    title = source.get("title", "")
                    url = source.get("url", "")
                    if title:
                        context_parts.append(f"- {title}")
    
    full_context = "\n".join(context_parts)
    
    system = """You are an expert fact-checker. Write a concise, decisive executive summary (maximum 50-60 words).
    
    Follow this structure strictly:
    1.  **The Verdict First:** Start immediately with the final conclusion (e.g., "This claim is False," "This assertion is unsubstantiated").
    2.  **The Evidence:** Briefly explain *why* based on the provided verification results.
    3.  **The Consensus:** meaningful reference to the sources (e.g., "refuted by multiple health organizations").

    Avoid passive voice. Be direct. Do not simply repeat the original claim; explain why it is true or false."""
    
    try:
        # Retry logic for LLM calls
        response = None
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = llm_client.chat(
                    system,
                    f"Content to summarize:\n{full_context}",
                    provider=llm,
                    model_id=model_id,
                    max_tokens=max_tokens,
                    temperature=0.5,  # Moderate temperature for summaries
                )
                
                # Validate response quality
                if response and len(response.strip()) >= 30:
                    break  # Good response received
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


# ==========================================================================
# Verification
# ==========================================================================

def verify_claim(
    claim: str,
    google_query: str,
    newsapi_query: str,
    llm: str = None,
    model_id: str = None,
    num_google: int = 5,
    num_news: int = 5,
    num_tavily: int = 5,
    top_k: int = 10,
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
        evidence_items = evidence.collect_evidence(
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
        # Truncate evidence text to avoid context window overflow (approx 100k chars ~ 25k tokens)
        MAX_EVIDENCE_CHARS = 100000
        evidence_text = '\n'.join([f"- {item['text']}" for item in evidence_items])
        
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
        evidence_token_estimate = len(evidence_text) // 3
        dynamic_max_tokens = min(2000, max(800, evidence_token_estimate + 500))
        
        # Retry logic for LLM calls with error handling
        response = None
        max_retries = 2
        for attempt in range(max_retries):
            try:
                response = llm_client.chat(
                    system,
                    user_msg,
                    provider=llm,
                    model_id=model_id,
                    max_tokens=dynamic_max_tokens,
                    temperature=0.3,  # Lower temperature for more consistent factual analysis
                )
                
                # Validate response quality
                if response and len(response.strip()) >= 50:
                    break  # Good response received
                else:
                    logger.warning(f"[VERIFY] Attempt {attempt + 1}/{max_retries}: Response too short ({len(response) if response else 0} chars)")
                    if attempt < max_retries - 1:
                        # Retry with adjusted parameters
                        dynamic_max_tokens = min(2500, dynamic_max_tokens + 500)
                        continue
                    
            except Exception as e:
                logger.error(f"[VERIFY] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt < max_retries - 1:
                    continue
                else:
                    # Final attempt failed, return unverifiable
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
                    # Clamp to valid range
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
            "sources": unique_sources[:5],  # Limit to top 5 sources
            "source_quotes": source_quotes  # Add formatted quotes for frontend
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
        
        elif intent in ["news_paragraph", "multi_claim"]:
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
                result = verify_claim(
                    claim,
                    google_query,
                    newsapi_query,
                    llm=reasoning_cfg.get("provider", llm),
                    model_id=reasoning_cfg.get("model_id"),
                    num_google=num_google,
                    num_news=num_news,
                    top_k=10,
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
        
        elif intent in ["news_paragraph", "multi_claim"]:
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
                result = verify_claim(
                    claim,
                    google_query,
                    newsapi_query,
                    llm=reasoning_cfg.get("provider", llm),
                    model_id=reasoning_cfg.get("model_id"),
                    num_google=num_google,
                    num_news=num_news,
                    num_tavily=num_tavily,
                    top_k=10,
                    enabled_search_providers=enabled_search_providers,
                    verification_question=verification_question,
                )
                all_results.append(result)
                yield {"type": "result", "result": {"claim": claim, **result}, "claim_index": i + 1}
        
        else:  # fact_claim
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

        # Signal completion for consumers expecting a final event
        yield {"type": "complete", "progress": 100}
            
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
    "verify_claim",
    "check_text",
    "check_text_stream",
]
