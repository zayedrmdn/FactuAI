"""
Intent detection module.

Detects user intent and generates optimized search queries.
"""

import re
import json
from typing import Dict, Optional

from utils.logging import get_logger
from services.llm import chat

logger = get_logger(__name__)


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
        for intent_type in ["fact_claim", "fact_question", "multi_claim", "opinion", "nonsense", "instructional"]:
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
        logger.warning(f"[INTENT] Fallback extraction failed: {e}")
        return None


def detect_intent(text: str, llm: str = None, model_id: str = None) -> Dict[str, str]:
    """
    Detect intent and generate optimized search queries in a single LLM call.
    
Intent categories:
- fact_claim: Single factual claim
- fact_question: Question about facts
- multi_claim: Multiple claims or news paragraph with multiple claims
- opinion: Subjective opinion
- nonsense: Invalid/nonsense input
- instructional: How-to/instruction text    Args:
        text: Input text to classify
        llm: Optional LLM provider (uses default if None)
        model_id: Optional model identifier
        
    Returns:
        Dict with keys: 'intent', 'google_query', 'newsapi_query', 'verification_question'
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
1. intent must be one of: fact_claim, fact_question, multi_claim, opinion, nonsense, instructional

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
        
        response = chat(
            system,
            f"Text: {text_stripped}",
            provider=llm,
            model_id=model_id,
            max_tokens=500,
            temperature=0.3,
        )
        
        # Parse JSON response with robust error handling
        response_clean = response.strip()
        
        # Handle reasoning models
        if not response_clean.startswith('{'):
            last_brace = response_clean.rfind('{')
            if last_brace != -1:
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
                        response_clean = potential_json
                    except json.JSONDecodeError:
                        pass
        
        # Handle markdown code blocks
        if response_clean.startswith('```'):
            parts = response_clean.split('```')
            if len(parts) >= 3:
                for part in parts[1:-1]:
                    part = part.strip()
                    if part.startswith('json'):
                        part = part[4:].strip()
                    if part.startswith('{') and part.endswith('}'):
                        response_clean = part
                        break
                else:
                    response_clean = parts[1].strip()
                    if response_clean.startswith('json'):
                        response_clean = response_clean[4:].strip()
            else:
                response_clean = response_clean.replace('```', '').strip()
        
        # Try to find JSON object
        json_start = response_clean.find('{')
        json_end = response_clean.rfind('}') + 1
        
        if json_start != -1 and json_end > json_start:
            potential_json = response_clean[json_start:json_end]
            try:
                result = json.loads(potential_json)
                response_clean = potential_json
            except json.JSONDecodeError:
                pass
        
        # Final JSON parsing
        try:
            result = json.loads(response_clean)
        except json.JSONDecodeError as e:
            logger.warning(f"[INTENT] Failed to parse JSON response: {e}")
            fallback_result = extract_fallback_from_text(response)
            if fallback_result:
                logger.info(f"[INTENT] Using fallback extraction: {fallback_result}")
                result = fallback_result
            else:
                raise
        
        # Validate intent
        valid_intents = ["fact_claim", "fact_question", "multi_claim", "opinion", "nonsense", "instructional"]
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
                verification_question = text_stripped
            else:
                verification_question = f"Is this true: {text_stripped[:150]}?"
        
        logger.debug(f"[INTENT] Detected: {intent}, Google: {google_query[:50]}..., NewsAPI: {newsapi_query[:50]}...")
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


__all__ = ["detect_intent", "extract_fallback_from_text"]
