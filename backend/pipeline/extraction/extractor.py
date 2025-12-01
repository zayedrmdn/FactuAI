"""
extractor.py
Claim extraction from text using LLM.
"""

from core.logging import logger


def extract_claims_llm(text: str, max_claims: int = 5, llm=None) -> list:
    """Extract factual claims from text using LLM."""
    logger.debug(f"[EXTRACTOR] Extracting claims from {len(text)} characters, max_claims={max_claims}")
    
    if llm is None:
        # Get LLM from service manager
        try:
            from services.service_manager import service_manager
            llm = service_manager.get_llm_client()
        except Exception as e:
            logger.error(f"[EXTRACTOR] Failed to get LLM client: {e}")
            return []
    
    # Clear cache before new task
    if hasattr(llm, 'clear_cache'):
        llm.clear_cache()
    
    # Dynamic max_claims based on text length
    word_count = len(text.split())
    dynamic_max_claims = min(max_claims, max(1, word_count // 50))
    logger.debug(f"[EXTRACTOR] Dynamic max claims: {dynamic_max_claims} (based on {word_count} words)")
    
    # Super simple prompt
    system = f"Extract {dynamic_max_claims} factual claims from this text. List them one per line."
    user_msg = text
    
    try:
        logger.debug(f"[EXTRACTOR] Calling LLM for claim extraction...")
        response = llm.generate_response(f"{system}\n\n{user_msg}", max_tokens=512)
        logger.debug(f"[EXTRACTOR] LLM response: {response}")
        
        # Parse response into individual claims
        claims = []
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                # Remove numbering if present (1., 2., etc.)
                if line[0].isdigit() and '. ' in line:
                    line = line.split('. ', 1)[1]
                # Remove bullet points
                if line.startswith('- '):
                    line = line[2:]
                claims.append(line)
                if len(claims) >= dynamic_max_claims:
                    break
        
        logger.debug(f"[EXTRACTOR] Extracted {len(claims)} claims: {claims}")
        return claims
        
    except Exception as e:
        logger.error(f"[EXTRACTOR] Error in claim extraction: {e}")
        return []
