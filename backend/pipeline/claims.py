"""
Claim extraction module.

Extracts factual claims from multi-claim text.
"""

import re
from typing import List

from utils.logging import get_logger
from services.llm import chat

logger = get_logger(__name__)


def extract_claims(text: str, max_claims: int = 5, llm: str = None, model_id: str = None) -> List[str]:
    """
    Extract factual claims from text.
    
    Args:
        text: Input text
        max_claims: Maximum number of claims to extract
        llm: Optional LLM provider
        model_id: Optional model identifier
        
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
        
        response = chat(
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
            
            # Clean up formatting markers
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


__all__ = ["extract_claims"]
