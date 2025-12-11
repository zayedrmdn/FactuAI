"""
Test suite for reasoning model JSON parsing in intent detection

Tests the robust JSON extraction from various reasoning model response formats.
Includes both simulated responses and live API calls to OpenRouter GLM 4.5 Air.
"""

import pytest
import json
import os
from unittest.mock import patch, MagicMock
from pipeline import detect_intent
from utils.logging import get_logger

logger = get_logger(__name__)


class TestReasoningModelParsing:
    """Test JSON parsing from reasoning model responses"""
    
    def test_reasoning_with_final_json(self):
        """Test Case 1: Reasoning model with step-by-step analysis followed by JSON"""
        mock_response = """Let me analyze this claim about COVID vaccines containing microchips.

First, I need to identify the intent. The user is making a definitive statement about vaccines containing tracking devices. This is a conspiracy theory claim that requires fact-checking, so the intent is clearly a fact_claim.

Next, I'll construct search queries:

For Google (4-8 multiword terms):
- "covid vaccine microchip tracking conspiracy debunk"
- "bill gates vaccine microchip myth"
- "covid vaccine conspiracy theories"
- "vaccine microchip claims false"
- "tracking chips vaccines hoax"
- "covid vaccine rfid chip"
- "microchip vaccine conspiracy debunked"

For NewsAPI (3-6 shorter keywords):
- "covid vaccine microchip"
- "vaccine conspiracy"
- "tracking chips"
- "bill gates vaccine"
- "microchip conspiracy"
- "vaccine debunked"

Final output:
{
  "intent": "fact_claim",
  "google_query": "covid vaccine microchip tracking conspiracy debunk",
  "newsapi_query": "covid vaccine microchip"
}"""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("COVID vaccines contain microchips for tracking")
            
            logger.info(f"[TEST] Case 1 Result: {json.dumps(result, indent=2)}")
            
            assert result["intent"] == "fact_claim"
            assert "microchip" in result["google_query"].lower()
            assert "microchip" in result["newsapi_query"].lower()
            assert len(result["google_query"].split()) >= 3
            assert len(result["newsapi_query"].split()) >= 2
    
    def test_reasoning_with_markdown_json(self):
        """Test Case 2: Reasoning model outputting JSON in markdown code block"""
        mock_response = """Analysis:
The user is asking about climate change causes, which is a factual question requiring verification.

Intent Classification:
- This is a fact_question because it's seeking factual information
- It's not opinion or instructional

Search Query Strategy:
- Google needs comprehensive terms for scientific articles
- NewsAPI needs concise keywords for news articles

```json
{
  "intent": "fact_question",
  "google_query": "climate change causes greenhouse gases carbon emissions deforestation",
  "newsapi_query": "climate change causes"
}
```"""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("What causes climate change?")
            
            logger.info(f"[TEST] Case 2 Result: {json.dumps(result, indent=2)}")
            
            assert result["intent"] == "fact_question"
            assert "climate" in result["google_query"].lower()
            assert "climate" in result["newsapi_query"].lower()
    
    def test_reasoning_with_json_embedded(self):
        """Test Case 3: JSON embedded in middle of reasoning text"""
        mock_response = """Step 1: Identify the core claim
The text states "5G towers cause cancer" which is a conspiracy theory claim.

Step 2: Generate queries
{
  "intent": "fact_claim",
  "google_query": "5g towers cancer health effects scientific studies debunk",
  "newsapi_query": "5g cancer health"
}

Step 3: Validation
The intent is correct as fact_claim, and queries are optimized for each provider."""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("5G towers cause cancer")
            
            logger.info(f"[TEST] Case 3 Result: {json.dumps(result, indent=2)}")
            
            assert result["intent"] == "fact_claim"
            assert "5g" in result["google_query"].lower()
            assert len(result["google_query"].split()) >= 3
    
    def test_reasoning_with_complex_structure(self):
        """Test Case 4: Complex reasoning with multiple JSON-like structures"""
        mock_response = """Reasoning Process:

Input Analysis:
- Text length: 45 characters
- Contains opinion markers: "I think"
- Sentiment: subjective

Intent Matrix:
{ "fact_claim": 0.1, "opinion": 0.9, "question": 0.0 }

Search Term Extraction:
- Primary: "artificial intelligence job market"
- Secondary: "ai unemployment automation"
- Tertiary: "future jobs technology"

FINAL OUTPUT:
{
  "intent": "opinion",
  "google_query": "artificial intelligence job market impact automation employment future",
  "newsapi_query": "ai jobs unemployment"
}

Confidence: 95%"""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("I think AI will take all our jobs")
            
            logger.info(f"[TEST] Case 4 Result: {json.dumps(result, indent=2)}")
            
            # Should extract the FINAL OUTPUT JSON
            assert result["intent"] == "opinion"
            assert "artificial intelligence" in result["google_query"].lower() or "ai" in result["google_query"].lower()
    
    def test_reasoning_with_malformed_response(self):
        """Test Case 5: Malformed response that requires fallback extraction"""
        mock_response = """Analysis of the claim about moon landing being fake:

This is clearly a fact_claim intent type.

Search queries should be:
- Google: moon landing fake hoax conspiracy theory debunked apollo evidence
- NewsAPI: moon landing fake debunked

Note: No proper JSON formatting provided, testing fallback"""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("The moon landing was fake")
            
            logger.info(f"[TEST] Case 5 Result: {json.dumps(result, indent=2)}")
            
            # Should use fallback extraction or return valid structure
            assert "intent" in result
            assert "google_query" in result
            assert "newsapi_query" in result
            assert result["intent"] in ["fact_claim", "fact_question", "multi_claim", "opinion", "nonsense", "instructional"]


class TestLiveAPIReasoningModels:
    """Test with actual OpenRouter API using GLM 4.5 Air reasoning model"""
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping live API test"
    )
    def test_live_glm4_5_air_simple_claim(self):
        """Test Case 6: Live API call with GLM 4.5 Air - Simple fact claim"""
        test_claim = "The Earth is flat"
        
        logger.info(f"[TEST] Live API Test 1: Testing with claim: '{test_claim}'")
        
        # Call with GLM 4.5 Air model
        result = detect_intent(
            test_claim,
            llm="openrouter",
            model_id="z-ai/glm-4.5-air:free"  # GLM 4.5 Air model
        )
        
        logger.info(f"[TEST] Live API Test 1 Result: {json.dumps(result, indent=2)}")
        
        # Validate structure
        assert "intent" in result
        assert "google_query" in result
        assert "newsapi_query" in result
        
        # Validate intent is reasonable
        assert result["intent"] in ["fact_claim", "fact_question", "multi_claim", "opinion", "nonsense", "instructional"]
        
        # Validate queries are not empty
        assert len(result["google_query"]) > 0
        assert len(result["newsapi_query"]) > 0
        
        # Validate queries contain relevant terms
        assert any(term in result["google_query"].lower() for term in ["earth", "flat", "sphere", "planet"])
        
        logger.info(f"[TEST] Live API Test 1: ✅ PASSED")
    
    @pytest.mark.skipif(
        not os.getenv("OPENROUTER_API_KEY"),
        reason="OPENROUTER_API_KEY not set - skipping live API test"
    )
    def test_live_glm4_5_air_complex_claim(self):
        """Test Case 7: Live API call with GLM 4.5 Air - Complex multi-aspect claim"""
        test_claim = "Recent studies show that drinking coffee reduces the risk of Alzheimer's disease by 30% in people over 65"
        
        logger.info(f"[TEST] Live API Test 2: Testing with claim: '{test_claim}'")
        
        # Call with GLM 4.5 Air model
        result = detect_intent(
            test_claim,
            llm="openrouter",
            model_id="z-ai/glm-4.5-air:free"  # GLM 4.5 Air model
        )
        
        logger.info(f"[TEST] Live API Test 2 Result: {json.dumps(result, indent=2)}")
        
        # Validate structure
        assert "intent" in result
        assert "google_query" in result
        assert "newsapi_query" in result
        
        # Validate intent is reasonable for this type of claim
        assert result["intent"] in ["fact_claim", "fact_question", "multi_claim"]
        
        # Validate queries are substantial (not just fallback)
        assert len(result["google_query"].split()) >= 3
        assert len(result["newsapi_query"].split()) >= 2
        
        # Validate queries contain relevant medical terms
        google_lower = result["google_query"].lower()
        newsapi_lower = result["newsapi_query"].lower()
        
        # At least one query should mention coffee or alzheimer's
        assert any(term in google_lower for term in ["coffee", "alzheimer", "caffeine"])
        assert any(term in newsapi_lower for term in ["coffee", "alzheimer", "caffeine"])
        
        # Validate queries are optimized differently
        # Google should have more detailed terms (4-8 words)
        # NewsAPI should be more concise (3-6 words)
        assert len(result["google_query"].split()) >= len(result["newsapi_query"].split()) or \
               len(result["google_query"]) >= len(result["newsapi_query"])
        
        logger.info(f"[TEST] Live API Test 2: ✅ PASSED")


class TestEdgeCases:
    """Test edge cases and production readiness"""
    
    def test_empty_response_handling(self):
        """Test Case 8: Handle empty or minimal LLM response"""
        with patch('services.llm.chat', return_value="{}"):
            result = detect_intent("Test claim")
            
            logger.info(f"[TEST] Edge Case 1 Result: {json.dumps(result, indent=2)}")
            
            # Should return valid structure with fallback values
            assert "intent" in result
            assert "google_query" in result
            assert "newsapi_query" in result
    
    def test_json_with_extra_fields(self):
        """Test Case 9: JSON response with extra fields (should be ignored gracefully)"""
        mock_response = """{
  "intent": "fact_claim",
  "google_query": "extra fields test query google search",
  "newsapi_query": "extra fields test",
  "confidence": 0.95,
  "reasoning_chain": ["step1", "step2"],
  "metadata": {"model": "test", "version": "1.0"}
}"""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("Test claim with extra fields")
            
            logger.info(f"[TEST] Edge Case 2 Result: {json.dumps(result, indent=2)}")
            
            # Should extract only the required fields
            assert result["intent"] == "fact_claim"
            assert result["google_query"] == "extra fields test query google search"
            assert result["newsapi_query"] == "extra fields test"
            
            # Should not include extra fields
            assert "confidence" not in result
            assert "reasoning_chain" not in result
            assert "metadata" not in result
    
    def test_unicode_and_special_chars(self):
        """Test Case 10: Handle unicode and special characters in reasoning response"""
        mock_response = """🤔 Analyzing the claim about émigré populations...

The text contains unicode characters: émigré, café, naïve

Intent: This is a fact_question about immigration patterns.

Search Strategy 🔍:
- Google needs comprehensive terms
- NewsAPI needs concise keywords

Result:
{
  "intent": "fact_question",
  "google_query": "emigre immigration population statistics demographics trends",
  "newsapi_query": "emigre immigration statistics"
}

✅ Analysis complete"""
        
        with patch('services.llm.chat', return_value=mock_response):
            result = detect_intent("What is the émigré population in the US?")
            
            logger.info(f"[TEST] Edge Case 3 Result: {json.dumps(result, indent=2)}")
            
            assert result["intent"] == "fact_question"
            assert len(result["google_query"]) > 0
            assert len(result["newsapi_query"]) > 0


def test_centralized_logging():
    """Test Case 11: Verify centralized logging is being used"""
    from pipeline import detect_intent
    
    # Check that pipeline uses utils.logging.get_logger
    assert hasattr(pipeline, 'logger')
    # Logger name includes the full module path
    assert 'pipeline' in pipeline.logger.name
    assert hasattr(pipeline.logger, 'info')  # Verify it's a proper logger
    
    logger.info("[TEST] Centralized logging verification: ✅ PASSED")


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short", "-s"])
