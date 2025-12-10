"""
Test Suite for Tavily API Integration

This module tests the Tavily search integration including:
- Basic search functionality
- Answer generation
- Error handling and fallback behavior
- Integration with the evidence collection pipeline
"""

import pytest
from unittest.mock import patch, MagicMock
from search.tavily import search_tavily
from utils.helpers import SearchError


# ==========================================================================
# Dummy/Mock Tests (No API calls)
# ==========================================================================

class TestTavilySearchMocked:
    """Test Tavily search with mocked API responses"""
    
    def test_search_tavily_empty_query(self):
        """Test that empty query raises SearchError"""
        with pytest.raises(SearchError, match="Query cannot be empty"):
            search_tavily("")
    
    def test_search_tavily_no_api_key(self):
        """Test that missing API key raises SearchError"""
        with patch('factcheck.evidence.TAVILY_API_KEY', None):
            with pytest.raises(SearchError, match="Tavily API key not configured"):
                search_tavily("test query")
    
    def test_search_tavily_success(self):
        """Test successful search with mocked response"""
        # Mock search response
        mock_response = {
            'answer': 'Prabowo Subianto is the current president of Indonesia.',
            'results': [
                {
                    'url': 'https://example.com/article1',
                    'title': 'Indonesia President 2024',
                    'content': 'Prabowo Subianto took office...',
                    'score': 0.95
                }
            ],
            'query': 'Who is the president of Indonesia?',
            'response_time': 1.2
        }
        
        # Mock the TavilyClient - must mock at the point it's imported (inside search_tavily)
        mock_client = MagicMock()
        mock_client.search.return_value = mock_response
        
        with patch('factcheck.evidence.TAVILY_API_KEY', 'test-key'):
            # Mock the import inside search_tavily function
            with patch('tavily.TavilyClient', return_value=mock_client):
                results = search_tavily("Who is the president of Indonesia?")
        
        # Should have AI answer as first result + 1 search result
        assert len(results) == 2
        assert results[0]['answer'] == 'Prabowo Subianto is the current president of Indonesia.'
        assert results[0]['source'] == 'Tavily AI'
        assert results[1]['score'] == 0.95
    
    def test_search_tavily_api_error(self):
        """Test that API errors are caught and wrapped in SearchError"""
        mock_client = MagicMock()
        mock_client.search.side_effect = Exception("API rate limit exceeded")
        
        with patch('factcheck.evidence.TAVILY_API_KEY', 'test-key'):
            with patch('tavily.TavilyClient', return_value=mock_client):
                with pytest.raises(SearchError, match="Tavily search failed"):
                    search_tavily("test query")


# ==========================================================================
# Integration Tests with Evidence Collection
# ==========================================================================

class TestTavilyEvidenceIntegration:
    """Test Tavily integration with evidence collection pipeline"""
    
    def test_tavily_returns_answer_first(self):
        """Test that Tavily returns AI answer as first result"""
        # Mock TavilyClient
        mock_client = MagicMock()
        mock_client.search.return_value = {
            'answer': 'This is the AI generated answer.',
            'results': [
                {'url': 'https://example.com', 'title': 'Article 1', 'content': 'Content 1', 'score': 0.9}
            ]
        }
        
        with patch('factcheck.evidence.TAVILY_API_KEY', 'test-key'):
            with patch('tavily.TavilyClient', return_value=mock_client):
                results = search_tavily("test question")
        
        # First result should be AI answer
        assert len(results) == 2
        assert results[0]['source'] == 'Tavily AI'
        assert results[0]['answer'] == 'This is the AI generated answer.'
        assert results[1]['source'] == 'Tavily'


# ==========================================================================
# Live API Tests (2 tests - requires valid API key)
# ==========================================================================


@pytest.mark.live
class TestTavilyLiveAPI:
    """Live tests with actual Tavily API (requires TAVILY_API_KEY in env)"""
    
    def test_live_simple_fact_query(self):
        """Live test: Simple factual query"""
        from config import TAVILY_API_KEY
        
        if not TAVILY_API_KEY:
            pytest.skip("TAVILY_API_KEY not configured")
        
        results = search_tavily(
            query="Who is the current president of Indonesia in 2025?",
            num_results=3
        )
        
        # Verify response structure
        assert len(results) > 0
        assert 'title' in results[0]
        assert 'url' in results[0]
        assert 'source' in results[0]
        
        # First result should be AI answer
        if results[0].get('answer'):
            assert len(results[0]['answer']) > 20
            print(f"[LIVE TEST] AI Answer: {results[0]['content'][:200]}...")
        
        print(f"[LIVE TEST] Total results: {len(results)}")
    
    def test_live_verification_workflow(self):
        """Live test: Full verification workflow with intent detection"""
        from config import TAVILY_API_KEY
        from pipeline import detect_intent
        
        if not TAVILY_API_KEY:
            pytest.skip("TAVILY_API_KEY not configured")
        
        # Test with a known false claim
        claim = "Trump is the president of Indonesia"
        
        # Detect intent and get verification question
        intent_result = detect_intent(claim)
        
        assert 'verification_question' in intent_result
        verification_question = intent_result['verification_question']
        assert len(verification_question) > 0
        
        print(f"\n[LIVE TEST] Claim: {claim}")
        print(f"[LIVE TEST] Verification Question: {verification_question}")
        
        # Use verification question with Tavily
        results = search_tavily(verification_question, num_results=5)
        
        assert len(results) > 0
        # Check if first result has AI answer
        if results[0].get('answer'):
            answer = results[0]['content']
            assert 'prabowo' in answer.lower() or 'president' in answer.lower()
            print(f"[LIVE TEST] Tavily Answer: {answer[:300]}...")
# ==========================================================================
# Pytest Configuration
# ==========================================================================

# ==========================================================================
# Pytest Configuration
# ==========================================================================

def pytest_configure(config):
    """Register custom markers"""
    config.addinivalue_line(
        "markers",
        "live: mark test as requiring live API calls (deselect with -m 'not live')"
    )