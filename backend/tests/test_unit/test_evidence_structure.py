import pytest
from unittest.mock import MagicMock, patch
from factcheck import evidence

def test_collect_evidence_returns_title():
    """Test that collect_evidence returns items with 'title' key."""
    
    # Mock search results
    mock_google_results = [
        {"title": "Google Title", "url": "http://google.com", "source": "Google"}
    ]
    
    # Mock scraping and extraction
    mock_text = "This is a sentence."
    mock_sentences = ["This is a sentence."]
    mock_ranked = [("This is a sentence.", 0.9)]
    
    with patch('factcheck.evidence.search_google', return_value=mock_google_results), \
         patch('factcheck.evidence.scrape_article', return_value=mock_text), \
         patch('factcheck.evidence.extract_sentences', return_value=mock_sentences), \
         patch('factcheck.evidence.rank_sentences', return_value=mock_ranked), \
         patch.dict(evidence.PROVIDER_FUNCTIONS, {'google': evidence.search_google}):
         
        items = evidence.collect_evidence(
            claim="test",
            google_query="test",
            newsapi_query="test",
            enabled_providers=['google']
        )
        
        assert len(items) > 0
        for item in items:
            assert "title" in item
            assert item["title"] == "Google Title"
            assert "url" in item
            assert "source" in item
            assert "score" in item

def test_collect_evidence_tavily_title():
    """Test that Tavily answer has a title."""
    
    mock_tavily_results = [
        {
            "content": "Tavily answer", 
            "url": "http://tavily.com", 
            "answer": "Yes", 
            "title": "Tavily Title"
        }
    ]
    
    with patch('factcheck.evidence.search_tavily', return_value=mock_tavily_results), \
         patch.dict(evidence.PROVIDER_FUNCTIONS, {'tavily': evidence.search_tavily}):
         
        items = evidence.collect_evidence(
            claim="test",
            google_query="test",
            newsapi_query="test",
            enabled_providers=['tavily']
        )
        
        # Should have at least the answer item
        assert len(items) > 0
        answer_item = items[0]
        assert "title" in answer_item
        assert answer_item["title"] == "Tavily Title"
