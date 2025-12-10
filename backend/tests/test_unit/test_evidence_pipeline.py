"""
Unit tests for the evidence collection module.
"""
import pytest
from unittest.mock import Mock, patch, MagicMock
from factcheck import evidence


# build_search_query was removed from evidence.py as it's now handled by LLM in detect_intent
# Removing obsolete tests: test_build_search_query and test_build_search_query_empty

def test_scrape_article_returns_text():
    """Test that article scraping returns text content."""
    with patch('factcheck.evidence.requests.get') as mock_get:
        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.content = b'<html><body><p>Test article content with substantial text.</p></body></html>'
        mock_get.return_value = mock_response
        
        url = "https://example.com/test"
        text = evidence.scrape_article(url)
        
        assert isinstance(text, str)
        assert len(text) > 0
        assert "Test article content" in text


def test_scrape_article_handles_errors():
    """Test that scraping handles errors gracefully."""
    with patch('factcheck.evidence.requests.get') as mock_get:
        mock_get.side_effect = Exception("Network error")
        
        url = "https://example.com/test"
        text = evidence.scrape_article(url)
        
        # Should return empty string on error
        assert text == ""


def test_extract_sentences():
    """Test sentence extraction from text."""
    text = "This is a first sentence with many words. This is a second sentence with many words. This is a third sentence with many words."
    sentences = evidence.extract_sentences(text)
    
    assert isinstance(sentences, list)
    assert len(sentences) >= 2  # Some sentences might be filtered out
    assert any("first sentence" in s for s in sentences)


def test_extract_sentences_empty():
    """Test sentence extraction from empty text."""
    sentences = evidence.extract_sentences("")
    assert sentences == []


def test_rank_sentences_returns_scored_tuples():
    """Test that rank_sentences returns sentences with similarity scores."""
    sentences = [
        "OpenAI released GPT-5 in January 2025 with advanced capabilities.",
        "The weather is nice today and sunny all around.",
        "GPT-5 features advanced reasoning capabilities and improved performance.",
        "I like pizza and other Italian foods very much."
    ]
    claim = "OpenAI released GPT-5"
    
    with patch('factcheck.evidence._get_embed_model') as mock_get_model:
        # Mock the model to return None (triggers fallback)
        mock_get_model.return_value = None
        
        ranked = evidence.rank_sentences(claim, sentences)
        
        assert isinstance(ranked, list)
        assert len(ranked) == 4
        # Check it returns (sentence, score) tuples  
        assert all(isinstance(item, tuple) and len(item) == 2 for item in ranked)
        assert all(isinstance(item[0], str) and isinstance(item[1], float) for item in ranked)
        # Fallback returns 0.0 scores
        assert all(item[1] == 0.0 for item in ranked)


@patch('factcheck.evidence.search_google')
@patch('factcheck.evidence.search_newsapi')
@patch('factcheck.evidence.scrape_article')
@patch('factcheck.evidence.extract_sentences')
@patch('factcheck.evidence.rank_sentences')
def test_collect_evidence_integration(mock_rank, mock_extract, mock_scrape, mock_news, mock_google):
    """Test the full collect_evidence pipeline."""
    # Setup mocks
    mock_google.return_value = [
        {"title": "Article 1", "url": "http://example.com/1", "snippet": "Snippet 1", "source": "Google"}
    ]
    mock_news.return_value = [
        {"title": "News 1", "url": "http://news.com/1", "snippet": "News snippet", "source": "NewsAPI"}
    ]
    mock_scrape.return_value = "Full article text content here with many words to meet minimum requirements"
    mock_extract.return_value = ["Sentence 1 with enough words", "Sentence 2 with enough words", "Sentence 3 with enough words"]
    # rank_sentences returns list of (sentence, score) tuples
    mock_rank.return_value = [("Sentence 1 with enough words", 0.9), ("Sentence 2 with enough words", 0.8), ("Sentence 3 with enough words", 0.7)]
    
    claim = "Test claim for evidence"
    result = evidence.collect_evidence(
        claim, 
        google_query=claim, 
        newsapi_query=claim, 
        num_google=5, 
        num_news=5, 
        top_k=10
    )
    
    # Verify structure - collect_evidence returns a list, not a dict
    assert isinstance(result, list)
    
    # Verify API calls were made
    mock_google.assert_called_once()
    mock_news.assert_called_once()


def test_search_google_requires_api_key():
    """Test that Google search requires API key."""
    with patch('factcheck.evidence.GOOGLE_API_KEY', None):
        with patch('factcheck.evidence.GOOGLE_CX_ID', 'test-id'):
            try:
                results = evidence.search_google("test query")
                # Should either return empty list or raise SearchError
                assert results == []
            except Exception as e:
                # SearchError is expected when credentials are missing
                assert "credentials" in str(e).lower() or "api" in str(e).lower()


def test_search_newsapi_requires_api_key():
    """Test that NewsAPI search requires API key."""
    with patch('factcheck.evidence.NEWS_API_KEY', None):
        results = evidence.search_newsapi("test query")
        assert results == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
