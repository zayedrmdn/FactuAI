import pytest
from unittest.mock import MagicMock, patch
from factcheck.pipeline import extract_claims

def test_extract_claims_with_dashes():
    """Test that claims starting with dashes are correctly extracted."""
    mock_llm_response = """
- Heavy tropical storms and torrential monsoon rains have triggered massive floods.
- The downpours have ravaged Sri Lanka, Indonesia, Thailand and Malaysia.
- More than 1,800 people have been killed by flooding.
    """
    
    with patch('factcheck.llm_client.chat', return_value=mock_llm_response):
        # Pass enough text to allow multiple claims (50 words per claim approx)
        long_text = "word " * 200 
        claims = extract_claims(long_text)
        
        assert len(claims) == 3
        assert claims[0] == "Heavy tropical storms and torrential monsoon rains have triggered massive floods."
        assert claims[1] == "The downpours have ravaged Sri Lanka, Indonesia, Thailand and Malaysia."
        assert claims[2] == "More than 1,800 people have been killed by flooding."

def test_extract_claims_with_numbers():
    """Test that claims starting with numbers are correctly extracted."""
    mock_llm_response = """
1. First claim.
2. Second claim.
    """
    
    with patch('factcheck.llm_client.chat', return_value=mock_llm_response):
        # Pass enough text to allow multiple claims
        long_text = "word " * 200
        claims = extract_claims(long_text)
        
        assert len(claims) == 2
        assert claims[0] == "First claim."
        assert claims[1] == "Second claim."

def test_extract_claims_no_substance_filtering():
    """Test that claims without specific 'substance' keywords are NOT filtered out."""
    # This claim doesn't have words like 'cure', 'cause', 'prevent' etc.
    mock_llm_response = "- The sky is blue."
    
    with patch('factcheck.llm_client.chat', return_value=mock_llm_response):
        claims = extract_claims("some text")
        
        assert len(claims) == 1
        assert claims[0] == "The sky is blue."

def test_extract_claims_filtering_metadata():
    """Test that metadata lines are filtered out."""
    mock_llm_response = """
- Valid claim.
Source: Wikipedia
Note: This is a note.
Disclaimer: Not advice.
    """
    
    with patch('factcheck.llm_client.chat', return_value=mock_llm_response):
        claims = extract_claims("some text")
        
        assert len(claims) == 1
        assert claims[0] == "Valid claim."
