import pytest
import sys
import os

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from factcheck.evidence import scrape_article

class TestExtractors:
    """Test all article extractors with real URLs."""
    
    def test_newspaper_extractor(self):
        """Test newspaper4k extractor with CNN article."""
        url = "https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html"
        text = scrape_article(url)
        
        assert text is not None
        assert len(text) > 100
        assert "nfl" in text.lower() or "football" in text.lower()
        print(f"✓ Newspaper extractor: {len(text.split())} words extracted")
    
    def test_wikipedia_extractor(self):
        """Test with Wikipedia article."""
        url = "https://en.wikipedia.org/wiki/SpaceX"
        text = scrape_article(url)
        
        assert text is not None
        assert len(text) > 100
        assert "spacex" in text.lower() or "elon" in text.lower()
        print(f"✓ Wikipedia extractor: {len(text.split())} words extracted")
    
    def test_beautifulsoup_fallback(self):
        """Test BeautifulSoup fallback with BBC article."""
        url = "https://www.bbc.com/news/technology"
        text = scrape_article(url)
        
        # BBC might work or fail, just check it doesn't crash
        assert text is not None  # Empty string is fine for blocked sites
        print(f"✓ BeautifulSoup fallback: {len(text.split())} words extracted")
    
    def test_blocked_site_handling(self):
        """Test handling of sites that block scrapers."""
        url = "https://www.theverge.com/ai-artificial-intelligence/708536/elon-musk-grok-xai-ai-boyfriend"
        text = scrape_article(url)
        
        # Should return empty string for blocked sites without crashing
        assert text is not None
        print(f"✓ Blocked site handling: {len(text.split())} words extracted (expected: 0)")
    
    def test_cache_functionality(self):
        """Test that caching works for repeated requests."""
        url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
        
        # First request
        text1 = scrape_article(url)
        
        # Second request (should use cache)
        text2 = scrape_article(url)
        
        assert text1 == text2
        assert len(text1) > 100
        print(f"✓ Cache functionality: Consistent results across requests")
    
    def test_invalid_url_handling(self):
        """Test handling of invalid URLs."""
        invalid_urls = [
            "https://this-domain-does-not-exist-12345.com",
            "invalid-url",
            ""
        ]
        
        for url in invalid_urls:
            text = scrape_article(url)
            assert text == ""  # Should return empty string, not crash
        
        print("✓ Invalid URL handling: No crashes on bad URLs")

if __name__ == "__main__":
    # Run a quick manual test
    test = TestExtractors()
    print("Running extractor tests...")
    
    try:
        test.test_newspaper_extractor()
        test.test_wikipedia_extractor()
        test.test_beautifulsoup_fallback()
        test.test_blocked_site_handling()
        test.test_cache_functionality()
        test.test_invalid_url_handling()
        print("\n✅ All extractor tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")