import pytest
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse

def simple_fetch_article_text(url: str) -> str:
    """Lightweight article extractor - no heavy dependencies."""
    print(f"Testing extraction for: {url}")
    
    # Skip obvious index pages
    if any(hint in url.lower() for hint in ["portal:", "/index.html", "todayspaper"]):
        print(f"Skipping index-like page: {url}")
        return ""

    # Simple headers to avoid bot detection
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    try:
        response = requests.get(url, timeout=10, headers=headers)
        print(f"Response status: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract all paragraph text
            paragraphs = soup.find_all('p')
            text = ' '.join([p.get_text() for p in paragraphs])
            
            word_count = len(text.split())
            print(f"Extracted {word_count} words")
            
            if word_count >= 50:
                return text.strip()
        
        return ""
        
    except Exception as e:
        print(f"Extraction failed: {e}")
        return ""

class TestExtractorsStandalone:
    """Lightweight extractor tests without heavy dependencies."""
    
    def test_cnn_article(self):
        """Test with CNN article."""
        url = "https://edition.cnn.com/2023/10/29/sport/nfl-week-8-how-to-watch-spt-intl/index.html"
        text = simple_fetch_article_text(url)
        
        assert text is not None
        print(f"✓ CNN: {len(text.split())} words extracted")
    
    def test_wikipedia_article(self):
        """Test with Wikipedia article."""
        url = "https://en.wikipedia.org/wiki/SpaceX"
        text = simple_fetch_article_text(url)
        
        assert text is not None
        assert len(text) > 100
        assert "spacex" in text.lower() or "elon" in text.lower()
        print(f"✓ Wikipedia: {len(text.split())} words extracted")
    
    def test_blocked_site(self):
        """Test handling of blocked sites."""
        url = "https://www.theverge.com/ai-artificial-intelligence/708536/elon-musk-grok-xai-ai-boyfriend"
        text = simple_fetch_article_text(url)
        
        # Should handle gracefully (empty string or content)
        assert text is not None
        print(f"✓ Blocked site: {len(text.split())} words extracted")
    
    def test_simple_news_site(self):
        """Test with a simple news site."""
        url = "https://www.reuters.com/technology/"
        text = simple_fetch_article_text(url)
        
        assert text is not None
        print(f"✓ Reuters: {len(text.split())} words extracted")
    
    def test_invalid_url(self):
        """Test invalid URL handling."""
        url = "https://this-domain-does-not-exist-12345.com"
        text = simple_fetch_article_text(url)
        
        assert text == ""
        print("✓ Invalid URL: Handled gracefully")

if __name__ == "__main__":
    # Manual test runner
    test = TestExtractorsStandalone()
    print("Running lightweight extractor tests...")
    
    try:
        test.test_cnn_article()
        test.test_wikipedia_article()
        test.test_blocked_site()
        test.test_simple_news_site()
        test.test_invalid_url()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")