import os
import pytest
from dotenv import load_dotenv
from services import llm
from search.base import collect_evidence

load_dotenv(dotenv_path="D:/Projects/FactuAI/backend/.env")

REQUIRED_VARS = ["GOOGLE_API_KEY", "NEWS_API_KEY"]

@pytest.fixture(scope="module")
def live_enabled():
    return all(os.getenv(k) for k in REQUIRED_VARS)

@pytest.mark.slow
@pytest.mark.live
@pytest.mark.skipif(
    not all(os.getenv(k) for k in REQUIRED_VARS),
    reason="API keys not configured"
)
@pytest.mark.parametrize("claim", [
    "OpenAI released GPT-5 in January 2025",
])
def test_evidence_live_fetch(claim):
    """Test evidence collection with real API calls."""
    # Initialize LLM
    llm.initialize()
    
    # Collect evidence using the new simplified API
    result = collect_evidence(claim, max_results=5)
    
    print("\n=== CLAIM ===")
    print(claim)
    print("\n=== EVIDENCE TEXT ===")
    print(result['evidence_text'])
    print("\n=== SOURCES ===")
    for source in result['sources']:
        print(f"- {source['title']}: {source['url']}")
    print("\n=== TOP QUOTES ===")
    for quote in result['top_quotes'][:3]:
        print(f"• {quote}")

    assert isinstance(result['evidence_text'], str)
    assert len(result['evidence_text'].split()) >= 10, "Evidence too short"
    assert isinstance(result['sources'], list)
    assert len(result['sources']) > 0, "No sources found"
    assert isinstance(result['top_quotes'], list)