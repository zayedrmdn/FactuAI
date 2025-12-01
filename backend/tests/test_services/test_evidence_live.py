import os
import pytest
from dotenv import load_dotenv
from backend.services.llm.llm_client import QwenClient
from services.search.google_search import GoogleSearchClient
from pipeline.factcheck.claims.fetchers.newsapi import fetch_newsapi_articles
from pipeline.factcheck.claims.pipeline import build_evidence

load_dotenv(dotenv_path="D:/Projects/FactuAI/backend/.env")

REQUIRED_VARS = ["GOOGLE_API_KEY", "NEWSAPI_KEY"]

@pytest.fixture(scope="module")
def live_enabled():
    return all(os.getenv(k) for k in REQUIRED_VARS)

@pytest.fixture(scope="module")
def shared_llm():
    return QwenClient()

@pytest.fixture(scope="module")
def search_client():
    return GoogleSearchClient()

@pytest.mark.slow
@pytest.mark.live
@pytest.mark.skipif(
    not all(os.getenv(k) for k in REQUIRED_VARS),
    reason="API keys not configured"
)
@pytest.mark.parametrize("claim", [
    "OpenAI released GPT-5 in January 2025",
])
def test_evidence_live_fetch_with_newsapi(claim, shared_llm, search_client):
    query = search_client.build_query(claim, shared_llm)

    google_results = search_client.google_fetch(query, num=3)
    newsapi_results = fetch_newsapi_articles(claim, max_results=2)

    google_items = google_results.get("items", []) if isinstance(google_results, dict) else google_results

    assert google_items or newsapi_results, "No search results fetched"

    search_resp = {"items": google_items + newsapi_results}

    evidence, urls, quotes = build_evidence(
        search_resp=search_resp,
        claim=claim,
        llm=shared_llm
    )

    print("\n=== CLAIM ===")
    print(claim)
    print("\n=== EVIDENCE ===")
    print(evidence)
    print("\n=== URLS ===")
    for u in urls:
        print("-", u)
    print("\n=== QUOTES ===")
    for q in quotes:
        print(f"• {q['quote']} — {q['source']}")

    assert isinstance(evidence, str)
    assert len(evidence.split()) >= 3, "Evidence too short"
    assert isinstance(urls, list)
    assert isinstance(quotes, list)