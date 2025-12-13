import os

import pytest
import requests

RUN_LIVE = os.getenv("RUN_LIVE_API_TESTS") == "1"
BASE_URL = os.getenv("LIVE_BASE_URL", "http://localhost:8000")
TIMEOUT = 30

pytestmark = pytest.mark.skipif(not RUN_LIVE, reason="Requires running API (set RUN_LIVE_API_TESTS=1)")


def test_analyze_endpoint_processes_multi_claim_text_input():
    url = f"{BASE_URL}/api/analyze"
    headers = {"Content-Type": "application/json"}
    payload = {
        "text": (
            "Claim one: The Earth revolves around the Sun. "
            "Claim two: Water boils at 100 degrees Celsius at sea level. "
            "Claim three: The Great Wall of China is visible from space."
        ),
        "provider": "openrouter",
        "max_claims": 3,
        "enable_web_search": False,
        "enable_kb": False,
    }

    response = requests.post(url, json=payload, headers=headers, timeout=TIMEOUT)
    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert data.get("request_id")
    assert data.get("model_used")
    assert isinstance(data.get("latency_ms"), (int, float)) and data["latency_ms"] >= 0

    claims = data.get("claims")
    assert isinstance(claims, list)
    assert 1 <= len(claims) <= payload["max_claims"]

    for claim in claims:
        assert isinstance(claim.get("claim_text"), str) and claim["claim_text"]
        assert isinstance(claim.get("verdict"), str) and claim["verdict"]
        assert isinstance(claim.get("confidence"), (int, float))
        assert (0 <= claim["confidence"] <= 1) or (0 <= claim["confidence"] <= 100)
        assert isinstance(claim.get("reasoning"), str) and claim["reasoning"]

        evidence = claim.get("evidence")
        assert isinstance(evidence, list)
        for ev in evidence:
            assert isinstance(ev.get("snippet"), str) and ev["snippet"]
            assert isinstance(ev.get("source_url"), str) and ev["source_url"]
            assert isinstance(ev.get("source_domain"), str) and ev["source_domain"]
            assert isinstance(ev.get("relevance_score"), (int, float))
            assert (0 <= ev["relevance_score"] <= 1) or (0 <= ev["relevance_score"] <= 100)
