import os

import pytest
import requests

RUN_LIVE = os.getenv("RUN_LIVE_API_TESTS") == "1"
BASE_URL = os.getenv("LIVE_BASE_URL", "http://localhost:8000")
TIMEOUT = 30

pytestmark = pytest.mark.skipif(not RUN_LIVE, reason="Requires running API (set RUN_LIVE_API_TESTS=1)")


def test_health_endpoint_returns_service_liveness_status():
    url = f"{BASE_URL}/health"
    response = requests.get(url, timeout=TIMEOUT)

    assert response.status_code == 200

    data = response.json()
    assert isinstance(data, dict)
    assert data.get("status") == "ok"
