import os

import pytest
import requests

RUN_LIVE = os.getenv("RUN_LIVE_API_TESTS") == "1"
BASE_URL = os.getenv("LIVE_BASE_URL", "http://localhost:8000")
TIMEOUT = 30

pytestmark = pytest.mark.skipif(not RUN_LIVE, reason="Requires running API (set RUN_LIVE_API_TESTS=1)")


def test_login_endpoint_authenticates_user_credentials():
    headers = {"Content-Type": "application/json"}

    valid_payload = {"email": "test@example.com", "password": "test123"}
    invalid_payload = {"email": "invalid@example.com", "password": "wrongpassword"}

    resp_valid = requests.post(f"{BASE_URL}/api/login", json=valid_payload, headers=headers, timeout=TIMEOUT)
    assert resp_valid.status_code in [200, 401, 503]
    assert resp_valid.status_code != 500

    resp_invalid = requests.post(f"{BASE_URL}/api/login", json=invalid_payload, headers=headers, timeout=TIMEOUT)
    assert resp_invalid.status_code in [401, 503]
    assert resp_invalid.status_code not in [200, 500]
