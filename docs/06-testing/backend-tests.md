# Backend Testing Guide

Using pytest for testing the FactuAI backend.

---

## Overview

**Framework:** pytest  
** Test Location:** `backend/tests/`  
**Coverage:** Unit tests, integration tests, async tests

---

## Setup

```bash
cd backend
pip install -r requirements-dev.txt
```

**Dependencies:**
- `pytest` - Test framework
- `pytest-asyncio` - Async test support
- `pytest-cov` - Coverage reports
- `httpx` - HTTP testing

---

## Running Tests

### All Tests

```bash
pytest
```

### Specific Test File

```bash
pytest tests/test_intent.py
```

### With Coverage

```bash
pytest --cov=app --cov-report=html
```

### Quick Mode (Quiet)

```bash
pytest -q
```

### Verbose Mode

```bash
pytest -v
```

---

## Test Structure

```
backend/tests/
├── conftest.py              # Shared fixtures
├── test_intent_llm.py       # Intent extraction tests
├── test_search_native.py    # Search provider tests
├── test_verifier_native.py  # Verification adapter tests
├── test_rag_learning.py     # RAG learning tests
└── integration/             # Integration tests
```

---

## Writing Tests

### Async Tests

```python
import pytest

@pytest.mark.asyncio
async def test_extract_claims():
    from app.features.intent.adapters.llm import LLMIntentAdapter
    
    adapter = LLMIntentAdapter()
    result = await adapter.extract("The Earth is flat")
    
    assert len(result) >= 1
    assert "flat" in result[0].claim_text.lower()
```

### Using Fixtures

```python
# conftest.py
import pytest
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker

@pytest.fixture
async def db_session():
    """Provide a test database session"""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:")
    SessionLocal = async_sessionmaker(engine, expire_on_commit=False)
    
    async with SessionLocal() as session:
        yield session

# test_database.py
@pytest.mark.asyncio
async def test_create_claim(db_session):
    from app.features.verification.persistence.models import Claim
    
    claim = Claim(claim_text="Test claim")
    db_session.add(claim)
    await db_session.commit()
    
    assert claim.id is not None
```

### Mocking External APIs

```python
from unittest.mock import AsyncMock, patch

@pytest.mark.asyncio
async def test_tavily_search():
    with patch('app.features.search.providers.tavily.TavilySearchProvider.search') as mock_search:
        mock_search.return_value = [
            {"url": "https://example.com", "title": "Test", "snippet": "..."}
        ]
        
        provider = TavilySearchProvider(api_key="test")
        results = await provider.search("test query")
        
        assert len(results) == 1
        assert results[0].url == "https://example.com"
```

---

## Example Tests

### Test Intent Extraction

```python
# tests/test_intent.py
import pytest
from app.features.intent.adapters.llm import LLMIntentAdapter

@pytest.mark.asyncio
async def test_extract_single_claim():
    adapter = LLMIntentAdapter()
    result = await adapter.extract("The Earth is flat")
    
    assert len(result) == 1
    assert "Earth" in result[0].claim_text
    assert result[0].search_query is not None

@pytest.mark.asyncio
async def test_extract_multiple_claims():
    adapter = LLMIntentAdapter()
    result = await adapter.extract("Vaccines cause autism and 5G causes cancer")
    
    assert len(result) == 2
```

### Test Search Provider

```python
# tests/test_search.py
import pytest
from app.features.search.providers.tavily import TavilySearchProvider

@pytest.mark.asyncio
async def test_tavily_search_blocks_social_media():
    provider = TavilySearchProvider(api_key=os.getenv("TAVILY_API_KEY"))
    results = await provider.search("test query")
    
    for result in results:
        domain = extract_domain(result.url)
        assert domain not in ["facebook.com", "twitter.com", "reddit.com"]
```

### Test Verification

```python
# tests/test_verification.py
import pytest
from app.features.verification.adapters.openai_compatible import OpenAICompatibleClaimVerifier

@pytest.mark.asyncio
async def test_verify_with_strong_evidence():
    adapter = OpenAICompatibleClaimVerifier()
    
    claim = "The Earth is round"
    evidence = [
        {"snippet": "NASA confirms Earth is spherical", "score": 0.98},
        {"snippet": "Satellite images show curved horizon", "score": 0.95}
    ]
    
    result = await adapter.verify(claim, evidence)
    
    assert result.verdict in ["TRUE", "MOSTLY_TRUE"]
    assert result.confidence >= 0.80
```

---

## Continuous Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        cd backend
        pip install -r requirements-core.txt
        pip install -r requirements-dev.txt
    
    - name: Run tests
      run: |
        cd backend
        pytest --cov=app --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
```

---

## Best Practices

1. **Test async code with pytest-asyncio**
2. **Mock external API calls** (don't hit real APIs in tests)
3. **Use fixtures** for common setup (database, clients)
4. **Test edge cases** (empty input, malformed data, errors)
5. **Keep tests fast** (< 1s per test ideally)

---

See also:
- [Architecture: Backend](../03-architecture/backend.md)
- [Frontend Testing](frontend-tests.md)
- [Test Claims Benchmark](test-claims.md)
