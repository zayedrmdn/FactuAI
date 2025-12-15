import anyio
from unittest.mock import AsyncMock, MagicMock, patch

from app.core.settings import Settings
from app.features.search.adapters.native import NativeSearchService


def test_native_search_returns_empty_without_keys():
    """External search returns empty when no API keys are configured."""
    settings = Settings(
        tavily_api_key="",
        newsapi_api_key="",
        search_provider_paths_csv=(
            "app.features.search.providers.tavily.TavilySearchProvider"
        ),
    )
    search = NativeSearchService(settings=settings, redis=None)

    async def run():
        results = await search.hybrid_search(query="test query", max_results=3)
        assert isinstance(results, list)
        assert results == []

    anyio.run(run)


def test_rag_internal_search_hit():
    """
    RAG Hit Test: When DB contains a claim with embedding close to query,
    and distance is below threshold, the result should be included.
    """
    settings = Settings(
        tavily_api_key="",
        newsapi_api_key="",
        search_provider_paths_csv="",  # No external providers
        embedding_api_base_url="http://fake-embedding-service",
        embedding_api_key="fake-key",
        embedding_model="test-model",
        rag_retrieval_threshold=0.20,  # Strict threshold (0.80 similarity)
    )
    search = NativeSearchService(settings=settings, redis=None)

    # Mock embedding response
    mock_embedding = [0.1] * 384

    # Mock DB row: distance is BELOW threshold (0.15 < 0.20) -> HIT
    mock_row = MagicMock()
    mock_row.claim_text = "Mars is called the Red Planet"
    mock_row.reasoning = "Mars has iron oxide on its surface giving it a red appearance."
    mock_row.distance = 0.15  # Close match (similarity = 0.85)

    async def run():
        with patch("httpx.AsyncClient") as mock_http, \
             patch("app.features.search.adapters.native.AsyncOpenAI") as mock_oai, \
             patch("app.features.search.adapters.native.get_sessionmaker") as mock_session:

            # Mock health check
            mock_http_instance = AsyncMock()
            mock_http_instance.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
            mock_http_instance.__aexit__ = AsyncMock(return_value=None)
            mock_http.return_value = mock_http_instance

            # Mock embedding generation
            mock_oai_instance = MagicMock()
            mock_embed_resp = MagicMock()
            mock_embed_resp.data = [MagicMock(embedding=mock_embedding)]
            mock_oai_instance.embeddings.create = AsyncMock(return_value=mock_embed_resp)
            mock_oai.return_value = mock_oai_instance

            # Mock DB session
            mock_session_ctx = MagicMock()
            mock_async_session = AsyncMock()

            # Claims query returns a match
            mock_claims_result = MagicMock()
            mock_claims_result.fetchall.return_value = [mock_row]

            # Evidence query returns nothing
            mock_evidence_result = MagicMock()
            mock_evidence_result.fetchall.return_value = []

            mock_async_session.execute = AsyncMock(side_effect=[mock_claims_result, mock_evidence_result])
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_async_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = MagicMock(return_value=mock_session_ctx)

            results = await search._search_internal(query="Red Planet", max_results=5)

            assert len(results) == 1
            assert "[INTERNAL MEMORY]" in results[0]["title"]
            assert "Mars" in results[0]["title"] or "Mars" in results[0]["text"]
            assert results[0]["score"] == 0.85  # 1.0 - 0.15 distance

    anyio.run(run)


def test_rag_internal_search_miss():
    """
    RAG Miss Test: When DB contains a claim but distance is ABOVE threshold,
    the result should NOT be included (correctly filtering noise).
    """
    settings = Settings(
        tavily_api_key="",
        newsapi_api_key="",
        search_provider_paths_csv="",  # No external providers
        embedding_api_base_url="http://fake-embedding-service",
        embedding_api_key="fake-key",
        embedding_model="test-model",
        rag_retrieval_threshold=0.20,  # Strict threshold (0.80 similarity)
    )
    search = NativeSearchService(settings=settings, redis=None)

    mock_embedding = [0.1] * 384

    async def run():
        with patch("httpx.AsyncClient") as mock_http, \
             patch("app.features.search.adapters.native.AsyncOpenAI") as mock_oai, \
             patch("app.features.search.adapters.native.get_sessionmaker") as mock_session:

            # Mock health check
            mock_http_instance = AsyncMock()
            mock_http_instance.get = AsyncMock(return_value=MagicMock(status_code=200))
            mock_http_instance.__aenter__ = AsyncMock(return_value=mock_http_instance)
            mock_http_instance.__aexit__ = AsyncMock(return_value=None)
            mock_http.return_value = mock_http_instance

            # Mock embedding generation
            mock_oai_instance = MagicMock()
            mock_embed_resp = MagicMock()
            mock_embed_resp.data = [MagicMock(embedding=mock_embedding)]
            mock_oai_instance.embeddings.create = AsyncMock(return_value=mock_embed_resp)
            mock_oai.return_value = mock_oai_instance

            # Mock DB session - returns EMPTY because WHERE clause filters out
            # results with distance >= threshold (SQL does the filtering)
            mock_session_ctx = MagicMock()
            mock_async_session = AsyncMock()

            mock_claims_result = MagicMock()
            mock_claims_result.fetchall.return_value = []  # Nothing matched threshold

            mock_evidence_result = MagicMock()
            mock_evidence_result.fetchall.return_value = []

            mock_async_session.execute = AsyncMock(side_effect=[mock_claims_result, mock_evidence_result])
            mock_session_ctx.__aenter__ = AsyncMock(return_value=mock_async_session)
            mock_session_ctx.__aexit__ = AsyncMock(return_value=None)
            mock_session.return_value = MagicMock(return_value=mock_session_ctx)

            results = await search._search_internal(query="Apples", max_results=5)

            # The DB returned nothing because WHERE filtered out all results
            assert len(results) == 0

    anyio.run(run)
