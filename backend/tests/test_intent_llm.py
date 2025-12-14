# Full Path: backend/tests/test_intent_llm.py
"""
Unit tests for LLMIntentAdapter.

These tests use mocked LLM responses to verify the adapter's behavior
without making actual API calls.
"""
import anyio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.features.intent.adapters.llm import LLMIntentAdapter, _ClaimListOutput, _ClaimOutput


@pytest.fixture
def mock_settings():
    """Create mock settings with intent LLM configuration."""
    settings = MagicMock()
    settings.intent_llm_api_base_url = "https://api.example.com/v1"
    settings.intent_llm_api_key = "test-api-key"
    settings.intent_llm_model = "test-model"
    settings.llm_api_base_url = "https://fallback.example.com/v1"
    settings.llm_api_key = "fallback-key"
    return settings


@pytest.fixture
def mock_settings_no_intent_config():
    """Create mock settings without intent-specific config (falls back to main)."""
    settings = MagicMock()
    settings.intent_llm_api_base_url = ""
    settings.intent_llm_api_key = ""
    settings.intent_llm_model = ""
    settings.llm_api_base_url = "https://main.example.com/v1"
    settings.llm_api_key = "main-key"
    return settings


@pytest.fixture
def mock_settings_no_key():
    """Create mock settings with no API key."""
    settings = MagicMock()
    settings.intent_llm_api_base_url = ""
    settings.intent_llm_api_key = ""
    settings.intent_llm_model = ""
    settings.llm_api_base_url = ""
    settings.llm_api_key = ""
    return settings


def test_llm_intent_adapter_extracts_claims(mock_settings):
    """Test that the adapter correctly extracts claims from LLM response."""
    adapter = LLMIntentAdapter(settings=mock_settings)

    # Mock the LLM chain response
    mock_response = _ClaimListOutput(
        claims=[
            _ClaimOutput(
                claim_text="The Eiffel Tower is 330 meters tall.",
                search_query="Eiffel Tower height meters",
                verification_question="Is the Eiffel Tower 330 meters tall?",
            ),
            _ClaimOutput(
                claim_text="Apple was founded in 1976.",
                search_query="Apple company founded year",
                verification_question="Was Apple founded in 1976?",
            ),
        ]
    )

    async def run():
        with patch.object(adapter, "_extract_claims", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = [
                {
                    "claim_text": "The Eiffel Tower is 330 meters tall.",
                    "search_query": "Eiffel Tower height meters",
                    "verification_question": "Is the Eiffel Tower 330 meters tall?",
                },
                {
                    "claim_text": "Apple was founded in 1976.",
                    "search_query": "Apple company founded year",
                    "verification_question": "Was Apple founded in 1976?",
                },
            ]

            items = await adapter.parse_and_route(
                text="The Eiffel Tower is 330 meters tall. Apple was founded in 1976.",
                max_claims=5,
                provider="openrouter",
                model="test-model",
            )

            assert len(items) == 2
            assert items[0]["claim_text"] == "The Eiffel Tower is 330 meters tall."
            assert items[0]["search_query"] == "Eiffel Tower height meters"
            assert items[1]["claim_text"] == "Apple was founded in 1976."

    anyio.run(run)


def test_llm_intent_adapter_handles_empty_input(mock_settings):
    """Test that empty input returns empty list without calling LLM."""
    adapter = LLMIntentAdapter(settings=mock_settings)

    async def run():
        items = await adapter.parse_and_route(
            text="",
            max_claims=5,
            provider="openrouter",
            model="test-model",
        )
        assert items == []

        items = await adapter.parse_and_route(
            text="   ",
            max_claims=5,
            provider="openrouter",
            model="test-model",
        )
        assert items == []

    anyio.run(run)


def test_llm_intent_adapter_handles_no_api_key(mock_settings_no_key):
    """Test graceful handling when no API key is configured."""
    adapter = LLMIntentAdapter(settings=mock_settings_no_key)

    async def run():
        items = await adapter.parse_and_route(
            text="The sky is blue.",
            max_claims=5,
            provider="openrouter",
            model="test-model",
        )
        assert items == []

    anyio.run(run)


def test_llm_intent_adapter_falls_back_to_main_config(mock_settings_no_intent_config):
    """Test that adapter falls back to main LLM config when intent-specific not set."""
    adapter = LLMIntentAdapter(settings=mock_settings_no_intent_config)

    async def run():
        with patch.object(adapter, "_extract_claims", new_callable=AsyncMock) as mock_extract:
            mock_extract.return_value = []

            await adapter.parse_and_route(
                text="Test claim.",
                max_claims=5,
                provider="openrouter",
                model="fallback-model",
            )

            # Verify _extract_claims was called with fallback config
            mock_extract.assert_called_once()
            call_kwargs = mock_extract.call_args.kwargs
            assert call_kwargs["api_key"] == "main-key"
            assert call_kwargs["api_base"] == "https://main.example.com/v1"
            assert call_kwargs["model"] == "fallback-model"

    anyio.run(run)


def test_llm_intent_adapter_respects_max_claims(mock_settings):
    """Test that max_claims limit is passed to _extract_claims."""
    adapter = LLMIntentAdapter(settings=mock_settings)

    async def run():
        with patch.object(adapter, "_extract_claims", new_callable=AsyncMock) as mock_extract:
            # Mock returns exactly 3 claims (as would be limited by max_claims in _extract_claims)
            mock_extract.return_value = [
                {"claim_text": f"Claim {i}", "search_query": f"query {i}", "verification_question": None}
                for i in range(3)
            ]

            items = await adapter.parse_and_route(
                text="Many claims here.",
                max_claims=3,
                provider="openrouter",
                model="test-model",
            )

            # Verify max_claims was passed to _extract_claims
            call_kwargs = mock_extract.call_args.kwargs
            assert call_kwargs["max_claims"] == 3

            # And the result should have at most 3 items
            assert len(items) == 3

    anyio.run(run)


def test_llm_intent_adapter_handles_llm_error(mock_settings):
    """Test graceful degradation when LLM call fails."""
    adapter = LLMIntentAdapter(settings=mock_settings)

    async def run():
        with patch.object(adapter, "_extract_claims", new_callable=AsyncMock) as mock_extract:
            mock_extract.side_effect = Exception("LLM API error")

            items = await adapter.parse_and_route(
                text="Some text to analyze.",
                max_claims=5,
                provider="openrouter",
                model="test-model",
            )

            # Should return empty list on error, not raise
            assert items == []

    anyio.run(run)
