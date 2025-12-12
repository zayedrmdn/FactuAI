import anyio

from app.core.settings import Settings
from app.features.verification.adapters.openai_compatible import OpenAICompatibleClaimVerifier


def test_verifier_returns_unverifiable_without_key():
    settings = Settings(llm_api_key="")
    verifier = OpenAICompatibleClaimVerifier(settings=settings)

    async def run():
        verdict = await verifier.verify_claim(
            claim="The Earth is flat",
            evidence=[
                {
                    "text": "Scientists agree the Earth is roughly spherical.",
                    "url": "https://example.com",
                    "title": "Example",
                    "source_domain": "web",
                    "score": 0.9,
                }
            ],
            provider="openrouter",
            model="test-model",
        )
        assert verdict["verdict"] == "unverifiable"
        assert verdict["confidence"] == 0.0

    anyio.run(run)
