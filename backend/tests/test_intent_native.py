import anyio

from app.features.intent.adapters.native import NativeIntentAdapter


def test_native_intent_extracts_claims():
    adapter = NativeIntentAdapter()

    async def run():
        items = await adapter.parse_and_route(
            text="- The sky is blue\n- Water is wet",
            max_claims=5,
            provider="openrouter",
            model="test",
        )
        assert len(items) == 2
        assert items[0]["claim_text"]
        assert items[0]["search_query"]
        assert items[0]["verification_question"]

    anyio.run(run)
