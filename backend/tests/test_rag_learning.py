import anyio

from app.core.settings import Settings
from app.features.verification.learning import RagLearningService


def test_rag_learning_skips_without_embeddings_key():
    settings = Settings(embedding_api_key="", llm_api_key="")
    learner = RagLearningService(settings=settings)

    async def run():
        # Should not raise even without DB, because it exits early without credentials.
        await learner.learn_from_verification(verification_id=123)

    anyio.run(run)
