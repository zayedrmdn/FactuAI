from app.core.logging import get_logger

logger = get_logger(__name__)


async def enqueue_learning_job(verification_id: int) -> None:
    """Placeholder for background learning job (vectorization + upsert)."""
    logger.info(f"[LEARN] Enqueue learning job for verification_id={verification_id}")
