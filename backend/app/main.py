# Full Path: backend/app/main.py
# Load environment variables FIRST (before any settings imports)
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi.errors import RateLimitExceeded

from app.core.deps import lifespan
from app.core.settings import get_settings
from app.core.rate_limit import limiter, rate_limit_exceeded_handler
from app.features.analyze.router import router as analyze_router
from app.features.auth.router import router as auth_router
from app.features.system.router import router as system_router

app = FastAPI(title="FactuAI API", version="1.0.0", lifespan=lifespan)

# Configure rate limiting
settings = get_settings()
if settings.rate_limit_enabled:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")
app.include_router(auth_router, prefix="/api")
app.include_router(system_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/health/detailed")
async def health_detailed(request: Request):
    """
    Detailed health check endpoint that reports infrastructure status.
    
    Returns status of Database, Redis, and Embedding Service.
    Useful for monitoring and debugging infrastructure connectivity.
    """
    from app.core.health import InfrastructureHealthChecker
    from app.core.db import get_sessionmaker
    
    settings = get_settings()
    redis = getattr(request.app.state, "redis", None)
    
    # Create a session for the health check
    session_maker = get_sessionmaker()
    async with session_maker() as db:
        checker = InfrastructureHealthChecker(settings=settings, db=db, redis=redis)
        report = await checker.check_all()
    
    return {
        "status": "healthy" if report.is_ready else "unhealthy",
        "services": {
            "database": {
                "status": report.database.status.value,
                "latency_ms": report.database.latency_ms,
                "error": report.database.error,
            },
            "redis": {
                "status": report.redis.status.value,
                "latency_ms": report.redis.latency_ms,
                "error": report.redis.error,
            },
            "embedding_service": {
                "status": report.embedding_service.status.value,
                "latency_ms": report.embedding_service.latency_ms,
                "error": report.embedding_service.error,
            },
        },
    }
