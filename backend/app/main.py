from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.deps import lifespan
from app.features.analyze.router import router as analyze_router
from app.features.auth.router import router as auth_router

app = FastAPI(title="FactuAI API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(analyze_router, prefix="/api")
app.include_router(auth_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}
