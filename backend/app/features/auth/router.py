from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.deps import get_app_settings, get_db
from app.core.settings import Settings
from app.features.auth.service import AuthService
from app.contracts.auth import LoginRequest, LoginResponse

router = APIRouter()


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    session: AsyncSession = Depends(get_db),
    settings: Settings = Depends(get_app_settings),
) -> LoginResponse:
    """Login endpoint."""
    service = AuthService(session)
    try:
        user = await service.authenticate_user(request.email, request.password)
    except Exception as exc:
        if settings.db_required:
            raise
        raise HTTPException(status_code=503, detail="Database unavailable") from exc

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    return LoginResponse(user=user)