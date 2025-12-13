import bcrypt
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logging import get_logger
from app.features.auth.models import User as UserModel
from app.contracts.auth import User

logger = get_logger(__name__)


class AuthService:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def authenticate_user(self, email: str, password: str) -> User | None:
        """Authenticate user by email and password."""
        stmt = select(UserModel).where(UserModel.email == email, UserModel.is_active == True)
        result = await self.session.execute(stmt)
        user_row = result.scalar_one_or_none()

        if not user_row:
            return None

        if not bcrypt.checkpw(password.encode('utf-8'), user_row.password_hash.encode('utf-8')):
            return None

        return User(
            id=user_row.id,
            email=user_row.email,
            username=user_row.username,
            is_active=user_row.is_active,
        )