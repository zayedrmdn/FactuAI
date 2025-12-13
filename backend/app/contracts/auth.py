# Auth feature contracts (shared types)
from pydantic import BaseModel


class User(BaseModel):
    id: int
    email: str
    username: str | None
    is_active: bool


class LoginRequest(BaseModel):
    email: str
    password: str


class LoginResponse(BaseModel):
    user: User
    message: str = "Login successful"