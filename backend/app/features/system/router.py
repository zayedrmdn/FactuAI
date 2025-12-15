# Full Path: backend/app/features/system/router.py
"""
System Configuration API

Exposes active backend settings to the Frontend so it can dynamically
configure UI elements (model selectors, feature flags) without hardcoding.
"""

from fastapi import APIRouter
from pydantic import BaseModel

from app.core.settings import get_settings


router = APIRouter(prefix="/system", tags=["system"])


class ModelsConfig(BaseModel):
    """Active model configuration from backend settings."""
    default_reasoning: str
    default_intent: str
    provider: str
    api_base_url: str


class FeaturesConfig(BaseModel):
    """Feature flags derived from backend settings."""
    tavily_enabled: bool
    learning_enabled: bool
    rate_limit_enabled: bool
    preflight_checks_enabled: bool


class SystemConfigResponse(BaseModel):
    """Full system configuration response."""
    models: ModelsConfig
    features: FeaturesConfig


@router.get("/config", response_model=SystemConfigResponse)
async def get_system_config() -> SystemConfigResponse:
    """
    Get active system configuration.
    
    This endpoint exposes backend settings to the Frontend, allowing
    dynamic UI configuration without hardcoded values.
    
    Returns:
        SystemConfigResponse: Active models and feature flags.
    """
    settings = get_settings()
    
    return SystemConfigResponse(
        models=ModelsConfig(
            default_reasoning=settings.openrouter_model,
            default_intent=settings.intent_llm_model,
            provider=settings.llm_provider,
            api_base_url=settings.llm_api_base_url,
        ),
        features=FeaturesConfig(
            tavily_enabled=bool(settings.tavily_api_key),
            learning_enabled=bool(settings.embedding_api_base_url),
            rate_limit_enabled=settings.rate_limit_enabled,
            preflight_checks_enabled=settings.preflight_checks_enabled,
        ),
    )
