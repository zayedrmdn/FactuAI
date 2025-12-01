"""
Centralized Configuration for FactuAI Backend

This module provides the Config class which serves as the single source of truth
for all application configuration. It supports dual run modes:
- 'cloud': Lightweight mode using external APIs (no local ML models)
- 'local': Full mode with local PyTorch/CUDA models

Configuration is loaded from environment variables with sensible defaults.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    """
    Application configuration class.
    
    Loads configuration from environment variables and provides
    computed properties based on the APP_RUN_MODE setting.
    """
    
    # ==========================================================================
    # Run Mode Configuration
    # ==========================================================================
    # APP_RUN_MODE determines whether to use local ML models or cloud APIs
    # Values: 'cloud' (default) | 'local'
    APP_RUN_MODE = os.getenv("APP_RUN_MODE", "cloud").lower()
    
    # Computed flags based on run mode
    # These determine which service implementations to use
    USE_LOCAL_LLM = APP_RUN_MODE == "local"
    USE_LOCAL_CLASSIFIER = APP_RUN_MODE == "local"
    
    # ==========================================================================
    # Database Configuration
    # ==========================================================================
    SQLALCHEMY_DATABASE_URI = os.getenv("DB_URI")
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # ==========================================================================
    # Email Configuration
    # ==========================================================================
    RESEND_API_KEY = os.getenv("RESEND_API_KEY")
    FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@your-domain.com")
    
    # ==========================================================================
    # Security Configuration
    # ==========================================================================
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")
    
    # ==========================================================================
    # Frontend Configuration
    # ==========================================================================
    FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")
    
    # ==========================================================================
    # API Keys (for Cloud Mode)
    # ==========================================================================
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
    OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
    OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
    NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
    
    # LLM Provider selection: openrouter, nvidia, or local
    # If not set, defaults based on APP_RUN_MODE
    LLM_PROVIDER = os.getenv("LLM_PROVIDER")
    
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")
    NEWS_API_KEY = os.getenv("NEWS_API_KEY")
    
    @classmethod
    def is_cloud_mode(cls) -> bool:
        """Check if application is running in cloud mode."""
        return cls.APP_RUN_MODE == "cloud"
    
    @classmethod
    def is_local_mode(cls) -> bool:
        """Check if application is running in local mode."""
        return cls.APP_RUN_MODE == "local"
    
    @classmethod
    def get_mode_info(cls) -> dict:
        """Get information about the current run mode configuration."""
        return {
            "run_mode": cls.APP_RUN_MODE,
            "use_local_llm": cls.USE_LOCAL_LLM,
            "use_local_classifier": cls.USE_LOCAL_CLASSIFIER,
        }
