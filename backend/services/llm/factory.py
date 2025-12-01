"""
LLM Factory

Creates the appropriate LLM provider based on configuration.
Supports: openrouter, nvidia, local
"""

import os
from typing import Optional

from services.llm.base import BaseLLM
from core.logging import logger


class LLMFactory:
    """
    Factory for creating LLM provider instances.
    
    The provider is determined by the LLM_PROVIDER environment variable:
    - "openrouter": Use OpenRouter API (default for cloud mode)
    - "nvidia": Use Nvidia NIM API
    - "local": Use local Unsloth model (requires GPU)
    
    If LLM_PROVIDER is not set, it defaults based on APP_RUN_MODE:
    - cloud mode -> openrouter
    - local mode -> local
    """
    
    # Supported providers
    PROVIDERS = ["openrouter", "nvidia", "local"]
    
    @classmethod
    def create(cls, provider: Optional[str] = None, **kwargs) -> BaseLLM:
        """
        Create an LLM provider instance.
        
        Args:
            provider: Provider name (openrouter, nvidia, local)
                     If None, uses LLM_PROVIDER env var or defaults based on APP_RUN_MODE
            **kwargs: Additional arguments passed to the provider constructor
            
        Returns:
            BaseLLM instance
            
        Raises:
            ValueError: If provider is not supported
        """
        # Determine provider
        if provider is None:
            provider = os.getenv("LLM_PROVIDER")
        
        if provider is None:
            # Default based on run mode
            run_mode = os.getenv("APP_RUN_MODE", "cloud").lower()
            provider = "local" if run_mode == "local" else "openrouter"
        
        provider = provider.lower()
        
        if provider not in cls.PROVIDERS:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported providers: {', '.join(cls.PROVIDERS)}"
            )
        
        logger.info(f"[LLM_FACTORY] Creating provider: {provider}")
        
        # Create the appropriate provider
        if provider == "openrouter":
            from services.llm.openrouter import OpenRouterLLM
            return OpenRouterLLM(**kwargs)
        
        elif provider == "nvidia":
            from services.llm.nvidia import NvidiaLLM
            return NvidiaLLM(**kwargs)
        
        elif provider == "local":
            from services.llm.local import LocalLLM
            return LocalLLM(**kwargs)
        
        # Should never reach here due to validation above
        raise ValueError(f"Unknown provider: {provider}")
    
    @classmethod
    def get_available_providers(cls) -> list:
        """
        Get list of available (configured) providers.
        
        Returns:
            List of provider names that have required configuration
        """
        available = []
        
        # Check OpenRouter
        if os.getenv("OPENROUTER_API_KEY"):
            available.append("openrouter")
        
        # Check Nvidia
        if os.getenv("NVIDIA_API_KEY"):
            available.append("nvidia")
        
        # Check local (always potentially available if torch is installed)
        try:
            import torch
            available.append("local")
        except ImportError:
            pass
        
        return available
    
    @classmethod
    def get_default_provider(cls) -> str:
        """
        Get the default provider based on configuration.
        
        Returns:
            Provider name
        """
        # Check explicit setting
        provider = os.getenv("LLM_PROVIDER")
        if provider and provider.lower() in cls.PROVIDERS:
            return provider.lower()
        
        # Default based on run mode
        run_mode = os.getenv("APP_RUN_MODE", "cloud").lower()
        return "local" if run_mode == "local" else "openrouter"
