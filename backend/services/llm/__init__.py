"""
FactuAI LLM Service Module

Provides a modular LLM system with multiple provider support:
- OpenRouter (cloud)
- Nvidia NIM (cloud)
- Local Unsloth/Qwen (local)

Usage:
    from services.llm import LLMFactory
    
    llm = LLMFactory.create()  # Uses LLM_PROVIDER from .env
    response = llm.generate_response("Hello!")
"""

from services.llm.factory import LLMFactory
from services.llm.base import BaseLLM

__all__ = ["LLMFactory", "BaseLLM"]
