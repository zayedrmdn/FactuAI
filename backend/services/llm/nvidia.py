"""
Nvidia NIM LLM Provider

Cloud-based LLM provider using Nvidia's NIM API.
Supports Nvidia-hosted models like Llama, Mistral, etc.
"""

import os
from typing import Dict, Any, Optional, List

from services.llm.base import BaseLLM
from core.logging import logger
from core.exceptions import LLMClientError


class NvidiaLLM(BaseLLM):
    """
    Nvidia NIM LLM provider.
    
    Uses Nvidia's API (OpenAI-compatible) for inference.
    Commonly used for enterprise deployments.
    """
    
    DEFAULT_MODEL = "meta/llama-3.1-8b-instruct"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.7
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize Nvidia NIM provider.
        
        Args:
            api_key: Nvidia API key (defaults to NVIDIA_API_KEY env var)
            model: Model to use (defaults to NVIDIA_MODEL env var)
        """
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        self.model = model or os.getenv("NVIDIA_MODEL", self.DEFAULT_MODEL)
        self.base_url = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")
        self._client = None
        self._available = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client for Nvidia."""
        if not self.api_key:
            logger.warning("[NVIDIA] No API key provided")
            return
        
        try:
            from openai import OpenAI
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self._available = True
            logger.info(f"[NVIDIA] Initialized with model: {self.model}")
        except Exception as e:
            logger.error(f"[NVIDIA] Failed to initialize: {e}")
            self._available = False
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate a response using Nvidia NIM."""
        if not self._available:
            raise LLMClientError("Nvidia client not available")
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS,
                temperature=temperature or self.DEFAULT_TEMPERATURE,
                **kwargs
            )
            
            result = response.choices[0].message.content
            logger.debug(f"[NVIDIA] Generated {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"[NVIDIA] Generation failed: {e}")
            raise LLMClientError(f"Nvidia generation failed: {e}")
    
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate a response using chat format."""
        if not self._available:
            raise LLMClientError("Nvidia client not available")
        
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user}
                ],
                max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS,
                temperature=temperature or self.DEFAULT_TEMPERATURE,
                **kwargs
            )
            
            result = response.choices[0].message.content
            logger.debug(f"[NVIDIA] Chat generated {len(result)} chars")
            return result
            
        except Exception as e:
            logger.error(f"[NVIDIA] Chat failed: {e}")
            raise LLMClientError(f"Nvidia chat failed: {e}")
    
    def is_available(self) -> bool:
        """Check if Nvidia NIM is available."""
        return self._available
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get Nvidia model info."""
        return {
            "provider": "nvidia",
            "model_name": self.model,
            "base_url": self.base_url,
            "available": self._available,
        }
