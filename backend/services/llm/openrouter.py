"""
OpenRouter LLM Provider

Cloud-based LLM provider using OpenRouter API.
Supports multiple models through a unified interface.
"""

import os
from typing import Dict, Any, Optional, List

from services.llm.base import BaseLLM
from core.logging import logger
from core.exceptions import LLMClientError


class OpenRouterLLM(BaseLLM):
    """
    OpenRouter LLM provider.
    
    Uses the OpenAI-compatible API provided by OpenRouter to access
    various LLM models (Claude, GPT, Llama, etc.)
    """
    
    DEFAULT_MODEL = "anthropic/claude-3-haiku"
    DEFAULT_MAX_TOKENS = 1024
    DEFAULT_TEMPERATURE = 0.7
    
    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        """
        Initialize OpenRouter provider.
        
        Args:
            api_key: OpenRouter API key (defaults to OPENROUTER_API_KEY env var)
            model: Model to use (defaults to OPENROUTER_MODEL env var or claude-3-haiku)
        """
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.model = model or os.getenv("OPENROUTER_MODEL", self.DEFAULT_MODEL)
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._client = None
        self._available = False
        
        self._initialize_client()
    
    def _initialize_client(self):
        """Initialize the OpenAI client for OpenRouter."""
        if not self.api_key:
            logger.warning("[OPENROUTER] No API key provided")
            return
        
        try:
            from openai import OpenAI
            from core.logging import log_model_init
            self._client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
            )
            self._available = True
            log_model_init(logger, "openrouter", self.model, "success")
        except Exception as e:
            from core.logging import log_model_init
            logger.error(f"[OPENROUTER] Failed to initialize: {e}")
            log_model_init(logger, "openrouter", self.model, "failed")
            self._available = False
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate a response using OpenRouter."""
        if not self._available:
            raise LLMClientError("OpenRouter client not available")
        
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
            
            # Robust null checks
            if not response:
                logger.error("[OPENROUTER] API returned None response")
                raise LLMClientError("OpenRouter returned empty response")
            
            if not hasattr(response, 'choices') or not response.choices:
                logger.error("[OPENROUTER] API response missing 'choices' field")
                raise LLMClientError("OpenRouter response malformed: no choices")
            
            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                logger.error("[OPENROUTER] First choice missing or has no message")
                raise LLMClientError("OpenRouter response malformed: no message")
            
            result = response.choices[0].message.content
            if result is None:
                logger.error("[OPENROUTER] Message content is None")
                raise LLMClientError("OpenRouter returned None content")
            
            if not result.strip():
                logger.warning("[OPENROUTER] Generated empty string")
            
            logger.debug(f"[OPENROUTER] Generated {len(result)} chars")
            return result
            
        except LLMClientError:
            raise
        except Exception as e:
            logger.error(f"[OPENROUTER] Generation failed: {e}")
            raise LLMClientError(f"OpenRouter generation failed: {e}")
    
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
            raise LLMClientError("OpenRouter client not available")
        
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
            
            # Robust null checks
            if not response or not hasattr(response, 'choices') or not response.choices:
                logger.error("[OPENROUTER] Chat API returned malformed response")
                raise LLMClientError("OpenRouter chat response malformed")
            
            if not response.choices[0] or not hasattr(response.choices[0], 'message'):
                logger.error("[OPENROUTER] Chat response missing message")
                raise LLMClientError("OpenRouter chat response has no message")
            
            result = response.choices[0].message.content
            if result is None:
                logger.error("[OPENROUTER] Chat message content is None")
                raise LLMClientError("OpenRouter chat returned None content")
            
            logger.debug(f"[OPENROUTER] Chat generated {len(result)} chars")
            return result
            
        except LLMClientError:
            raise
        except Exception as e:
            logger.error(f"[OPENROUTER] Chat failed: {e}")
            raise LLMClientError(f"OpenRouter chat failed: {e}")
    
    def is_available(self) -> bool:
        """Check if OpenRouter is available."""
        return self._available
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get OpenRouter model info."""
        return {
            "provider": "openrouter",
            "model_name": self.model,
            "base_url": self.base_url,
            "available": self._available,
        }
