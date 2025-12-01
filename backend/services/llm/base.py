"""
BaseLLM - Abstract base class for all LLM providers.

All LLM implementations must inherit from this class and implement
the required methods. This ensures consistent interface across
OpenRouter, Nvidia, and Local providers.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List


class BaseLLM(ABC):
    """
    Abstract base class for LLM providers.
    
    All providers must implement:
    - generate_response: Generate text from a prompt
    - chat: Chat-style generation with system/user messages
    - is_available: Check if the provider is ready
    - get_model_info: Get metadata about the model
    """
    
    @abstractmethod
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the LLM based on the provided prompt.
        
        Args:
            prompt: Input prompt for the LLM
            max_tokens: Maximum number of tokens to generate (default: provider-specific)
            temperature: Sampling temperature (default: provider-specific)
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated response text
            
        Raises:
            LLMClientError: If generation fails
        """
        pass
    
    @abstractmethod
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """
        Generate a response using chat-style messages.
        
        Args:
            system: System message setting the context
            user: User message/query
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional parameters
            
        Returns:
            Generated response text
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if the LLM provider is available and ready for use.
        
        Returns:
            True if available, False otherwise
        """
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the current model/provider.
        
        Returns:
            Dictionary containing:
            - provider: Provider name (openrouter, nvidia, local)
            - model_name: Model identifier
            - available: Whether the provider is ready
            - Additional provider-specific info
        """
        pass
    
    def validate_content(self, text: str) -> Dict[str, Any]:
        """
        Validate content before processing.
        Default implementation - can be overridden by providers.
        
        Args:
            text: Text to validate
            
        Returns:
            Dictionary with isValid, error, suggestion keys
        """
        if len(text.strip()) < 10:
            return {
                "isValid": False,
                "error": "Text too short",
                "suggestion": "Please provide more detailed text."
            }
        return {
            "isValid": True,
            "error": "",
            "suggestion": ""
        }
    
    def clear_cache(self) -> None:
        """
        Clear any cached state. Default implementation does nothing.
        Override in providers that maintain cache.
        """
        pass
    
    def get_embedding(self, text: str) -> Optional[List[float]]:
        """
        Get embedding vector for text. Optional - not all providers support this.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector or None if not supported
        """
        return None
