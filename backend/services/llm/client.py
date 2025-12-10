"""
LLM Client for FactuAI

Simplified LLM interface supporting multiple providers:
- OpenRouter (cloud)
- Nvidia NIM (cloud)
- Local (Unsloth/Qwen)

Uses simple functions instead of class hierarchies.
"""

import os
from typing import Optional, Dict, Any
from utils.logging import get_logger
import time
from utils.helpers import LLMClientError

logger = get_logger(__name__)

# Global state for initialized clients
_clients = {}
_initialized = False


# ==========================================================================
# Provider Initialization
# ==========================================================================

def _init_openrouter():
    """Initialize OpenRouter client."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("[OPENROUTER] No API key provided")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            api_key=api_key,
        )
        logger.info("[OPENROUTER] Initialized (model selection from frontend)")
        return {"client": client, "model": None}  # Model comes from frontend
    except Exception as e:
        logger.error(f"[OPENROUTER] Failed to initialize: {e}")
        return None


def _init_nvidia():
    """Initialize Nvidia NIM client."""
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key:
        logger.warning("[NVIDIA] No API key provided")
        return None
    
    try:
        from openai import OpenAI
        client = OpenAI(
            base_url=os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
            api_key=api_key,
        )
        logger.info("[NVIDIA] Initialized (model selection from frontend)")
        return {"client": client, "model": None}  # Model comes from frontend
    except Exception as e:
        logger.error(f"[NVIDIA] Failed to initialize: {e}")
        return None


def _init_local():
    """Initialize local Unsloth/Qwen model."""
    try:
        import torch
        from unsloth import FastLanguageModel
        
        model_name = os.getenv("QWEN_MODEL", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            logger.info(f"[LOCAL] GPU memory: {props.total_memory / 1e9:.1f} GB")
        
        logger.info(f"[LOCAL] Loading model: {model_name}")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            dtype=dtype,
            load_in_4bit=True,
            device_map="auto",
        )
        FastLanguageModel.for_inference(model)
        logger.info(f"[LOCAL] Model loaded successfully on {device}")
        
        return {
            "model": model,
            "tokenizer": tokenizer,
            "device": device,
            "dtype": dtype
        }
    except ImportError:
        logger.warning("[LOCAL] PyTorch/Unsloth not installed")
        return None
    except Exception as e:
        logger.error(f"[LOCAL] Model load failed: {e}")
        return None


def initialize():
    """Initialize all available LLM providers."""
    global _clients, _initialized
    
    if _initialized:
        return _clients
    
    run_mode = os.getenv("APP_RUN_MODE", "cloud")
    
    # Always try to initialize cloud providers
    _clients["openrouter"] = _init_openrouter()
    _clients["nvidia"] = _init_nvidia()
    
    # Only initialize local in local mode
    if run_mode == "local":
        _clients["local"] = _init_local()
    
    # Log available providers
    available = [k for k, v in _clients.items() if v is not None]
    logger.info(f"[LLM] Available providers: {', '.join(available) or 'none'}")
    
    _initialized = True
    return _clients


def get_provider(provider: str = None) -> Optional[str]:
    """
    Get the best available provider.
    
    Args:
        provider: Requested provider name (openrouter, nvidia, local)
                 If None, returns first available provider
    
    Returns:
        Provider name or None if no providers available
    """
    if not _initialized:
        initialize()
    
    if provider and provider in _clients and _clients[provider]:
        return provider
    
    # Fallback: return first available provider
    for name, client in _clients.items():
        if client:
            return name
    
    return None


# ==========================================================================
# Generation Functions
# ==========================================================================

def chat(
    system: str,
    user: str,
    provider: str = None,
    model_id: str = None,
    max_tokens: int = 1024,
    temperature: float = 0.7,
    **kwargs
) -> str:
    """
    Generate a response with system and user messages.
    
    Args:
        system: System prompt (role/instructions)
        user: User message
        provider: Provider to use (openrouter, nvidia, local)
        model_id: Specific model ID to use
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature (0-1)
        **kwargs: Additional provider-specific parameters
    
    Returns:
        Generated text
        
    Raises:
        LLMClientError: If generation fails
    """
    provider = get_provider(provider)
    if not provider:
        raise LLMClientError("No LLM providers available")
    
    logger.debug(f"[LLM] Chat with {provider}")
    
    # Handle cloud providers
    if provider in ["openrouter", "nvidia"]:
        return _chat_cloud(system, user, provider, model_id, max_tokens, temperature, **kwargs)
    
    # Handle local provider
    elif provider == "local":
        return _chat_local(system, user, max_tokens, temperature, **kwargs)
    
    raise LLMClientError(f"Unknown provider: {provider}")


# ==========================================================================
# Provider-Specific Implementation
# ==========================================================================

def _chat_cloud(
    system: str,
    user: str,
    provider: str,
    model_id: str,
    max_tokens: int,
    temperature: float,
    **kwargs
) -> str:
    """Chat using cloud provider (OpenRouter or Nvidia)."""
    client_data = _clients[provider]
    if not client_data:
        raise LLMClientError(f"{provider} not available")
    
    if not model_id:
        raise LLMClientError(f"{provider} requires model_id parameter (no default model configured)")
    
    selected_model = model_id
    
    try:
        start_time = time.time()
        logger.debug(f"[{provider.upper()}] Sending request to {selected_model}...")
        
        # Always try with standard system+user message format first
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        response = client_data["client"].chat.completions.create(
            model=selected_model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs
        )
        
        elapsed = time.time() - start_time
        
        if not response or not response.choices:
            logger.error(f"[{provider.upper()}] No response or choices from {selected_model}")
            raise LLMClientError(f"{provider} returned empty response")
        
        choice = response.choices[0]
        message = choice.message
        finish_reason = choice.finish_reason
        
        # Handle reasoning models that output to reasoning field instead of content
        result = message.content or ""
        
        if not result and hasattr(message, 'reasoning') and message.reasoning:
            result = message.reasoning.strip()
            logger.debug(f"[{provider.upper()}] Extracted from reasoning field ({len(result)} chars)")
        elif not result and hasattr(message, 'reasoning_details') and message.reasoning_details:
            # Fallback: try to extract from reasoning_details
            for detail in message.reasoning_details:
                if isinstance(detail, dict) and 'text' in detail and detail['text']:
                    result = detail['text'].strip()
                    logger.debug(f"[{provider.upper()}] Extracted from reasoning_details ({len(result)} chars)")
                    break
        
        # Properly clean the result - strip whitespace and special chars
        result = result.strip() if result else ""
        
        # Log finish reason issues
        if finish_reason == 'length':
            logger.warning(f"[{provider.upper()}] Response truncated (finish_reason=length). Consider increasing max_tokens (current: {max_tokens})")
        
        logger.info(f"[{provider.upper()}] Generated {len(result)} chars using {selected_model} in {elapsed:.2f}s (finish_reason={finish_reason})")
        
        # Enhanced validation - check for minimal/invalid responses
        if len(result) == 0:
            logger.error(f"[{provider.upper()}] Empty response from {selected_model}. Message content: {repr(message.content)}, finish_reason: {finish_reason}")
            raise LLMClientError(f"{provider} returned empty response from {selected_model}")
        
        # Check for suspiciously short responses (likely errors)
        if len(result) < 10 and finish_reason == 'stop':
            logger.warning(f"[{provider.upper()}] Suspiciously short response ({len(result)} chars): {repr(result)}. This may indicate a prompt formatting issue.")
            # Don't fail, but warn - the pipeline will handle it
        
        return result
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check if error is about system messages not being supported
        if ("developer instruction" in error_msg or 
            "system message" in error_msg or
            "system prompt" in error_msg):
            
            logger.warning(f"[{provider.upper()}] Model {selected_model} doesn't support system messages, retrying with merged prompt...")
            
            try:
                # Retry with merged system+user message
                combined_message = f"{system}\n\n{user}"
                messages = [{"role": "user", "content": combined_message}]
                
                response = client_data["client"].chat.completions.create(
                    model=selected_model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    **kwargs
                )
                
                elapsed = time.time() - start_time
                
                if not response or not response.choices:
                    raise LLMClientError(f"{provider} returned empty response")
                
                choice = response.choices[0]
                message = choice.message
                result = message.content or ""
                
                logger.info(f"[{provider.upper()}] Retry successful: {len(result)} chars in {elapsed:.2f}s")
                return result
                
            except Exception as retry_error:
                logger.error(f"[{provider.upper()}] Retry also failed: {retry_error}")
                raise LLMClientError(f"{provider} chat failed after retry: {retry_error}")
        
        # For other errors, just raise
        logger.error(f"[{provider.upper()}] Chat failed: {e}")
        raise LLMClientError(f"{provider} chat failed: {e}")


def _chat_local(system: str, user: str, max_tokens: int, temperature: float, **kwargs) -> str:
    """Chat using local Unsloth model."""
    client_data = _clients.get("local")
    if not client_data:
        # Fallback for missing local model
        logger.warning("[LOCAL] Model not available, returning truncated prompt")
        return user[:100] + "..." if len(user) > 100 else user
    
    try:
        model = client_data["model"]
        tokenizer = client_data["tokenizer"]
        
        # Build chat messages for Qwen template
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ]
        
        # Apply chat template
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False
            )
        else:
            # Fallback manual Qwen template
            prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
        
        # Tokenize and move to device
        inputs = tokenizer([prompt], return_tensors="pt")
        device = client_data["device"]
        for k in inputs:
            inputs[k] = inputs[k].to(device)
        
        # Generate
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=kwargs.get("top_p", 0.8),
            top_k=kwargs.get("top_k", 20),
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            repetition_penalty=kwargs.get("repetition_penalty", 1.1)
        )
        
        # Decode only new tokens
        input_length = inputs['input_ids'].shape[1]
        new_tokens = outputs[0][input_length:]
        result = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        
        if not result:
            logger.warning("[LOCAL] Model returned empty response")
            return "I apologize, but I couldn't generate a response."
        
        logger.debug(f"[LOCAL] Generated {len(result)} chars")
        return result
        
    except Exception as e:
        logger.error(f"[LOCAL] Generation failed: {e}")
        raise LLMClientError(f"Local generation failed: {e}")


# ==========================================================================
# Utility Functions
# ==========================================================================

def is_available(provider: str = None) -> bool:
    """
    Check if a provider is available.
    
    Args:
        provider: Provider name to check, or None to check if any available
        
    Returns:
        True if provider is available
    """
    if not _initialized:
        initialize()
    
    if provider:
        return provider in _clients and _clients[provider] is not None
    
    # Check if any provider is available
    return any(v is not None for v in _clients.values())


def get_available_providers() -> list:
    """Get list of available provider names."""
    if not _initialized:
        initialize()
    
    return [k for k, v in _clients.items() if v is not None]


__all__ = [
    "initialize",
    "chat",
    "is_available",
    "get_available_providers",
    "get_provider",
]
