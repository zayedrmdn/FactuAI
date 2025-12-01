"""
Local LLM Provider (Unsloth/Qwen)

Local GPU-based LLM provider using Unsloth for optimized inference.
Only loaded when APP_RUN_MODE=local to avoid PyTorch imports in cloud mode.
"""

import os
from typing import Dict, Any, Optional, List
import threading

from services.llm.base import BaseLLM
from core.logging import logger
from core.exceptions import LLMClientError


class LocalLLM(BaseLLM):
    """
    Local LLM provider using Unsloth.
    
    Loads models locally on GPU using Unsloth's optimized inference.
    Only import PyTorch/Unsloth when this class is instantiated.
    """
    
    DEFAULT_MODEL = "unsloth/Qwen3-4B-unsloth-bnb-4bit"
    DEFAULT_MAX_TOKENS = 2048
    DEFAULT_TEMPERATURE = 0.7
    
    _load_lock = threading.Lock()
    
    def __init__(self, model_name: Optional[str] = None):
        """
        Initialize local LLM provider.
        
        Args:
            model_name: Model to load (defaults to QWEN_MODEL env var)
        """
        self.model_name = model_name or os.getenv("QWEN_MODEL", self.DEFAULT_MODEL)
        self.model = None
        self.tokenizer = None
        self._available = False
        self._device = "cpu"
        self._dtype = None
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Load the local model - imports PyTorch/Unsloth only here."""
        with self._load_lock:
            try:
                # Lazy imports to avoid loading in cloud mode
                import torch
                from unsloth import FastLanguageModel
                
                self._device = "cuda" if torch.cuda.is_available() else "cpu"
                self._dtype = torch.float16 if torch.cuda.is_available() else torch.float32
                
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    logger.info(f"[LOCAL] GPU memory: {props.total_memory / 1e9:.1f} GB")
                
                logger.info(f"[LOCAL] Loading model: {self.model_name}")
                self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                    model_name=self.model_name,
                    max_seq_length=2048,
                    dtype=self._dtype,
                    load_in_4bit=True,
                    device_map="auto",
                )
                FastLanguageModel.for_inference(self.model)
                self._available = True
                logger.info(f"[LOCAL] Model loaded successfully on {self._device}")
                
            except ImportError as e:
                logger.error(f"[LOCAL] PyTorch/Unsloth not installed: {e}")
                self._available = False
            except Exception as e:
                logger.error(f"[LOCAL] Model load failed: {e}")
                self._available = False
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs
    ) -> str:
        """Generate a response using local model."""
        if not self._available:
            return prompt + "..." if len(prompt) > 100 else prompt
        
        # Delegate to chat method
        return self.chat(
            "You are a helpful assistant.",
            prompt,
            max_tokens=max_tokens or self.DEFAULT_MAX_TOKENS,
            temperature=temperature
        )
    
    def chat(
        self,
        system: str,
        user: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        first_line_only: bool = False,
        **kwargs
    ) -> str:
        """Generate a response using chat format."""
        if not self._available:
            logger.debug("[LOCAL] Model not available, returning fallback")
            return user + "..." if len(user) > 100 else user
        
        try:
            import torch
            
            # Build messages for chat template
            messages = [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ]
            
            # Apply chat template
            if hasattr(self.tokenizer, 'apply_chat_template'):
                prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False
                )
            else:
                # Fallback manual template for Qwen
                prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
            
            inputs = self.tokenizer([prompt], return_tensors="pt")
            device = getattr(self.model, "device", self._device)
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            
            # Generate
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens or self.DEFAULT_MAX_TOKENS,
                temperature=temperature or self.DEFAULT_TEMPERATURE,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Extract response
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            if not result:
                logger.warning("[LOCAL] Generated empty response, using fallback")
                return user + "..." if len(user) > 100 else user
            
            logger.debug(f"[LOCAL] Generated {len(result)} chars")
            
            if first_line_only and result:
                return result.split('\n', 1)[0].strip()
            return result
            
        except Exception as e:
            logger.error(f"[LOCAL] Generation failed: {e}")
            return user + "..." if len(user) > 100 else user
    
    def is_available(self) -> bool:
        """Check if local model is available."""
        return self._available
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get local model info."""
        return {
            "provider": "local",
            "model_name": self.model_name,
            "device": self._device,
            "dtype": str(self._dtype),
            "available": self._available,
        }
    
    def clear_cache(self) -> None:
        """Clear model cache to ensure fresh state."""
        if self._available and self.model is not None:
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
