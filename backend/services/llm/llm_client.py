# === FULL UPDATED llm_client.py (only changed / added parts are marked) ===
from typing import Dict, Any, Optional
from services.llm.base import BaseLLM as LLMInterface
from core.exceptions import LLMClientError
from core.logging import logger
import threading

_load_count = 0
_load_lock = threading.Lock()


try:
    from unsloth import FastLanguageModel
    import torch
    from pipeline.config import QWEN_MODEL, DEVICE, DTYPE
    HAS_UNSLOTH = True
except ImportError:
    HAS_UNSLOTH = False
    QWEN_MODEL = "mock-model"
    DEVICE = "cpu"
    DTYPE = "float32"



class QwenClient(LLMInterface):
    def __init__(self):
        self._available = False
        if not HAS_UNSLOTH:
            return
        try:
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                logger.info(f"GPU memory: {props.total_memory / 1e9:.1f} GB")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=QWEN_MODEL,
                max_seq_length=2048,
                dtype=DTYPE,
                load_in_4bit=True,
                device_map="auto",
            )
            FastLanguageModel.for_inference(self.model)
            self._available = True
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            self.model = None
            self.tokenizer = None

    # ---------- NEW CHAT METHOD (non‑thinking mode, always present) ----------
    def chat(self, system: str, user: str, max_new: int = 2048, first_line_only=False) -> str:
        """Chat method with proper Qwen chat template and better error handling."""
        logger.debug(f"[LLM] chat called - available: {self._available}")
        
        if not self._available:
            logger.debug("[LLM] Model not available, returning fallback")
            return user + "..." if len(user) > 100 else user
        
        try:
            # Build messages for Qwen chat template
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
            device = getattr(self.model, "device", DEVICE)
            for k in inputs:
                inputs[k] = inputs[k].to(device)
            
            # Keep existing inference parameters as requested
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                temperature=0.7,
                top_p=0.8,
                top_k=20,
                min_p=0.0,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Simple response extraction - just get the new tokens
            input_length = inputs['input_ids'].shape[1]
            new_tokens = outputs[0][input_length:]
            result = self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            
            # Handle empty responses
            if not result:
                logger.warning("[LLM] Generated empty response, using fallback")
                return user + "..." if len(user) > 100 else user
            
            logger.debug(f"[LLM] Generated response: {result}")
            
            if first_line_only and result:
                return result.split('\n', 1)[0].strip()
            return result
            
        except Exception as e:
            logger.error(f"Chat generation failed: {e}")
            return user + "..." if len(user) > 100 else user  # Return input as fallback

    def generate_response(self, prompt: str, max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None, **kwargs) -> str:
        if not self._available:
            return prompt + "..." if len(prompt) > 100 else prompt  # Better fallback
        # always delegate to chat so that prompt isn't swallowed
        return self.chat("You are a helpful assistant.", prompt, max_new=max_tokens or 1024)

    # remaining helper methods (is_available, get_model_info, etc.) stay unchanged

    def is_available(self) -> bool:
        """Return True if the model loaded successfully."""
        return getattr(self, "_available", False)

    def get_model_info(self) -> Dict[str, Any]:
        """Return basic metadata about the loaded model."""
        return {
            "model_name": QWEN_MODEL,
            "device": DEVICE,
            "dtype": DTYPE,
            "available": self._available,
            "has_unsloth": HAS_UNSLOTH
        }
    def validate_content(self, text: str) -> Dict[str, Any]:
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
    
    def clear_cache(self):
        """Clear model cache to ensure fresh state."""
        if self.model and hasattr(self.model, 'past_key_values'):
            self.model.past_key_values = None
        logger.debug("[LLM] Model cache cleared")
