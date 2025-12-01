"""
Pipeline Configuration Constants

Centralized configuration for the fact-checking pipeline.
Moved from modules/factcheck/utils/config.py for cleaner imports.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# Device Configuration (lazy import to avoid PyTorch in cloud mode)
# =============================================================================
def get_device_config():
    """Get device configuration lazily to avoid importing torch in cloud mode."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return device, dtype
    except ImportError:
        return "cpu", "float32"


# Lazy device/dtype - only evaluate when needed
_device_config = None

def get_device():
    global _device_config
    if _device_config is None:
        _device_config = get_device_config()
    return _device_config[0]

def get_dtype():
    global _device_config
    if _device_config is None:
        _device_config = get_device_config()
    return _device_config[1]


# For backward compatibility with modules that import directly
DEVICE = property(lambda self: get_device())
DTYPE = property(lambda self: get_dtype())

# =============================================================================
# Model Paths
# =============================================================================
QWEN_MODEL = os.getenv("QWEN_MODEL", "unsloth/Qwen3-4B-unsloth-bnb-4bit")
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH",
    "D:/Projects/FactuAI/scripts/factchecker/distilbert_liar2_final/final-1"
)

# =============================================================================
# API Keys
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "1b185c94c71d4e9381fbb185f5552225")

# =============================================================================
# Pipeline Constants
# =============================================================================
MAX_EVIDENCE_WORDS = int(os.getenv("MAX_EVIDENCE_WORDS", "1024"))
SENTS_PER_ARTICLE_DEFAULT = 5
MIN_SENT_WORDS = 8
SIMILARITY_THRESHOLD = 0.25

# =============================================================================
# Paths
# =============================================================================
BACKEND_ROOT = Path(__file__).resolve().parent.parent
SCRAPING_LOG_PATH = BACKEND_ROOT / "pipeline" / "scraping_logs.txt"
ARTICLE_CACHE_PATH = BACKEND_ROOT / "pipeline" / "article_cache.json"

# =============================================================================
# Legacy constants for compatibility
# =============================================================================
DEVICE = get_device()
DTYPE = str(get_dtype()).replace('torch.', '')
