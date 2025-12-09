"""
Centralized Configuration for FactuAI Backend

Consolidated from:
- core/config.py
- pipeline/config.py
- services/classifier/constants.py

This module provides all application configuration with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Set HF_HOME to suppress TRANSFORMERS_CACHE deprecation warning
if "HF_HOME" not in os.environ and "TRANSFORMERS_CACHE" not in os.environ:
    # Default to .cache directory in project root
    os.environ["HF_HOME"] = str(Path(__file__).parent.parent / ".cache" / "huggingface")


# ==========================================================================
# Run Mode Configuration
# ==========================================================================
APP_RUN_MODE = os.getenv("APP_RUN_MODE", "cloud").lower()
USE_LOCAL_LLM = APP_RUN_MODE == "local"
USE_LOCAL_CLASSIFIER = APP_RUN_MODE == "local"


# ==========================================================================
# Database Configuration
# ==========================================================================
DATABASE_URI = os.getenv("DB_URI")
SQLALCHEMY_DATABASE_URI = DATABASE_URI  # Alias for SQLAlchemy
SQLALCHEMY_TRACK_MODIFICATIONS = False


# ==========================================================================
# Security Configuration
# ==========================================================================
SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-here")


# ==========================================================================
# Frontend Configuration
# ==========================================================================
FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")


# ==========================================================================
# Email Configuration
# ==========================================================================
RESEND_API_KEY = os.getenv("RESEND_API_KEY")
FROM_EMAIL = os.getenv("FROM_EMAIL", "noreply@your-domain.com")


# ==========================================================================
# LLM Configuration
# ==========================================================================
# Provider selection: openrouter, nvidia, or local
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "nvidia" if APP_RUN_MODE == "cloud" else "local")

# OpenRouter
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "anthropic/claude-3-haiku")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# NVIDIA
NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_MODEL = os.getenv("NVIDIA_MODEL", "meta/llama-3.1-8b-instruct")
NVIDIA_BASE_URL = os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1")

# Local Model
QWEN_MODEL = os.getenv("QWEN_MODEL", "unsloth/Qwen2.5-7B-unsloth-bnb-4bit")


# ==========================================================================
# Search API Configuration
# ==========================================================================
# Support both GOOGLE_CSE_ID and GOOGLE_CX_ID for compatibility
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CX_ID = os.getenv("GOOGLE_CX_ID") or os.getenv("GOOGLE_CSE_ID")
GOOGLE_CSE_ID = GOOGLE_CX_ID  # Backwards-compatible alias
NEWS_API_KEY = os.getenv("NEWS_API_KEY") or os.getenv("NEWSAPI_KEY")


# ==========================================================================
# Pipeline Settings
# ==========================================================================
MAX_EVIDENCE_WORDS = int(os.getenv("MAX_EVIDENCE_WORDS", "300"))
MAX_CLAIMS = int(os.getenv("MAX_CLAIMS", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.3"))
EVIDENCE_MAX_RESULTS = int(os.getenv("EVIDENCE_MAX_RESULTS", "5"))
SENTS_PER_ARTICLE_DEFAULT = 5
MIN_SENT_WORDS = 8


# ==========================================================================
# Classifier Settings
# ==========================================================================
CLASSIFIER_PATH = os.getenv(
    "CLASSIFIER_PATH",
    "D:/Projects/FactuAI/scripts/factchecker/distilbert_liar2_final/final-1"
)

CLASSIFIER_LABELS = [
    "false",
    "mostly_false",
    "barely_true",
    "half_true",
    "mostly_true",
    "true",
]


# ==========================================================================
# Paths
# ==========================================================================
BACKEND_ROOT = Path(__file__).resolve().parent
SCRAPING_LOG_PATH = BACKEND_ROOT / "pipeline" / "scraping_logs.txt"
ARTICLE_CACHE_PATH = BACKEND_ROOT / "pipeline" / "article_cache.json"


# ==========================================================================
# Device Configuration (lazy load to avoid PyTorch in cloud mode)
# ==========================================================================
_device_config = None

def get_device_config():
    """Get device configuration lazily to avoid importing torch in cloud mode."""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return device, dtype
    except ImportError:
        return "cpu", "float32"

def get_device():
    """Get the compute device (cuda/cpu)."""
    global _device_config
    if _device_config is None:
        _device_config = get_device_config()
    return _device_config[0]

def get_dtype():
    """Get the data type for tensors."""
    global _device_config
    if _device_config is None:
        _device_config = get_device_config()
    return _device_config[1]


# ==========================================================================
# Flask Configuration Class
# ==========================================================================
class Config:
    """Flask application configuration."""
    SECRET_KEY = SECRET_KEY
    SQLALCHEMY_DATABASE_URI = SQLALCHEMY_DATABASE_URI or "sqlite:///factuai.db"
    SQLALCHEMY_TRACK_MODIFICATIONS = SQLALCHEMY_TRACK_MODIFICATIONS


# ==========================================================================
# Helper Functions
# ==========================================================================
def is_cloud_mode() -> bool:
    """Check if application is running in cloud mode."""
    return APP_RUN_MODE == "cloud"

def is_local_mode() -> bool:
    """Check if application is running in local mode."""
    return APP_RUN_MODE == "local"

def get_mode_info() -> dict:
    """Get information about the current run mode configuration."""
    return {
        "run_mode": APP_RUN_MODE,
        "use_local_llm": USE_LOCAL_LLM,
        "use_local_classifier": USE_LOCAL_CLASSIFIER,
        "llm_provider": LLM_PROVIDER,
    }
