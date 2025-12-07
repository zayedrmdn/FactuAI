"""
Logging configuration for FactuAI Backend

Provides centralized logging setup with both file and console handlers.
"""

import logging
from pathlib import Path

# Log file path - stored in backend root
LOG_PATH = Path(__file__).resolve().parent.parent / "factcheck_debug.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Create the main logger
logger = logging.getLogger("fact_check")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Prevent duplicate handlers if module is reloaded
if not logger.handlers:
    # File handler (detailed log)
    file_handler = logging.FileHandler(str(LOG_PATH), mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler (simple log) - UTF-8 encoding to prevent Unicode errors
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    # Force UTF-8 encoding for Windows console
    if hasattr(console_handler.stream, 'reconfigure'):
        console_handler.stream.reconfigure(encoding='utf-8')
    console_formatter = logging.Formatter("%(message)s")
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)


def get_logger(name: str = None):
    """
    Get a logger instance for the given name.
    
    Args:
        name: Optional name for the logger. If provided, creates a child logger
              under the main fact_check logger.
    
    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"fact_check.{name}")
    return logger
