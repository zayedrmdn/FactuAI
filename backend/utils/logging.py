"""
Logging configuration for FactuAI Backend

Provides centralized logging setup with both file and console handlers.
"""

import logging
from pathlib import Path

# Log file path - stored in backend root
LOG_PATH = Path(__file__).resolve().parent.parent / "factcheck_debug.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Sanitization helper for Windows console
def sanitize_for_console(text: str) -> str:
    """
    Sanitize text for Windows console output.
    Replaces non-ASCII characters to prevent UnicodeEncodeError.
    """
    if isinstance(text, str):
        try:
            # Try to encode as ASCII, replace problematic chars
            return text.encode('ascii', 'replace').decode('ascii')
        except Exception:
            return str(text)
    return str(text)


class SafeFormatter(logging.Formatter):
    """Formatter that sanitizes Unicode for Windows console."""
    def format(self, record):
        # Sanitize the message
        if isinstance(record.msg, str):
            record.msg = sanitize_for_console(record.msg)
        # Sanitize args
        if record.args:
            record.args = tuple(sanitize_for_console(str(arg)) for arg in record.args)
        return super().format(record)


# Create the main logger
logger = logging.getLogger("fact_check")
logger.setLevel(logging.DEBUG)
logger.propagate = False

# Prevent duplicate handlers if module is reloaded
if not logger.handlers:
    # File handler (detailed log) - UTF-8 for full Unicode support
    file_handler = logging.FileHandler(str(LOG_PATH), mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    file_handler.setFormatter(file_formatter)

    # Console handler (simple log) - sanitized for Windows console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_formatter = SafeFormatter("%(message)s")
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


def log_model_init(logger_instance, provider: str, model_id: str, status: str = "success"):
    """
    Log model initialization with consistent formatting.
    
    Args:
        logger_instance: Logger instance to use
        provider: Provider name (openrouter, nvidia, local)
        model_id: Model identifier
        status: Initialization status (success, failed, fallback)
    """
    status_emoji = {"success": "✓", "failed": "✗", "fallback": "⚠"}
    logger_instance.info(
        f"[MODEL_INIT] {status_emoji.get(status, '•')} Provider: {provider} | "
        f"Model: {model_id} | Status: {status.upper()}"
    )


def log_api_request(logger_instance, endpoint: str, method: str = "POST", **kwargs):
    """
    Log API requests with structured format.
    
    Args:
        logger_instance: Logger instance to use
        endpoint: API endpoint path
        method: HTTP method
        **kwargs: Additional request parameters to log
    """
    params_str = " | ".join([f"{k}={v}" for k, v in kwargs.items() if v is not None])
    logger_instance.info(f"[API_REQUEST] {method} {endpoint} | {params_str}")


def log_pipeline_stage(logger_instance, stage: str, claim: str = None, progress: int = 0):
    """
    Log pipeline processing stages.
    
    Args:
        logger_instance: Logger instance to use
        stage: Pipeline stage name
        claim: Optional claim being processed
        progress: Progress percentage (0-100)
    """
    claim_str = ""
    if claim:
        if len(claim) > 50:
            claim_str = f" | Claim: {claim[:50]}..."
        else:
            claim_str = f" | Claim: {claim}"
    logger_instance.debug(f"[PIPELINE_STAGE] {stage} | Progress: {progress}%{claim_str}")
