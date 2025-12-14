# Full Path: backend/app/core/circuit_breaker.py
"""
Circuit Breaker and Retry Strategy for external API calls.

Implements the Circuit Breaker pattern to prevent cascading failures
when external services (LLM, Search APIs) are experiencing issues.

Uses `tenacity` for retry logic with exponential backoff.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Any, Callable, Optional, TypeVar, ParamSpec

from app.core.logging import get_logger

logger = get_logger(__name__)

P = ParamSpec("P")
T = TypeVar("T")


class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = "closed"      # Normal operation, requests flow through
    OPEN = "open"          # Failure threshold exceeded, requests blocked
    HALF_OPEN = "half_open"  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for a circuit breaker."""
    failure_threshold: int = 5          # Number of failures before opening
    success_threshold: int = 2          # Successes needed to close from half-open
    timeout_seconds: float = 60.0       # How long to stay open before half-open
    # Retry configuration
    max_retries: int = 3                # Maximum retry attempts
    retry_delay_seconds: float = 1.0    # Initial delay between retries
    retry_multiplier: float = 2.0       # Exponential backoff multiplier
    retry_max_delay_seconds: float = 30.0  # Maximum delay between retries


@dataclass
class CircuitBreakerState:
    """Mutable state for a circuit breaker."""
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    success_count: int = 0
    last_failure_time: float = 0.0
    last_error: Optional[str] = None


# Global registry of circuit breakers by name
_circuit_breakers: dict[str, CircuitBreakerState] = {}
_circuit_configs: dict[str, CircuitBreakerConfig] = {}


def get_circuit_state(name: str) -> Optional[CircuitBreakerState]:
    """Get the current state of a named circuit breaker."""
    return _circuit_breakers.get(name)


def reset_circuit(name: str) -> None:
    """Reset a circuit breaker to closed state."""
    if name in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreakerState()
        logger.info(f"[CIRCUIT:{name}] Reset to CLOSED")


def reset_all_circuits() -> None:
    """Reset all circuit breakers."""
    for name in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreakerState()
    logger.info("[CIRCUIT] All circuits reset to CLOSED")


class CircuitOpenError(Exception):
    """Raised when a request is blocked by an open circuit."""
    def __init__(self, circuit_name: str, retry_after: float, last_error: Optional[str] = None):
        self.circuit_name = circuit_name
        self.retry_after = retry_after
        self.last_error = last_error
        super().__init__(
            f"Circuit '{circuit_name}' is OPEN. Retry after {retry_after:.1f}s. "
            f"Last error: {last_error or 'unknown'}"
        )


def _should_retry_exception(exc: Exception) -> bool:
    """
    Determine if an exception is retryable.
    
    - Transient network errors: RETRY
    - Authentication errors (401, 403): DO NOT RETRY (trip breaker)
    - Rate limiting (429): RETRY with backoff
    - Server errors (5xx): RETRY
    - Client errors (4xx except 429): DO NOT RETRY
    """
    import httpx

    # httpx specific errors
    if isinstance(exc, httpx.ConnectError):
        return True  # Network connectivity issue, retry
    if isinstance(exc, httpx.TimeoutException):
        return True  # Timeout, retry
    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        if status == 429:  # Rate limited
            return True
        if 500 <= status < 600:  # Server error
            return True
        if status in (401, 403):  # Auth errors - don't retry, trip breaker
            return False
        return False  # Other client errors
    
    # OpenAI client errors
    error_str = str(exc).lower()
    if "connection" in error_str or "timeout" in error_str:
        return True
    if "rate limit" in error_str or "429" in error_str:
        return True
    if "500" in error_str or "502" in error_str or "503" in error_str or "504" in error_str:
        return True
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return False
    
    # Default: retry for unknown errors
    return True


def _should_trip_breaker(exc: Exception) -> bool:
    """
    Determine if an exception should count toward tripping the breaker.
    
    Authentication errors should immediately trip the breaker since
    retrying won't help.
    """
    error_str = str(exc).lower()
    if "401" in error_str or "403" in error_str or "unauthorized" in error_str:
        return True
    if "invalid api key" in error_str or "authentication" in error_str:
        return True
    return True  # Most errors should count


def circuit_breaker(
    name: str,
    config: Optional[CircuitBreakerConfig] = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """
    Decorator to wrap an async function with circuit breaker and retry logic.
    
    Usage:
        @circuit_breaker("llm_api", CircuitBreakerConfig(failure_threshold=3))
        async def call_llm_api(...):
            ...
    
    Args:
        name: Unique identifier for this circuit breaker
        config: Optional configuration, uses defaults if not provided
    """
    cfg = config or CircuitBreakerConfig()
    
    # Initialize circuit state if not exists
    if name not in _circuit_breakers:
        _circuit_breakers[name] = CircuitBreakerState()
        _circuit_configs[name] = cfg
    
    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            state = _circuit_breakers[name]
            
            # Check if circuit is OPEN
            if state.state == CircuitState.OPEN:
                elapsed = time.time() - state.last_failure_time
                if elapsed < cfg.timeout_seconds:
                    retry_after = cfg.timeout_seconds - elapsed
                    raise CircuitOpenError(name, retry_after, state.last_error)
                else:
                    # Transition to HALF_OPEN
                    state.state = CircuitState.HALF_OPEN
                    state.success_count = 0
                    logger.info(f"[CIRCUIT:{name}] Transitioning to HALF_OPEN")
            
            # Attempt with retries
            last_exception: Optional[Exception] = None
            delay = cfg.retry_delay_seconds
            
            for attempt in range(cfg.max_retries + 1):
                try:
                    result = await func(*args, **kwargs)
                    
                    # Success - update state
                    if state.state == CircuitState.HALF_OPEN:
                        state.success_count += 1
                        if state.success_count >= cfg.success_threshold:
                            state.state = CircuitState.CLOSED
                            state.failure_count = 0
                            logger.info(f"[CIRCUIT:{name}] Recovered, now CLOSED")
                    else:
                        # Reset failure count on success in CLOSED state
                        state.failure_count = 0
                    
                    return result
                    
                except Exception as exc:
                    last_exception = exc
                    
                    # Check if we should retry
                    if attempt < cfg.max_retries and _should_retry_exception(exc):
                        logger.warning(
                            f"[CIRCUIT:{name}] Attempt {attempt + 1}/{cfg.max_retries + 1} "
                            f"failed: {exc}. Retrying in {delay:.1f}s..."
                        )
                        await asyncio.sleep(delay)
                        delay = min(delay * cfg.retry_multiplier, cfg.retry_max_delay_seconds)
                        continue
                    
                    # No more retries or non-retryable error
                    if _should_trip_breaker(exc):
                        state.failure_count += 1
                        state.last_failure_time = time.time()
                        state.last_error = str(exc)[:200]
                        
                        if state.state == CircuitState.HALF_OPEN:
                            # Failed during half-open, go back to OPEN
                            state.state = CircuitState.OPEN
                            logger.warning(f"[CIRCUIT:{name}] Failed during HALF_OPEN, back to OPEN")
                        elif state.failure_count >= cfg.failure_threshold:
                            # Exceeded threshold, trip the breaker
                            state.state = CircuitState.OPEN
                            logger.error(
                                f"[CIRCUIT:{name}] Threshold exceeded ({state.failure_count}/{cfg.failure_threshold}), "
                                f"tripping to OPEN for {cfg.timeout_seconds}s"
                            )
                    
                    raise
            
            # Should not reach here, but just in case
            if last_exception:
                raise last_exception
            raise RuntimeError(f"Circuit breaker {name} exhausted retries without result")
        
        return wrapper
    return decorator


# Pre-configured circuit breakers for common use cases
LLM_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    success_threshold=2,
    timeout_seconds=120.0,  # LLM APIs might have longer recovery times
    max_retries=2,
    retry_delay_seconds=2.0,
    retry_multiplier=2.0,
    retry_max_delay_seconds=15.0,
)

SEARCH_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=5,
    success_threshold=2,
    timeout_seconds=60.0,
    max_retries=2,
    retry_delay_seconds=1.0,
    retry_multiplier=2.0,
    retry_max_delay_seconds=10.0,
)

EMBEDDING_CIRCUIT_CONFIG = CircuitBreakerConfig(
    failure_threshold=3,
    success_threshold=1,
    timeout_seconds=30.0,
    max_retries=1,
    retry_delay_seconds=1.0,
    retry_multiplier=2.0,
    retry_max_delay_seconds=5.0,
)
