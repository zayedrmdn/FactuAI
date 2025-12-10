"""
API Rate Limits Routes

Proxy for OpenRouter API key limits with rate limiting and caching.
"""

from flask import Blueprint, jsonify
import requests
from datetime import datetime, timedelta
from functools import wraps
import os

from utils.logging import get_logger
from utils.helpers import handle_errors, ValidationError

logger = get_logger(__name__)

bp = Blueprint("limits", __name__)

# Rate limiting: cache result for 10 seconds to prevent abuse
_cache = {
    "data": None,
    "timestamp": None
}
CACHE_DURATION = timedelta(seconds=10)


def rate_limit(f):
    """Rate limiter decorator using simple in-memory cache."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        now = datetime.now()
        
        # Return cached data if available and fresh
        if _cache["data"] and _cache["timestamp"]:
            age = now - _cache["timestamp"]
            if age < CACHE_DURATION:
                logger.debug(f"Returning cached limits (age: {age.total_seconds():.1f}s)")
                return jsonify(_cache["data"]), 200
        
        # Fetch new data
        result = f(*args, **kwargs)
        
        # Update cache
        if isinstance(result, tuple) and result[1] == 200:
            _cache["data"] = result[0].get_json()
            _cache["timestamp"] = now
            logger.debug("Updated limits cache")
        
        return result
    
    return decorated_function


@bp.route("/api/limits", methods=["GET"])
@handle_errors
@rate_limit
def get_limits():
    """
    Get OpenRouter API key limits and usage.
    
    Returns:
        JSON with rate limits, credit usage, and quotas
    
    Rate Limited: 1 request per 10 seconds (cached)
    """
    # Get API key from environment
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValidationError("OpenRouter API key not configured")
    
    # Make request to OpenRouter
    try:
        response = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={
                "Authorization": f"Bearer {api_key}"
            },
            timeout=10
        )
        
        response.raise_for_status()
        data = response.json()
        
        logger.info("Successfully fetched OpenRouter limits")
        return jsonify(data), 200
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to fetch OpenRouter limits: {str(e)}")
        raise ValidationError(f"Failed to fetch API limits: {str(e)}")
