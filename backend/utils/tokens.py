"""
Token utilities for FactuAI Backend

Handles generation and verification of secure tokens (password reset, etc.).
"""

import os
from itsdangerous import URLSafeTimedSerializer, BadTimeSignature, SignatureExpired
from utils.logging import get_logger

logger = get_logger(__name__)

# Use a secure secret key for token generation
SECRET_KEY = os.environ.get("RESEND_API_KEY", "fallback-secret-key-change-in-production")
serializer = URLSafeTimedSerializer(SECRET_KEY)


def generate_reset_token(user_id: str) -> str:
    """
    Generate a secure password reset token for a user
    
    Args:
        user_id: User ID to encode in the token
        
    Returns:
        Secure token string
        
    Raises:
        Exception if token generation fails
    """
    try:
        token = serializer.dumps(user_id, salt="password-reset")
        logger.info(f"Generated reset token for user {user_id}")
        return token
    except Exception as e:
        logger.error(f"Failed to generate reset token for user {user_id}: {str(e)}")
        raise


def verify_reset_token(token: str, max_age: int = 1800) -> str | None:
    """
    Verify and decode a password reset token
    
    Args:
        token: The token to verify
        max_age: Maximum age in seconds (default: 1800 = 30 minutes)
        
    Returns:
        User ID if token is valid, None otherwise
    """
    try:
        user_id = serializer.loads(token, salt="password-reset", max_age=max_age)
        logger.info(f"Successfully verified reset token for user {user_id}")
        return user_id
    except SignatureExpired:
        logger.warning("Password reset token has expired")
        return None
    except BadTimeSignature:
        logger.warning("Invalid password reset token signature")
        return None
    except Exception as e:
        logger.error(f"Failed to verify reset token: {str(e)}")
        return None
