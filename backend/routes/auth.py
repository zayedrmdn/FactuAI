"""
Authentication Routes

User registration, login, and password reset.
"""

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
import re

from models.user import User
from database import db
from utils.email import send_password_reset_email
from utils.tokens import generate_reset_token, verify_reset_token
from utils.logging import get_logger
from utils.helpers import handle_errors, AuthenticationError, ValidationError

logger = get_logger(__name__)

bp = Blueprint("auth", __name__)

# Constants
RESET_LINK_SENT_MSG = "If that email exists, a reset link was sent."
INVALID_TOKEN_MSG = "Invalid or expired token"


def find_user_by_email_or_username(identifier: str):
    """Find user by email or username."""
    # Try email first
    user = User.query.filter_by(email=identifier).first()
    
    # Try username if not found
    if not user:
        try:
            user = User.query.filter_by(username=identifier).first()
        except Exception:
            pass  # Username column might not exist
    
    return user


@bp.route("/api/register", methods=["POST"])
@handle_errors
def register():
    """Register a new user."""
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        raise ValidationError("Email and password are required")

    # Check if email exists
    if User.query.filter_by(email=email).first():
        raise ValidationError("Email already registered")

    # Check if username is taken
    if username:
        if find_user_by_email_or_username(username):
            raise ValidationError("Username already taken")

    # Hash password
    hashed = generate_password_hash(password)

    # Create user
    try:
        if username and hasattr(User, 'username'):
            new_user = User(email=email, username=username, password=hashed)
        else:
            new_user = User(email=email, password=hashed)
    except Exception:
        new_user = User(email=email, password=hashed)

    db.session.add(new_user)
    db.session.commit()

    return jsonify({"message": "User registered successfully"}), 201


@bp.route("/api/login", methods=["POST"])
@handle_errors
def login():
    """User login."""
    data = request.get_json()
    identifier = data.get("email")
    password = data.get("password")

    if not identifier or not password:
        raise ValidationError("Email/username and password are required")

    user = find_user_by_email_or_username(identifier)

    if not user or not check_password_hash(user.password, password):
        raise AuthenticationError("Invalid email/username or password")

    return jsonify({
        "message": "Login successful",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": getattr(user, "username", None)
        }
    }), 200


@bp.route("/api/auth/request-password-reset", methods=["POST"])
def request_password_reset():
    """Request password reset email."""
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()

        # Basic email validation (never reveal if email exists)
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not email or not re.match(email_pattern, email):
            return jsonify({"message": RESET_LINK_SENT_MSG}), 200

        user = User.query.filter_by(email=email).first()

        # Always return success (don't reveal if email exists)
        if user:
            try:
                token = generate_reset_token(str(user.id))
                send_password_reset_email(email, token)
                logger.info(f"Password reset email sent to: {email}")
            except Exception as e:
                logger.error(f"Failed to send reset email: {e}")
        else:
            logger.info(f"Reset requested for non-existent email: {email}")

        return jsonify({"message": RESET_LINK_SENT_MSG}), 200

    except Exception as e:
        logger.error(f"Error in password reset request: {e}")
        return jsonify({"message": RESET_LINK_SENT_MSG}), 200


@bp.route("/api/auth/reset-password", methods=["POST"])
@handle_errors
def reset_password():
    """Reset password with token."""
    data = request.get_json()
    token = data.get("token", "").strip()
    new_password = data.get("new_password", "")

    if not token or not new_password or len(new_password) < 8:
        raise ValidationError(INVALID_TOKEN_MSG)

    user_id = verify_reset_token(token)
    if not user_id:
        raise ValidationError(INVALID_TOKEN_MSG)

    user = User.query.get(user_id)
    if not user:
        raise ValidationError(INVALID_TOKEN_MSG)

    user.password = generate_password_hash(new_password)
    db.session.commit()

    logger.info(f"Password reset successful for user {user_id}")
    return jsonify({"message": "Password reset successful"}), 200
