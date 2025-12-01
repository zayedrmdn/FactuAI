from flask import Blueprint, request, jsonify
from database.models.user import User
from database.connection import db
from werkzeug.security import generate_password_hash, check_password_hash
from services.email import send_password_reset_email
from services.tokens import generate_reset_token, verify_reset_token
from core.logging import get_logger
import re

logger = get_logger(__name__)
auth_bp = Blueprint("auth", __name__)

def find_user_by_email_or_username(email_or_username):
    """
    Find a user by email or username.
    Returns the user if found, None otherwise.
    """
    # Try to find user by email first
    user = User.query.filter_by(email=email_or_username).first()
    
    # If not found by email, try by username (if username column exists)
    if not user:
        try:
            user = User.query.filter_by(username=email_or_username).first()
        except Exception:
            # Username column might not exist yet, ignore this error
            pass
    
    return user

@auth_bp.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username")
    email = data.get("email")
    password = data.get("password")

    if not email or not password:
        return jsonify({"error": "Email and password are required"}), 400

    email_exists = User.query.filter_by(email=email).first()
    if email_exists:
        return jsonify({"error": "Email already registered"}), 400

    if username:
        username_taken = find_user_by_email_or_username(username)
        if username_taken:
            return jsonify({"error": "Username already taken"}), 400

    hashed = generate_password_hash(password)

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


@auth_bp.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    identifier = data.get("email")
    password = data.get("password")

    if not identifier or not password:
        return jsonify({"error": "Missing fields"}), 400

    user = find_user_by_email_or_username(identifier)

    if not user or not check_password_hash(user.password, password):
        return jsonify({"error": "Invalid email/username or password"}), 401

    return jsonify({
        "message": "Login successful",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": getattr(user, "username", None)
        }
    }), 200


@auth_bp.route("/api/auth/request-password-reset", methods=["POST"])
def request_password_reset():
    try:
        data = request.get_json()
        email = data.get("email", "").strip().lower()

        # Reject clearly invalid emails (but never reveal validity in response)
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not email or not re.match(email_pattern, email):
            return jsonify({"message": "If that email exists, a reset link was sent."}), 200

        user = User.query.filter_by(email=email).first()

        # Always return success to avoid exposing account existence
        if user:
            try:
                token = generate_reset_token(str(user.id))
                send_password_reset_email(email, token)
                logger.info(f"Password reset email sent to: {email}")
            except Exception as e:
                logger.error(f"Failed to send reset email to {email}: {str(e)}")
        else:
            logger.info(f"Reset requested for non-existent email: {email}")

        return jsonify({"message": "If that email exists, a reset link was sent."}), 200

    except Exception as e:
        logger.error(f"Unhandled error during password reset request: {str(e)}")
        return jsonify({"message": "If that email exists, a reset link was sent."}), 200

@auth_bp.route("/api/auth/reset-password", methods=["POST"])
def reset_password():
    try:
        data = request.get_json()
        token = data.get("token", "").strip()
        new_password = data.get("new_password", "")

        if not token or not new_password or len(new_password) < 8:
            return jsonify({"message": "Invalid or expired token"}), 400

        user_id = verify_reset_token(token)
        if not user_id:
            return jsonify({"message": "Invalid or expired token"}), 400

        user = User.query.get(user_id)
        if not user:
            return jsonify({"message": "Invalid or expired token"}), 400

        user.password = generate_password_hash(new_password)
        db.session.commit()

        logger.info(f"Password reset successful for user {user_id}")
        return jsonify({"message": "Password reset successful"}), 200

    except Exception as e:
        logger.error(f"Error in password reset: {str(e)}")
        return jsonify({"message": "An error occurred while resetting your password"}), 500

