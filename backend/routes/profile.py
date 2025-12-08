"""
User Profile Routes

User profile management including picture uploads.
"""

from flask import Blueprint, request, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os

from models.user import User
from database import db
from utils.logging import get_logger
from utils.helpers import handle_errors, ValidationError, AuthenticationError

logger = get_logger(__name__)

bp = Blueprint("profile", __name__)

# Constants
USER_NOT_FOUND_MSG = "User not found"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'uploads/profile_pictures'


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@bp.route("/api/profile/<int:user_id>", methods=["GET"])
@handle_errors
def get_profile(user_id):
    """Get user profile."""
    user = User.query.get(user_id)
    if not user:
        raise ValidationError(USER_NOT_FOUND_MSG)
    
    return jsonify({
        "id": user.id,
        "email": user.email,
        "username": getattr(user, 'username', None),
        "profile_picture": getattr(user, 'profile_picture', None)
    }), 200


@bp.route("/api/profile/<int:user_id>", methods=["PUT"])
@handle_errors
def update_profile(user_id):
    """Update user profile."""
    user = User.query.get(user_id)
    if not user:
        raise ValidationError(USER_NOT_FOUND_MSG)
    
    data = request.get_json()
    
    # Update username
    if 'username' in data and hasattr(user, 'username'):
        user.username = data['username']
    
    # Update email (check for duplicates)
    if 'email' in data and data['email'] != user.email:
        if User.query.filter_by(email=data['email']).first():
            raise ValidationError("Email already in use")
        user.email = data['email']
    
    db.session.commit()
    
    return jsonify({
        "message": "Profile updated successfully",
        "user": {
            "id": user.id,
            "email": user.email,
            "username": getattr(user, 'username', None),
            "profile_picture": getattr(user, 'profile_picture', None)
        }
    }), 200


@bp.route("/api/profile/<int:user_id>/password", methods=["PUT"])
@handle_errors
def change_password(user_id):
    """Change user password."""
    user = User.query.get(user_id)
    if not user:
        raise ValidationError(USER_NOT_FOUND_MSG)
    
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not current_password or not new_password:
        raise ValidationError("Current and new passwords are required")
    
    # Verify current password
    if not check_password_hash(user.password, current_password):
        raise AuthenticationError("Current password is incorrect")
    
    # Validate new password
    if len(new_password) < 6:
        raise ValidationError("New password must be at least 6 characters")
    
    user.password = generate_password_hash(new_password)
    db.session.commit()
    
    return jsonify({"message": "Password changed successfully"}), 200


@bp.route("/api/profile/<int:user_id>/picture", methods=["POST"])
@handle_errors
def upload_profile_picture(user_id):
    """Upload profile picture."""
    user = User.query.get(user_id)
    if not user:
        raise ValidationError(USER_NOT_FOUND_MSG)
    
    if 'file' not in request.files:
        raise ValidationError("No file provided")
    
    file = request.files['file']
    if file.filename == '':
        raise ValidationError("No file selected")
    
    if not file or not allowed_file(file.filename):
        raise ValidationError("Invalid file type. Only PNG, JPG, JPEG, and GIF are allowed")
    
    # Create upload directory
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    
    # Save file
    filename = secure_filename(f"user_{user_id}_{file.filename}")
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)
    
    # Update user profile
    if hasattr(user, 'profile_picture'):
        user.profile_picture = f"/uploads/profile_pictures/{filename}"
    db.session.commit()
    
    profile_picture = getattr(user, 'profile_picture', None)
    return jsonify({
        "message": "Profile picture uploaded successfully",
        "profile_picture": profile_picture,
        "user": {
            "id": user.id,
            "email": user.email,
            "username": getattr(user, 'username', None),
            "profile_picture": profile_picture
        }
    }), 200


@bp.route("/api/profile/<int:user_id>/picture", methods=["DELETE"])
@handle_errors
def delete_profile_picture(user_id):
    """Delete profile picture."""
    user = User.query.get(user_id)
    if not user:
        raise ValidationError(USER_NOT_FOUND_MSG)
    
    # Delete file if it exists
    profile_picture = getattr(user, 'profile_picture', None)
    if profile_picture:
        file_path = os.path.join(".", profile_picture.lstrip("/"))
        if os.path.exists(file_path):
            os.remove(file_path)
    
    # Update user profile
    if hasattr(user, 'profile_picture'):
        user.profile_picture = None
    db.session.commit()
    
    return jsonify({"message": "Profile picture deleted successfully"}), 200
