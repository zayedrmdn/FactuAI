from flask import Blueprint, request, jsonify
from database.models.user import User
from database.connection import db
from werkzeug.security import generate_password_hash, check_password_hash
import os
from werkzeug.utils import secure_filename

profile_bp = Blueprint("profile", __name__)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
UPLOAD_FOLDER = 'uploads/profile_pictures'

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@profile_bp.route("/api/profile/<int:user_id>", methods=["GET"])
def get_profile(user_id):
    """Get user profile information"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    # Handle case where new columns might not exist yet
    username = getattr(user, 'username', None)
    profile_picture = getattr(user, 'profile_picture', None)
    
    return jsonify({
        "id": user.id,
        "email": user.email,
        "username": username,
        "profile_picture": profile_picture
    }), 200

@profile_bp.route("/api/profile/<int:user_id>", methods=["PUT"])
def update_profile(user_id):
    """Update user profile information"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    
    # Update username if provided
    if 'username' in data:
        if hasattr(user, 'username'):
            user.username = data['username']
    
    # Update email if provided and not already taken
    if 'email' in data and data['email'] != user.email:
        existing_user = User.query.filter_by(email=data['email']).first()
        if existing_user:
            return jsonify({"error": "Email already in use"}), 400
        user.email = data['email']
    
    try:
        db.session.commit()
        
        # Handle case where new columns might not exist yet
        username = getattr(user, 'username', None)
        profile_picture = getattr(user, 'profile_picture', None)
        
        return jsonify({
            "message": "Profile updated successfully",
            "user": {
                "id": user.id,
                "email": user.email,
                "username": username,
                "profile_picture": profile_picture
            }
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to update profile"}), 500

@profile_bp.route("/api/profile/<int:user_id>/password", methods=["PUT"])
def change_password(user_id):
    """Change user password"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    data = request.get_json()
    current_password = data.get('current_password')
    new_password = data.get('new_password')
    
    if not current_password or not new_password:
        return jsonify({"error": "Current and new passwords are required"}), 400
    
    # Verify current password
    if not check_password_hash(user.password, current_password):
        return jsonify({"error": "Current password is incorrect"}), 401
    
    # Validate new password
    if len(new_password) < 6:
        return jsonify({"error": "New password must be at least 6 characters"}), 400
    
    try:
        user.password = generate_password_hash(new_password)
        db.session.commit()
        return jsonify({"message": "Password changed successfully"}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to change password"}), 500

@profile_bp.route("/api/profile/<int:user_id>/picture", methods=["POST"])
def upload_profile_picture(user_id):
    """Upload profile picture"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    
    if file and allowed_file(file.filename):
        try:
            # Create upload directory if it doesn't exist
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
            
            # Generate secure filename
            filename = secure_filename(f"user_{user_id}_{file.filename}")
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save file
            file.save(file_path)
            
            # Update user profile
            if hasattr(user, 'profile_picture'):
                user.profile_picture = f"/uploads/profile_pictures/{filename}"
            db.session.commit()
            
            # Return the updated user profile
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
        except Exception as e:
            return jsonify({"error": "Failed to upload profile picture"}), 500
    
    return jsonify({"error": "Invalid file type. Only PNG, JPG, JPEG, and GIF are allowed"}), 400

@profile_bp.route("/api/profile/<int:user_id>/picture", methods=["DELETE"])
def delete_profile_picture(user_id):
    """Delete profile picture"""
    user = User.query.get(user_id)
    if not user:
        return jsonify({"error": "User not found"}), 404
    
    try:
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
    except Exception as e:
        db.session.rollback()
        return jsonify({"error": "Failed to delete profile picture"}), 500
