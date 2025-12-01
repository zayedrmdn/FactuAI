"""
Authentication schemas for login and registration.
Defines the structure for auth-related API requests and responses.
"""
from typing import Optional, Dict, Any
from datetime import datetime


class LoginRequest:
    """Schema for login request payload."""
    
    def __init__(self, email: str, password: str):
        self.email = email.strip().lower() if email else ""
        self.password = password
    
    def validate(self) -> Optional[str]:
        """Validate the login request."""
        if not self.email:
            return "Email is required"
        
        if "@" not in self.email or "." not in self.email:
            return "Invalid email format"
        
        if not self.password:
            return "Password is required"
        
        if len(self.password) < 6:
            return "Password must be at least 6 characters"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'email': self.email,
            'password': self.password
        }


class RegisterRequest:
    """Schema for registration request payload."""
    
    def __init__(self, email: str, password: str, confirm_password: str):
        self.email = email.strip().lower() if email else ""
        self.password = password
        self.confirm_password = confirm_password
    
    def validate(self) -> Optional[str]:
        """Validate the registration request."""
        if not self.email:
            return "Email is required"
        
        if "@" not in self.email or "." not in self.email:
            return "Invalid email format"
        
        if not self.password:
            return "Password is required"
        
        if len(self.password) < 6:
            return "Password must be at least 6 characters"
        
        if self.password != self.confirm_password:
            return "Passwords do not match"
        
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'email': self.email,
            'password': self.password,
            'confirm_password': self.confirm_password
        }


class AuthResponse:
    """Schema for authentication response."""
    
    def __init__(
        self,
        success: bool,
        message: str,
        user_id: Optional[int] = None,
        email: Optional[str] = None,
        token: Optional[str] = None,
        timestamp: Optional[datetime] = None
    ):
        self.success = success
        self.message = message
        self.user_id = user_id
        self.email = email
        self.token = token
        self.timestamp = timestamp or datetime.utcnow()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        response = {
            'success': self.success,
            'message': self.message,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }
        
        if self.user_id is not None:
            response['user_id'] = self.user_id
        
        if self.email:
            response['email'] = self.email
        
        if self.token:
            response['token'] = self.token
        
        return response
