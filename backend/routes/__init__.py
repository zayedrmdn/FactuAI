"""
Routes module initialization.
Exports all blueprint objects.
"""
from routes.factcheck import bp as factcheck_bp
from routes.auth import bp as auth_bp
from routes.profile import bp as profile_bp

__all__ = ['factcheck_bp', 'auth_bp', 'profile_bp']
