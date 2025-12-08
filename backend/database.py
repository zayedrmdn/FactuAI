"""
Database configuration and connection for FactuAI Backend

Consolidated from:
- database/connection.py
- database/__init__.py
"""

try:
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()
except ImportError:
    # Create a mock DB for testing when flask_sqlalchemy is not available
    class MockDB:
        def init_app(self, app):
            pass
        
        def create_all(self):
            pass
    
    db = MockDB()


__all__ = ["db"]
