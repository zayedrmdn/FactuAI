try:
    from flask_sqlalchemy import SQLAlchemy
    db = SQLAlchemy()
except ImportError:
    # Create a mock DB for testing when flask_sqlalchemy is not available
    class MockDB:
        def init_app(self, app):
            # Mock initialization - no database setup needed for testing
            pass
        def create_all(self):
            # Mock table creation - no schema needed for testing
            pass
    db = MockDB()
