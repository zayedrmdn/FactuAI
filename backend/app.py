"""
FactuAI Backend Application

Main Flask application entry point.
"""

from dotenv import load_dotenv
import os

# Load environment from project root and backend folder to ensure API keys resolve
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
ROOT_ENV = os.path.join(BASE_DIR, ".env")
BACKEND_ENV = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(ROOT_ENV)
load_dotenv(BACKEND_ENV)
from flask import Flask, send_from_directory
from flask_cors import CORS

# Import configuration and database
from config import Config
from database import db

# Import logger
from utils.logging import logger

# Log startup configuration
logger.info("="*60)
logger.info("FACTUAI BACKEND - STARTING UP")
logger.info("="*60)
logger.info(f"[CONFIG] Run Mode: {os.getenv('APP_RUN_MODE', 'cloud')}")
logger.info(f"[CONFIG] LLM Provider: {os.getenv('LLM_PROVIDER', 'openrouter')}")
logger.info(f"[CONFIG] OpenRouter Model: {os.getenv('OPENROUTER_MODEL', 'not set')}")
logger.info(f"[CONFIG] NVIDIA Model: {os.getenv('NVIDIA_MODEL', 'not set')}")
logger.info(f"[CONFIG] Database: {'Connected' if os.getenv('DB_URI') else 'Not configured'}")
logger.info("="*60)

# Create Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Enable CORS
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Initialize database
db.init_app(app)

# Create tables and initialize everything within app context
with app.app_context():
    # Initialize LLM providers
    import factcheck
    factcheck.initialize()
    
    # Import blueprints (within app context so models can be loaded)
    from routes.factcheck import bp as factcheck_bp
    from routes.auth import bp as auth_bp
    from routes.profile import bp as profile_bp
    
    # Register blueprints
    app.register_blueprint(factcheck_bp)
    app.register_blueprint(auth_bp)
    app.register_blueprint(profile_bp)
    
    # Create database tables
    db.create_all()


@app.route('/uploads/<path:filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory('uploads', filename)


if __name__ == "__main__":
    # Disable reloader to prevent duplicate model loading
    app.run(debug=True, use_reloader=False)