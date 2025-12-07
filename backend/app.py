from dotenv import load_dotenv
load_dotenv()

import os
import sys
print("CWD:", os.getcwd())
print("sys.path[0]:", sys.path[0])


try:
    from flask import Flask, request, jsonify, send_from_directory
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False
    # Create mock classes for testing
    class Flask:
        def __init__(self, name):
            # Mock constructor - no initialization needed for testing
            pass
        def run(self, **kwargs):
            # Mock run method - prints instead of starting server
            print("Mock Flask app would run here")
    class MockCORS:
        def __init__(self, app, resources=None):
            # Mock CORS - no configuration needed for testing
            pass
    CORS = MockCORS
    def request():
        # Mock request object - no implementation needed
        pass
    def jsonify(data):
        # Mock jsonify - returns data as-is for testing
        return data

from core.config import Config
from database.connection import db

# Import logger first
from core.logging import logger

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

# Initialize services
from services.service_manager import service_manager
service_manager.initialize_services()

# Import pipeline orchestrator
from services.factcheck_service import PipelineOrchestrator

if HAS_FLASK:
    # blueprints - only import if Flask is available
    try:
        from api import (
            auth_bp,
            profile_bp,
            bp_fact,
            bp_process,
            bp_image,
            bp_video,
        )
        HAS_ROUTES = True
    except ImportError as e:
        logger.error(f"Failed to import routes: {e}")
        HAS_ROUTES = False
else:
    HAS_ROUTES = False
app = Flask(__name__)

if HAS_FLASK:
    app.config.from_object(Config)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    # database
    db.init_app(app)
    with app.app_context():
        db.create_all()

# Initialize pipeline orchestrator instance
pipeline_orchestrator = PipelineOrchestrator()

if HAS_FLASK and HAS_ROUTES:
    # register routes
    app.register_blueprint(auth_bp)
    app.register_blueprint(profile_bp)
    app.register_blueprint(bp_fact)
    app.register_blueprint(bp_process)
    app.register_blueprint(bp_image)
    app.register_blueprint(bp_video)

if HAS_FLASK:
    @app.route('/uploads/<path:filename>')
    def uploaded_file(filename):
        """Serve uploaded files"""
        return send_from_directory('uploads', filename)

    @app.route('/api/validate', methods=['POST'])
    def validate_content():
        """Validate content using LLM before fact-checking"""
        try:
            data = request.get_json()
            text = data.get('text', '').strip()
            
            if not text:
                return jsonify({
                    "isValid": True,
                    "error": "",
                    "suggestion": ""
                })
            
            # Use existing LLM instance from service manager
            llm_client = service_manager.get_llm_client()
            result = llm_client.validate_content(text)
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Validation error: {e}")
            return jsonify({
                "isValid": False,
                "error": "Validation service temporarily unavailable",
                "suggestion": "Please try again in a moment."
            }), 500

if __name__ == "__main__":
    # Disable reloader to prevent duplicate model loading
    app.run(debug=True, use_reloader=False)