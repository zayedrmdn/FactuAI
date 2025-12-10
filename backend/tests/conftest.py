# 🚨 MUST BE FIRST
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

"""
Shared pytest fixtures and configuration for FactuAI backend tests.
"""
import pytest
import tempfile
from unittest.mock import Mock, patch
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from services import llm

# Import your app and db
from app import app
from database import db


@pytest.fixture
def client():
    """Flask test client fixture."""
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    
    with app.test_client() as client:
        with app.app_context():
            db.create_all()
            yield client


@pytest.fixture
def mock_llm_client():
    """Mock LLM client for testing."""
    mock = Mock()
    mock.generate_response.return_value = "Test response"
    mock.is_available.return_value = True
    mock.get_model_info.return_value = {"model": "test_model"}
    return mock


@pytest.fixture
def mock_classifier():
    """Mock classifier for testing."""
    mock = Mock()
    mock.predict.return_value = {
        "label": "True",
        "confidence": 0.85
    }
    mock.predict_batch.return_value = [
        {"label": "True", "confidence": 0.85}
    ]
    mock.get_supported_labels.return_value = ["True", "False", "Partially True"]
    mock.is_available.return_value = True
    mock.get_model_info.return_value = {"model": "test_classifier"}
    return mock


@pytest.fixture
def mock_search_client():
    """Mock search client for testing."""
    mock = Mock()
    mock.search.return_value = [
        {
            "title": "Test Article",
            "url": "https://example.com/test",
            "snippet": "Test snippet content"
        }
    ]
    mock.is_available.return_value = True
    mock.get_search_info.return_value = {"service": "test_search"}
    return mock


@pytest.fixture
def sample_fact_check_request():
    """Sample fact-check request data."""
    return {
        "text": "The Earth is round.",
        "enable_search": True,
        "max_search_results": 5
    }


@pytest.fixture
def sample_auth_request():
    """Sample authentication request data."""
    return {
        "email": "test@example.com",
        "password": "testpassword123"
    }


@pytest.fixture(scope="session")
def shared_llm():
    """Shared LLM client for tests."""
    llm.initialize()
    return Mock()  # Return mock for testing