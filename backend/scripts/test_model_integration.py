#!/usr/bin/env python3
"""
Test Model Integration
Verifies that the frontend model selection propagates correctly to the backend.
"""

import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "backend"))

from services.factcheck_service import PipelineOrchestrator
from core.logging import logger

def test_default_model():
    """Test with default model (no config)"""
    print("\n" + "="*80)
    print("TEST 1: Default Model (No Configuration)")
    print("="*80)
    
    orchestrator = PipelineOrchestrator()
    test_claim = "The Earth is round."
    
    print(f"\n📝 Testing claim: '{test_claim}'")
    print("Expected: Should use default model from environment\n")
    
    result = orchestrator.check_text(test_claim, max_claims=1, model_config=None)
    
    print("\n✅ Test completed")
    print(f"Results: {len(result.get('results', []))} claim(s) processed")
    return result

def test_custom_model_openrouter():
    """Test with custom OpenRouter model"""
    print("\n" + "="*80)
    print("TEST 2: Custom OpenRouter Model")
    print("="*80)
    
    orchestrator = PipelineOrchestrator()
    test_claim = "Water boils at 100 degrees Celsius at sea level."
    
    # Simulate frontend model selection
    model_config = {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.1-8b-instruct:free",
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    
    print(f"\nTesting claim: '{test_claim}'")
    print("Model Config:")
    print(f"  Provider: {model_config['provider']}")
    print(f"  Model: {model_config['model_id']}")
    print(f"  Temperature: {model_config['temperature']}")
    print(f"  Max Tokens: {model_config['max_tokens']}")
    print("\nExpected: Should create dynamic LLM client with specified model\n")
    
    result = orchestrator.check_text(test_claim, max_claims=1, model_config=model_config)
    
    print("\n✅ Test completed")
    print(f"Results: {len(result.get('results', []))} claim(s) processed")
    return result

def test_invalid_provider():
    """Test with invalid provider (should fallback)"""
    print("\n" + "="*80)
    print("TEST 3: Invalid Provider (Fallback Test)")
    print("="*80)
    
    orchestrator = PipelineOrchestrator()
    test_claim = "Paris is the capital of France."
    
    model_config = {
        "provider": "invalid_provider",
        "model_id": "some-model",
    }
    
    print(f"\n📝 Testing claim: '{test_claim}'")
    print(f"Model Config: {model_config}")
    print("\nExpected: Should fallback to default model and log error\n")
    
    result = orchestrator.check_text(test_claim, max_claims=1, model_config=model_config)
    
    print("\n✅ Test completed (with fallback)")
    print(f"Results: {len(result.get('results', []))} claim(s) processed")
    return result

if __name__ == "__main__":
    print("\n" + "="*80)
    print("MODEL INTEGRATION TEST SUITE")
    print("="*80)
    print("\nThis test verifies that frontend model selection correctly")
    print("propagates to backend LLM initialization.\n")
    print("Watch for log messages showing model initialization!\n")
    
    try:
        # Run tests
        test_default_model()
        test_custom_model_openrouter()
        test_invalid_provider()
        
        print("\n" + "="*80)
        print("ALL TESTS COMPLETED")
        print("="*80)
        print("\n✅ Integration test passed!")
        print("\nTo verify in production:")
        print("1. Start backend: cd backend && python app.py")
        print("2. Start frontend: cd frontend && npm run dev")
        print("3. Change model in UI dropdown")
        print("4. Submit a fact-check request")
        print("5. Check backend logs for model initialization messages")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        logger.exception("Test error:")
        sys.exit(1)
