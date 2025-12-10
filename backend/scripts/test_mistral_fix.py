"""
Test script to verify Mistral-7b-instruct fix
Tests the LLM client and pipeline with the problematic model
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from factcheck import llm_client, pipeline
from utils.logging import get_logger

logger = get_logger(__name__)

def test_llm_client_direct():
    """Test LLM client directly with Mistral-7b-instruct"""
    print("\n" + "="*60)
    print("TEST 1: Direct LLM Client Test")
    print("="*60)
    
    try:
        llm_client.initialize()
        
        system = "You are a fact-checking AI. Analyze claims objectively."
        user = "Is this claim true: 'COVID-19 vaccines contain microchips'?"
        
        print(f"\nModel: mistralai/mistral-7b-instruct:free")
        print(f"System: {system[:50]}...")
        print(f"User: {user}")
        print("\nSending request...")
        
        response = llm_client.chat(
            system=system,
            user=user,
            provider="openrouter",
            model_id="mistralai/mistral-7b-instruct:free",
            max_tokens=1000,
            temperature=0.3
        )
        
        print(f"\n✓ Response received: {len(response)} chars")
        print(f"Preview: {response[:200]}...")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_pipeline_verification():
    """Test pipeline verification with Mistral-7b-instruct"""
    print("\n" + "="*60)
    print("TEST 2: Pipeline Verification Test")
    print("="*60)
    
    try:
        claim = "COVID-19 vaccines contain microchips that track your location and thoughts."
        
        print(f"\nClaim: {claim}")
        print(f"Model: mistralai/mistral-7b-instruct:free")
        print("\nVerifying with evidence collection...")
        
        result = pipeline.verify_claim(
            claim=claim,
            llm="openrouter",
            model_id="mistralai/mistral-7b-instruct:free",
            num_google=5,
            num_news=5,
            top_k=10
        )
        
        print(f"\n✓ Verification complete")
        print(f"Verdict: {result.get('verdict', 'UNKNOWN')}")
        print(f"Confidence: {result.get('confidence', 0.0):.2%}")
        print(f"Reasoning: {result.get('reasoning', 'No reasoning')[:100]}...")
        print(f"Evidence items: {len(result.get('evidence', []))}")
        print(f"Sources: {len(result.get('sources', []))}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_summary_generation():
    """Test summary generation with Mistral-7b-instruct"""
    print("\n" + "="*60)
    print("TEST 3: Summary Generation Test")
    print("="*60)
    
    try:
        text = """COVID-19 vaccines contain microchips that track your location and thoughts. 
        This claim has been circulating on social media since early 2021."""
        
        print(f"\nText: {text[:100]}...")
        print(f"Model: mistralai/mistral-7b-instruct:free")
        print("\nGenerating summary...")
        
        summary = pipeline.summarize_input(
            text=text,
            llm="openrouter",
            model_id="mistralai/mistral-7b-instruct:free",
            max_tokens=500
        )
        
        print(f"\n✓ Summary generated: {len(summary)} chars")
        print(f"Summary: {summary}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("MISTRAL-7B-INSTRUCT FIX VERIFICATION")
    print("="*60)
    print("\nTesting enhanced error handling and retry logic...")
    
    results = {
        "Direct LLM Client": test_llm_client_direct(),
        "Pipeline Verification": test_pipeline_verification(),
        "Summary Generation": test_summary_generation()
    }
    
    print("\n" + "="*60)
    print("TEST RESULTS SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    print("\n" + "="*60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("The Mistral-7b-instruct fix is working correctly!")
    else:
        print("✗ SOME TESTS FAILED")
        print("Review the error messages above for details")
    print("="*60 + "\n")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
