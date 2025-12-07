#!/usr/bin/env python3
"""
FactuAI Structure Verification Script

Verifies that all key modules can be imported correctly after the refactor.
Run from backend directory: python scripts/verify_structure.py
Or from project root: python backend/scripts/verify_structure.py

Exit codes:
    0 - All imports successful (or only missing external dependencies)
    1 - Path/structure errors found
"""

import sys
import os
from pathlib import Path

# Add project root and backend to sys.path
# Script is now in backend/scripts/, so parent.parent is project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
BACKEND_ROOT = PROJECT_ROOT / "backend"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(BACKEND_ROOT))

# Change working directory to backend for relative imports
os.chdir(BACKEND_ROOT)

# Known external dependencies that may not be installed
EXTERNAL_DEPS = {
    "dotenv", "flask", "torch", "transformers", "PIL", "pytesseract",
    "newsapi", "resend", "itsdangerous", "sentence_transformers",
    "unsloth", "keybert", "nltk", "bs4", "requests", "googleapiclient"
}


def is_external_dep_error(error_msg: str) -> bool:
    """Check if the error is due to a missing external dependency."""
    error_lower = error_msg.lower()
    for dep in EXTERNAL_DEPS:
        if dep.lower() in error_lower:
            return True
    return False


def verify_import(module_path: str, class_name: str = None) -> tuple[bool, str]:
    """
    Attempt to import a module and optionally verify a class exists.
    
    Args:
        module_path: The full module path (e.g., 'services.search.google_search')
        class_name: Optional class name to verify exists in the module
    
    Returns:
        Tuple of (success, status) where status is 'ok', 'dep', or 'error'
    """
    try:
        module = __import__(module_path, fromlist=[class_name] if class_name else [])
        if class_name:
            if not hasattr(module, class_name):
                print(f"❌ {module_path}.{class_name} - Class not found in module")
                return False, "error"
            print(f"✅ {module_path}.{class_name} - Loaded")
        else:
            print(f"✅ {module_path} - Loaded")
        return True, "ok"
    except ModuleNotFoundError as e:
        error_msg = str(e)
        if is_external_dep_error(error_msg):
            display = f"{module_path}.{class_name}" if class_name else module_path
            missing_dep = error_msg.split()[-1].strip("'")
            print(f"⚠️  {display} - Skipped (missing dep: {missing_dep})")
            return True, "dep"
        print(f"❌ {module_path} - ModuleNotFoundError: {e}")
        return False, "error"
    except ImportError as e:
        error_msg = str(e)
        if is_external_dep_error(error_msg):
            display = f"{module_path}.{class_name}" if class_name else module_path
            print(f"⚠️  {display} - Skipped (missing dep)")
            return True, "dep"
        print(f"❌ {module_path} - ImportError: {e}")
        return False, "error"
    except Exception as e:
        print(f"❌ {module_path} - Unexpected error: {type(e).__name__}: {e}")
        return False, "error"


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("FactuAI Structure Verification")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Backend Root: {BACKEND_ROOT}")
    print("-" * 60)
    print()

    # Define modules to verify
    modules_to_verify = [
        # Core modules
        ("core.config", "Config"),
        ("core.logging", "logger"),
        ("core.exceptions", "FactuAIException"),
        ("core.helpers", None),
        
        # Services - Search
        ("services.search.google_search", "GoogleSearchClient"),
        ("services.search.builder", "SmartQueryBuilder"),
        ("services.search.base", "SearchInterface"),
        
        # Services - Classifier
        ("services.classifier.bert_classifier", "ClaimClassifier"),
        ("services.classifier.base", "ClassifierInterface"),
        ("services.classifier.llm_classifier", "LLMClassifier"),
        
        # Services - Intent Classifier
        ("services.classifier_intent.intent_parser", "detect_intent"),
        
        # Services - LLM
        ("services.llm.base", "BaseLLM"),
        ("services.llm.factory", "LLMFactory"),
        
        # Services - Main
        ("services.factcheck_service", "PipelineOrchestrator"),
        ("services.service_manager", "service_manager"),
        
        # Pipeline
        ("pipeline.config", "GOOGLE_API_KEY"),
        ("pipeline.orchestrator", None),
        
        # API Blueprints
        ("api.factcheck", "bp_fact"),
        ("api.process", "bp"),
        ("api.auth", "auth_bp"),
        ("api.profile", "profile_bp"),
        ("api.image", "bp_image"),
        ("api.video", "bp_video"),
        
        # Database
        ("database.connection", None),
        ("database.models.user", "User"),
    ]

    print("Verifying module imports...")
    print("-" * 60)

    success_count = 0
    skipped_count = 0
    failure_count = 0

    for module_path, class_name in modules_to_verify:
        success, status = verify_import(module_path, class_name)
        if success:
            if status == "dep":
                skipped_count += 1
            else:
                success_count += 1
        else:
            failure_count += 1

    print()
    print("-" * 60)
    print(f"Results: {success_count} passed, {skipped_count} skipped (missing deps), {failure_count} failed")
    print("=" * 60)

    if failure_count > 0:
        print("\n⚠️  Structure errors found. Please fix the import paths above.")
        return 1
    else:
        print("\n🎉 All import paths verified successfully!")
        if skipped_count > 0:
            print(f"   ({skipped_count} modules skipped due to missing external dependencies)")
            print("   Run with venv activated for full verification.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
