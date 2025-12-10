"""
Script to clean up old factcheck folder after refactoring
Removes all old files and the factcheck directory itself
"""

import os
import shutil
import sys

# Add backend to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.logging import get_logger

logger = get_logger(__name__)

def main():
    """Clean up old factcheck folder"""
    backend_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    factcheck_path = os.path.join(backend_path, 'factcheck')
    
    print("=" * 60)
    print("FACTCHECK FOLDER CLEANUP")
    print("=" * 60)
    print(f"\nBackend path: {backend_path}")
    print(f"Factcheck path: {factcheck_path}")
    
    if not os.path.exists(factcheck_path):
        print("\n✓ Factcheck folder does not exist - nothing to clean up")
        return True
    
    print(f"\nFolder exists. Contents:")
    try:
        for root, dirs, files in os.walk(factcheck_path):
            level = root.replace(factcheck_path, '').count(os.sep)
            indent = ' ' * 2 * level
            print(f'{indent}{os.path.basename(root)}/')
            subindent = ' ' * 2 * (level + 1)
            for file in files:
                print(f'{subindent}{file}')
    except Exception as e:
        print(f"Error listing contents: {e}")
    
    print(f"\n⚠️  About to delete: {factcheck_path}")
    response = input("Continue? (yes/no): ").strip().lower()
    
    if response != 'yes':
        print("\n✗ Cleanup cancelled by user")
        return False
    
    try:
        shutil.rmtree(factcheck_path)
        print(f"\n✓ Successfully deleted: {factcheck_path}")
        logger.info(f"Deleted old factcheck folder: {factcheck_path}")
        return True
    except Exception as e:
        print(f"\n✗ Error deleting folder: {e}")
        logger.error(f"Failed to delete factcheck folder: {e}")
        return False


if __name__ == "__main__":
    success = main()
    print("=" * 60)
    if success:
        print("✓ CLEANUP COMPLETE")
    else:
        print("✗ CLEANUP FAILED")
    print("=" * 60)
    sys.exit(0 if success else 1)
