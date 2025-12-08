"""
Cleanup script for Phase 2 backend restructuring.

Safely deletes old folders that have been consolidated into the new structure.
"""

import os
import shutil
from pathlib import Path

# Folders to delete
FOLDERS_TO_DELETE = [
    "api",           # Replaced by routes/
    "core",          # Merged into utils/ and config.py
    "database",      # Replaced by database.py and models/
    "pipeline",      # Merged into factcheck/
    "services",      # Merged into factcheck/ and utils/
    "schemas",       # No longer used
]

# Files to delete
FILES_TO_DELETE = [
    "requirements-local.txt",  # Keep only requirements-core.txt
]

def main():
    backend_dir = Path(__file__).parent.parent
    print(f"Backend directory: {backend_dir}")
    print("\n" + "="*60)
    print("PHASE 2 CLEANUP - DELETING OLD FOLDERS")
    print("="*60 + "\n")
    
    deleted_folders = []
    deleted_files = []
    not_found = []
    errors = []
    
    # Delete folders
    print("Deleting old folders:")
    for folder_name in FOLDERS_TO_DELETE:
        folder_path = backend_dir / folder_name
        if folder_path.exists() and folder_path.is_dir():
            try:
                shutil.rmtree(folder_path)
                deleted_folders.append(folder_name)
                print(f"  ✅ Deleted: {folder_name}/")
            except Exception as e:
                errors.append(f"  ❌ Failed to delete {folder_name}/: {e}")
                print(errors[-1])
        else:
            not_found.append(folder_name)
            print(f"  ⚠️  Not found: {folder_name}/")
    
    # Delete files
    print("\nDeleting old files:")
    for file_name in FILES_TO_DELETE:
        file_path = backend_dir / file_name
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                deleted_files.append(file_name)
                print(f"  ✅ Deleted: {file_name}")
            except Exception as e:
                errors.append(f"  ❌ Failed to delete {file_name}: {e}")
                print(errors[-1])
        else:
            not_found.append(file_name)
            print(f"  ⚠️  Not found: {file_name}")
    
    # Summary
    print("\n" + "="*60)
    print("CLEANUP SUMMARY")
    print("="*60)
    print(f"Folders deleted: {len(deleted_folders)}")
    for folder in deleted_folders:
        print(f"  - {folder}/")
    
    print(f"\nFiles deleted: {len(deleted_files)}")
    for file in deleted_files:
        print(f"  - {file}")
    
    if not_found:
        print(f"\nNot found (already deleted or never existed): {len(not_found)}")
        for item in not_found:
            print(f"  - {item}")
    
    if errors:
        print(f"\n❌ ERRORS: {len(errors)}")
        for error in errors:
            print(error)
        return 1
    
    print("\n✅ Cleanup completed successfully!")
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
