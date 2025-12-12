import os
import sys

# Ensure backend/ is on sys.path so `import app` works in tests.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
