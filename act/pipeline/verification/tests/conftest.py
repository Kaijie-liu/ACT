# conftest.py
import sys
from pathlib import Path

# Ensure repository root is on sys.path when running pytest from repo root.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
