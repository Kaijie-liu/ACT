# act/pipeline/verification/tests/conftest.py
from __future__ import annotations

import sys
import warnings
from pathlib import Path

# Ensure repository root is on sys.path when running pytest from repo root.
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def pytest_configure(config):
    # Suppress known torchvision image extension warning without stubbing package
    warnings.filterwarnings(
        "ignore",
        message=r"Failed to load image Python extension",
        category=UserWarning,
        module="torchvision.io.image",
    )
