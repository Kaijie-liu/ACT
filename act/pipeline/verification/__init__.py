"""
ACT Pipeline Verification Module (import-light).

Exports are lazily imported to avoid pulling heavy dependencies (torch/torchvision)
when only the package is imported for argument parsing.
"""

import importlib
from typing import Any

__all__ = [
    "torch2act",
    "act2torch",
    "VerificationValidator",
    "model_factory",
    "utils",
    "llm_probe",
]

_lazy = {
    "torch2act": "act.pipeline.verification.torch2act",
    "act2torch": "act.pipeline.verification.act2torch",
    "VerificationValidator": "act.pipeline.verification.validate_verifier",
    "model_factory": "act.pipeline.verification.model_factory",
    "utils": "act.pipeline.verification.utils",
    "llm_probe": "act.pipeline.verification.llm_probe",
}


def __getattr__(name: str) -> Any:
    if name in _lazy:
        module = importlib.import_module(_lazy[name])
        return getattr(module, name) if hasattr(module, name) else module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
