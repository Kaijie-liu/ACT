"""ACT pipeline verification helpers.

Keep this package initializer lightweight and import optional submodules only
when callers request their symbols.
"""

from importlib import import_module
from typing import Any

_SUBMODULES = {
    "torch2act",
    "act2torch",
    "validate_verifier",
    "model_factory",
    "utils",
    "llm_probe",
}

_SYMBOL_TO_MODULE = {
    "TorchToACT": "torch2act",
    "build_act": "torch2act",
    "ACTToTorch": "act2torch",
    "ActGraphModule": "act2torch",
    "VerificationValidator": "validate_verifier",
    "ModelFactory": "model_factory",
    "PerformanceMetrics": "utils",
    "ParallelResult": "utils",
    "PerformanceProfiler": "utils",
    "ParallelExecutor": "utils",
    "ProgressTracker": "utils",
    "print_memory_usage": "utils",
    "clear_torch_cache": "utils",
    "setup_logging": "utils",
    "retry_on_failure": "utils",
    "timeout_handler": "utils",
}

__all__ = sorted(_SUBMODULES | set(_SYMBOL_TO_MODULE))


def __getattr__(name: str) -> Any:
    if name in _SUBMODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is not None:
        module = import_module(f"{__name__}.{module_name}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
