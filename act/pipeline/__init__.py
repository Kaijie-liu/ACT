#===- act/pipeline/__init__.py - ACT Pipeline Module -------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025- ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#

"""ACT pipeline convenience exports.

The package initializer stays lightweight so command modules such as the
HybridZ benchmark runner do not import conversion, validation, LLM, or optional
solver-diagnostic code unless callers explicitly request those symbols.
"""

from importlib import import_module
from typing import Any

_SYMBOL_TO_MODULE = {
    "ModelFactory": "act.pipeline.verification.model_factory",
    "TorchToACT": "act.pipeline.verification.torch2act",
    "PerformanceProfiler": "act.pipeline.verification.utils",
    "ParallelExecutor": "act.pipeline.verification.utils",
    "print_memory_usage": "act.pipeline.verification.utils",
    "clear_torch_cache": "act.pipeline.verification.utils",
    "setup_logging": "act.pipeline.verification.utils",
    "ProgressTracker": "act.pipeline.verification.utils",
}

__all__ = [
    "ModelFactory",
    "TorchToACT",
    "PerformanceProfiler",
    "ParallelExecutor",
    "print_memory_usage",
    "clear_torch_cache",
    "setup_logging",
    "ProgressTracker",
    "UTILS_AVAILABLE",
]


def __getattr__(name: str) -> Any:
    if name == "UTILS_AVAILABLE":
        try:
            import_module("act.pipeline.verification.utils")
            value = True
        except ImportError:
            value = False
        globals()[name] = value
        return value
    module_name = _SYMBOL_TO_MODULE.get(name)
    if module_name is not None:
        try:
            module = import_module(module_name)
            value = getattr(module, name)
        except ImportError:
            if module_name.endswith(".utils"):
                value = None
            else:
                raise
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
