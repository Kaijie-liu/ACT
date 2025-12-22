from __future__ import annotations

#===- act/pipeline/__init__.py - ACT Pipeline Module -------------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Lightweight package init for ACT Pipeline. Exposes key classes lazily to
#   avoid importing torch/torchvision during argument parsing or sampling.
#===---------------------------------------------------------------------===#

import importlib
from typing import Any

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

_lazy = {
    "ModelFactory": "act.pipeline.verification.model_factory",
    "TorchToACT": "act.pipeline.verification.torch2act",
    "PerformanceProfiler": "act.pipeline.verification.utils",
    "ParallelExecutor": "act.pipeline.verification.utils",
    "print_memory_usage": "act.pipeline.verification.utils",
    "clear_torch_cache": "act.pipeline.verification.utils",
    "setup_logging": "act.pipeline.verification.utils",
    "ProgressTracker": "act.pipeline.verification.utils",
}


def __getattr__(name: str) -> Any:
    if name == "UTILS_AVAILABLE":
        try:
            importlib.import_module("act.pipeline.verification.utils")
            return True
        except ImportError:
            return False
    if name in _lazy:
        module = importlib.import_module(_lazy[name])
        if hasattr(module, name):
            return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
