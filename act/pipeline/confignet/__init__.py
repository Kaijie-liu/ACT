#!/usr/bin/env python3
#===- act/pipeline/confignet/__init__.py - ConfigNet Public API ---------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   ConfigNet instance generation API (pipeline-side).
#   This package generates random MLP/CNN configs + robustness specs and can
#   build wrapped PyTorch models with embedded Input/Output specs.
#
# Notes:
#   - This package does NOT directly depend on ACT back_end.
#   - builders.py depends on act.front_end.verifiable_model wrapper layers.
#
#===---------------------------------------------------------------------===#

from .schema import (
    ModelFamily,
    MLPConfig,
    CNN2DConfig,
    TemplateConfig,
    InputSpecConfig,
    OutputSpecConfig,
    InstanceSpec,
    GeneratedInstance,
    ConfigNetConfig,
)

from .seeds import seed_everything, derive_seed
from .sampler import sample_instances
from .builders import build_wrapped_model, build_generated_instance

__all__ = [
    # schema
    "ModelFamily",
    "MLPConfig",
    "CNN2DConfig",
    "TemplateConfig",
    "InputSpecConfig",
    "OutputSpecConfig",
    "InstanceSpec",
    "GeneratedInstance",
    "ConfigNetConfig",
    # seeds
    "seed_everything",
    "derive_seed",
    # sampler/builders
    "sample_instances",
    "build_wrapped_model",
    "build_generated_instance",
    # input sampling (lazy)
    "make_feasible_input",
    "sample_feasible_inputs",
    # level1 (lazy)
    "run_level1_check",
    "run_level1_check_from_confignet",
]

# Lazy exports (avoid runpy double-import warning when running as -m)
# ----
def __getattr__(name: str):
    if name in ("make_feasible_input", "sample_feasible_inputs"):
        from . import input_sampling as _m
        return getattr(_m, name)
    if name in ("run_level1_check", "run_level1_check_from_confignet"):
        from . import level1_check as _m
        return getattr(_m, name)
    raise AttributeError(name)

def __dir__():
    return sorted(list(globals().keys()) + ["make_feasible_input","sample_feasible_inputs","run_level1_check","run_level1_check_from_confignet"])
