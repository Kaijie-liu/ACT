#===- act/back_end/validation/__init__.py - Validation Module Init -------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025 ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
#===---------------------------------------------------------------------====#

"""
Validation Module for Experiment Reproducibility

This module provides:
- Seed management for reproducibility
- Experiment metadata tracking
- Bug injection (mutation) for testing verifier soundness

For two-level validation (Level 1 / Level 2), use:
- act.pipeline.verification.validate_verifier.VerificationValidator
- act.pipeline.verification.per_neuron_bounds.run_per_neuron_bounds_check

For device management, use:
- act.util.device_manager
"""

# Re-export device management from canonical location
from act.util.device_manager import (
    initialize_device,
    get_default_device,
    get_default_dtype,
    get_current_settings,
)

# Seed management and reproducibility tools
from .reproducibility import (
    set_all_seeds,
    derive_seed,
    hash_file,
    get_git_commit,
    get_environment_info,
    ExperimentMetadata,
    ReproducibleNetFactory,
    verify_reproducibility,
)

# Bug injection mutations (for testing verifier soundness)
from .mutations import (
    MutationType,
    MutationConfig,
    TFMutator,
    is_soundness_mutation,
    is_control_mutation,
    SOUNDNESS_MUTATIONS,
    CONTROL_MUTATIONS,
    ALL_MUTATIONS,
)

__all__ = [
    # Device management (from act.util.device_manager)
    "initialize_device",
    "get_default_device",
    "get_default_dtype",
    "get_current_settings",
    # Seed management
    "set_all_seeds",
    "derive_seed",
    "hash_file",
    "get_git_commit",
    "get_environment_info",
    "ExperimentMetadata",
    "ReproducibleNetFactory",
    "verify_reproducibility",
    # Bug injection mutations
    "MutationType",
    "MutationConfig",
    "TFMutator",
    "is_soundness_mutation",
    "is_control_mutation",
    "SOUNDNESS_MUTATIONS",
    "CONTROL_MUTATIONS",
    "ALL_MUTATIONS",
]
