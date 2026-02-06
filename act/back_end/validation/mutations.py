#===- act/back_end/validation/mutations.py - Bug Injection Mutations -----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025 ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
#===---------------------------------------------------------------------====#

"""
Bug Injection (Mutation) Module

Injects known defects into transfer functions to evaluate the detection
capability of Level 1 (SCC) and Level 2 (BCA).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable, Optional, Dict, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from act.back_end.core import Bounds


class MutationType(Enum):
    """Mutation type enumeration."""
    M1_TIGHTEN = "tighten_bounds"        # Tighten bounds (soundness violation)
    M2_LOOSEN = "loosen_bounds"          # Loosen bounds (negative control)
    M3_SWAP = "swap_lb_ub"               # Swap lower/upper bounds (severe violation)
    M4_ZERO_LB = "zero_lower_bound"      # Zero out lower bound (soundness violation)
    M5_SCALE_UB = "scale_upper_bound"    # Scale upper bound (soundness violation)
    M6_NOISE = "add_noise"               # Add random noise (random violation)


@dataclass
class MutationConfig:
    """Mutation configuration."""
    mutation_type: MutationType
    target_layer_id: Optional[int] = None  # None means all layers
    factor: float = 0.1
    seed: Optional[int] = None

    def to_dict(self) -> dict:
        return {
            "mutation_type": self.mutation_type.value,
            "target_layer_id": self.target_layer_id,
            "factor": self.factor,
            "seed": self.seed,
        }


class TFMutator:
    """
    Transfer Function Mutator

    Applies mutations to bounds to simulate transfer function bugs.
    """

    @staticmethod
    def M1_tighten_bounds(bounds: "Bounds", factor: float = 0.1) -> "Bounds":
        """
        M1: Tighten bounds - reduce bound width.

        This causes soundness violations: concrete values may fall
        outside the tightened bounds.

        Expected: Level 2 should detect violations.
        """
        from act.back_end.core import Bounds as BoundsClass

        width = bounds.ub - bounds.lb
        shrink = width * factor
        return BoundsClass(
            lb=bounds.lb + shrink,
            ub=bounds.ub - shrink
        )

    @staticmethod
    def M2_loosen_bounds(bounds: "Bounds", factor: float = 0.1) -> "Bounds":
        """
        M2: Loosen bounds - increase bound width.

        This is a negative control: should NOT cause soundness violations.

        Expected: Neither Level 1 nor Level 2 should report errors.
        """
        from act.back_end.core import Bounds as BoundsClass

        width = bounds.ub - bounds.lb
        expand = width * factor
        return BoundsClass(
            lb=bounds.lb - expand,
            ub=bounds.ub + expand
        )

    @staticmethod
    def M3_swap_lb_ub(bounds: "Bounds") -> "Bounds":
        """
        M3: Swap lower and upper bounds.

        Severe soundness violation: lb > ub.

        Expected: Level 2 should immediately detect this.
        """
        from act.back_end.core import Bounds as BoundsClass

        return BoundsClass(lb=bounds.ub.clone(), ub=bounds.lb.clone())

    @staticmethod
    def M4_zero_lower_bound(bounds: "Bounds") -> "Bounds":
        """
        M4: Zero out lower bound.

        Causes soundness violations for negative activations.

        Expected: Level 2 detects when negative activations exist.
        """
        from act.back_end.core import Bounds as BoundsClass

        return BoundsClass(
            lb=torch.zeros_like(bounds.lb),
            ub=bounds.ub.clone()
        )

    @staticmethod
    def M5_scale_upper_bound(bounds: "Bounds", factor: float = 0.5) -> "Bounds":
        """
        M5: Scale upper bound.

        May cause upper bound to be too tight.

        Expected: Level 2 detects when activations exceed scaled upper bound.
        """
        from act.back_end.core import Bounds as BoundsClass

        return BoundsClass(
            lb=bounds.lb.clone(),
            ub=bounds.ub * factor
        )

    @staticmethod
    def M6_add_noise(bounds: "Bounds", std: float = 0.1, seed: Optional[int] = None) -> "Bounds":
        """
        M6: Add random noise to bounds.

        Random soundness violations.

        Expected: Level 2 may detect depending on noise direction.
        """
        from act.back_end.core import Bounds as BoundsClass

        if seed is not None:
            torch.manual_seed(seed)

        noise_lb = torch.randn_like(bounds.lb) * std
        noise_ub = torch.randn_like(bounds.ub) * std

        return BoundsClass(
            lb=bounds.lb + noise_lb,
            ub=bounds.ub + noise_ub
        )

    @classmethod
    def get_mutation_fn(cls, mutation_type: MutationType) -> Callable:
        """Get mutation function by type."""
        mapping = {
            MutationType.M1_TIGHTEN: cls.M1_tighten_bounds,
            MutationType.M2_LOOSEN: cls.M2_loosen_bounds,
            MutationType.M3_SWAP: cls.M3_swap_lb_ub,
            MutationType.M4_ZERO_LB: cls.M4_zero_lower_bound,
            MutationType.M5_SCALE_UB: cls.M5_scale_upper_bound,
            MutationType.M6_NOISE: cls.M6_add_noise,
        }
        return mapping[mutation_type]

    @classmethod
    def apply_mutation(
        cls,
        bounds: "Bounds",
        config: MutationConfig
    ) -> "Bounds":
        """Apply mutation configuration to bounds."""
        fn = cls.get_mutation_fn(config.mutation_type)

        if config.mutation_type == MutationType.M6_NOISE:
            return fn(bounds, std=config.factor, seed=config.seed)
        elif config.mutation_type in [MutationType.M3_SWAP, MutationType.M4_ZERO_LB]:
            return fn(bounds)
        else:
            return fn(bounds, factor=config.factor)

    @classmethod
    def mutate_layer_bounds(
        cls,
        layer_bounds: Dict[int, "Bounds"],
        config: MutationConfig
    ) -> Dict[int, "Bounds"]:
        """
        Apply mutation to layer bounds.

        Args:
            layer_bounds: Original bounds per layer
            config: Mutation configuration

        Returns:
            Mutated bounds (new dict, original unchanged)
        """
        mutated = {}

        for layer_id, bounds in layer_bounds.items():
            if config.target_layer_id is None or config.target_layer_id == layer_id:
                # Apply mutation
                mutated[layer_id] = cls.apply_mutation(bounds, config)
            else:
                # Keep original (clone to avoid side effects)
                from act.back_end.core import Bounds as BoundsClass
                mutated[layer_id] = BoundsClass(
                    lb=bounds.lb.clone(),
                    ub=bounds.ub.clone()
                )

        return mutated


def is_soundness_mutation(mutation_type: MutationType) -> bool:
    """Check if mutation is expected to cause soundness violations."""
    return mutation_type in {
        MutationType.M1_TIGHTEN,
        MutationType.M3_SWAP,
        MutationType.M4_ZERO_LB,
        MutationType.M5_SCALE_UB,
        MutationType.M6_NOISE,
    }


def is_control_mutation(mutation_type: MutationType) -> bool:
    """Check if mutation is a negative control (should not cause violations)."""
    return mutation_type == MutationType.M2_LOOSEN


# Convenience lists
SOUNDNESS_MUTATIONS = [
    MutationType.M1_TIGHTEN,
    MutationType.M3_SWAP,
    MutationType.M4_ZERO_LB,
    MutationType.M5_SCALE_UB,
    MutationType.M6_NOISE,
]

CONTROL_MUTATIONS = [
    MutationType.M2_LOOSEN,
]

ALL_MUTATIONS = SOUNDNESS_MUTATIONS + CONTROL_MUTATIONS
