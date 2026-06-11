# ===- act/back_end/fchz_tf/__init__.py - FCHZ Transfer Function ----====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# ===---------------------------------------------------------------------===#
#
# Purpose:
#   FCHZ (Forward Constrained Hybrid Zonotope) Transfer Function.
#
#   Strict P1-P5 forward HZ propagation:
#     P1: Forward only (no backward, no CROWN)
#     P2: No gradient (no PGD, no autograd)
#     P3: Open-source LP/MILP only (HiGHS/scipy or equivalent; no Gurobi)
#     P4: No input split (no BaB)
#     P5: No random certify (deterministic walker; ORT post-audit only)
#
#   Three soundness mechanisms (vs basic Z):
#     1. tail_radius: per-row independent box error (Add/Mul/Sigmoid-safe)
#     2. sparse-slack columns: G compression for deep CNN (cifar/tiny)
#     3. Sigmoid/Tanh analytical chord bound (replace sampling)
#
# ===---------------------------------------------------------------------===#

"""FCHZ transfer function module — strict P1-P5 forward hybrid zonotope."""

from .fchz_tf import FchzTF

__all__ = [
    "FchzTF",
]
