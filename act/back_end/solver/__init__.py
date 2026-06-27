#===- act/back_end/solver/__init__.py - Constraint Solvers --------------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Solvers for constraint satisfaction. Provides open-source solvers by
#   default; the Gurobi backend is loaded lazily only when requested.
#
#===---------------------------------------------------------------------===#

from .solver_base import Solver, SolverCaps, SolveStatus
from .solver_torchlp import TorchLPSolver
from .solver_hz import (
    HZSolver,
    HZono,
    hz_compute_bounds,
    hz_compute_lp_bounds,
    hz_fresh_col_ids,
    hz_split_constraints,
)
from .solver_dual import DualSolver, expand_bounds_dict

__all__ = [
    'Solver', 'SolverCaps', 'SolveStatus',
    'TorchLPSolver', 'GurobiSolver',
    'HZSolver', 'HZono', 'hz_compute_bounds', 'hz_compute_lp_bounds',
    'hz_fresh_col_ids', 'hz_split_constraints',
    'DualSolver', 'expand_bounds_dict',
]


def __getattr__(name):
    if name == "GurobiSolver":
        from .solver_gurobi import GurobiSolver

        return GurobiSolver
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
