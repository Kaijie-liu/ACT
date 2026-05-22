#===- act/back_end/hybridz_tf/algorithms/binary_probe.py - Binary Probe Hook ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Reserved hook for a binary-probe step in the v8 ReLU pipeline.
#   Currently no-op because no validated benefit has been demonstrated on
#   the bound-tightened HZ produced by the cascade.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from act.back_end.solver.solver_hz import HZono


def binary_probe(hz: HZono, **kwargs) -> HZono:
    """Reserved hook; currently no-op (returns input unchanged)."""
    return hz


def binary_probe_v8(hz: HZono, **kwargs) -> HZono:
    """Reserved v8 hook; currently no-op (returns input unchanged).

    Accepts and ignores v8 dispatch kwargs so the cons-walker's call site
    can pass parameters without raising.
    """
    return hz
