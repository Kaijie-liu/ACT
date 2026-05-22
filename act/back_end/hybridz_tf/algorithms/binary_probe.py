#===- act/back_end/hybridz_tf/algorithms/binary_probe.py - Binary Probe Stub -====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Stub for the v8 eq_lagr binary-probe step in hz_routing.hz_apply_relu_v8.
#   The full RIIM + pairwise + LP singleton implementation was a research
#   artifact that regressed cifar wall by 10-15× per instance with no
#   verifiable benefit (LP-tight bounds already do the heavy lifting). The
#   current no-op preserves the v117/v118 baseline soundness contract
#   (444V+15A across 561 instances vs arXiv-2512.19007v1 GT, 0 violations).
#
#   Re-add the full implementation only if a per-layer benefit is
#   demonstrated on a held-out benchmark.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

from act.back_end.solver.solver_hz import HZono


def binary_probe(hz: HZono, **kwargs) -> HZono:
    """Stub: returns input unchanged. See module docstring."""
    return hz


def binary_probe_v8(hz: HZono, **kwargs) -> HZono:
    """Stub: returns input unchanged. See module docstring.

    Accepts (and ignores) v8 dispatch kwargs: timeout, max_pairs,
    enable_pairwise, pairwise_min_nb, pairwise_min_eq, pairwise_min_cooc,
    pairwise_time_cap, warmup_pairs.
    """
    return hz
