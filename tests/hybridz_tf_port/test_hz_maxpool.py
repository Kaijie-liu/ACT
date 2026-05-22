#===- tests/hybridz_tf_port/test_hz_maxpool.py - HZ max-pool 2D ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Soundness checks for ``hz_maxpool2d``:
#     (1) when each pooling window has a stable winner, the output HZ
#         keeps that single source row intact (correlation preserved);
#     (2) when a window is unstable, the fallback ``[max(lb), max(ub)]``
#         envelope soundly contains the true max for sampled inputs;
#     (3) eq_mask passes through unchanged.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    HZono, hz_from_bounds, hz_compute_bounds, _eq_mask_of,
)
from act.back_end.hybridz_tf.tf_cnn import hz_maxpool2d


def test_stable_winner_keeps_row_exactly():
    """A 1x4 input where position 0 dominates all others: max-pool 1x4
    should reproduce position 0's row in the output HZ.
    """
    # 4 inputs, each a unit-radius interval around (4.0, 0.0, 0.0, 0.0).
    # Position 0 lb=3 > others ub=1, so the winner is stable.
    c = torch.tensor([[4.0], [0.0], [0.0], [0.0]], dtype=torch.float64)
    Gc = torch.eye(4, dtype=torch.float64)
    Gb = torch.zeros(4, 0, dtype=torch.float64)
    Ac = torch.zeros(0, 4, dtype=torch.float64)
    Ab = torch.zeros(0, 0, dtype=torch.float64)
    b = torch.zeros(0, 1, dtype=torch.float64)
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)
    out = hz_maxpool2d(
        hz, kernel_size=(1, 4), stride=(1, 4), padding=0,
        input_shape=(1, 1, 4),
    )
    # Output dim is 1; should be exactly position 0's row (c=4, generator=1
    # on the first generator), no extra generators added.
    assert int(out.c.shape[0]) == 1
    assert float(out.c[0, 0].item()) == 4.0
    # ng stays at the input's 4 (stable winner does not add a fresh generator).
    assert int(out.Gc.shape[1]) == 4


def test_unstable_block_falls_back_to_envelope():
    """All four inputs centered at 0 with radius 1: no stable winner; the
    output box must contain the true max of sampled inputs.
    """
    c = torch.zeros(4, 1, dtype=torch.float64)
    Gc = torch.eye(4, dtype=torch.float64)
    hz = HZono(
        c=c, Gc=Gc,
        Gb=torch.zeros(4, 0, dtype=torch.float64),
        Ac=torch.zeros(0, 4, dtype=torch.float64),
        Ab=torch.zeros(0, 0, dtype=torch.float64),
        b=torch.zeros(0, 1, dtype=torch.float64),
    )
    out = hz_maxpool2d(
        hz, kernel_size=(1, 4), stride=(1, 4), padding=0,
        input_shape=(1, 1, 4),
    )
    # Sample concrete points and check containment.
    samples = -1.0 + 2.0 * torch.rand(256, 4, dtype=torch.float64)
    true_max, _ = samples.max(dim=1)
    bds = hz_compute_bounds(out)
    lb = float(bds.lb.flatten()[0].item())
    ub = float(bds.ub.flatten()[0].item())
    assert torch.all(true_max <= ub + 1e-9), \
        f"true max {float(true_max.max())} > HZ ub {ub} (unsound)"
    assert torch.all(true_max >= lb - 1e-9), \
        f"true max {float(true_max.min())} < HZ lb {lb} (unsound)"


def test_eq_mask_passes_through():
    """An input HZ carrying an explicit inequality row must keep that row
    typed as inequality after max-pool.
    """
    n = 4
    c = torch.zeros(n, 1, dtype=torch.float64)
    Gc = torch.eye(n, dtype=torch.float64)
    Gb = torch.zeros(n, 0, dtype=torch.float64)
    # One trivially-inactive row: Gc·ξ ≤ 10 (always satisfied).
    Ac = torch.zeros(1, n, dtype=torch.float64); Ac[0, 0] = 1.0
    Ab = torch.zeros(1, 0, dtype=torch.float64)
    b = torch.tensor([[10.0]], dtype=torch.float64)
    em = torch.tensor([False])
    hz = HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b, eq_mask=em)
    out = hz_maxpool2d(
        hz, kernel_size=(1, 4), stride=(1, 4), padding=0,
        input_shape=(1, 1, 4),
    )
    em_out = _eq_mask_of(out)
    assert em_out.numel() == 1
    assert bool(em_out[0].item()) is False, \
        "max-pool must keep caller's inequality row typed False"


if __name__ == "__main__":
    test_stable_winner_keeps_row_exactly()
    test_unstable_block_falls_back_to_envelope()
    test_eq_mask_passes_through()
    print("OK: hz_maxpool2d stable-winner + envelope + eq_mask")
