#===- tests/hybridz_tf_port/test_hz_smooth.py - smooth activations ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Soundness checks for the piecewise sigmoid/tanh HZ encoding:
#     (1) the output HZ over-approximates the true reachable set
#         (every concrete output lands within hz_compute_bounds),
#     (2) eq_mask is correctly propagated through hz_apply_piecewise
#         and hz_reduce.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import (
    HZono, hz_from_bounds, hz_compute_bounds, _eq_mask_of,
)
from act.back_end.hybridz_tf.tf_mlp import (
    hz_apply_sigmoid, hz_apply_tanh, hz_reduce,
)


def _containment(hz_in: HZono, hz_out: HZono, activation, *, n_samples: int = 256):
    """Verify every sampled concrete output lies in the operator's output box."""
    bounds_in = hz_compute_bounds(hz_in)
    lb_in = bounds_in.lb.flatten()
    ub_in = bounds_in.ub.flatten()
    n = lb_in.numel()
    samples = lb_in + (ub_in - lb_in) * torch.rand(n_samples, n, dtype=lb_in.dtype)
    true_out = activation(samples)
    bounds_out = hz_compute_bounds(hz_out)
    lb_out = bounds_out.lb.flatten()
    ub_out = bounds_out.ub.flatten()
    tol = 1e-6
    assert torch.all(true_out >= lb_out.unsqueeze(0) - tol), \
        "concrete output below HZ lower bound (unsound)"
    assert torch.all(true_out <= ub_out.unsqueeze(0) + tol), \
        "concrete output above HZ upper bound (unsound)"


def test_sigmoid_contains_concrete():
    hz = hz_from_bounds(
        Bounds(lb=torch.tensor([-2.0, 0.0, 1.5]),
               ub=torch.tensor([2.0, 1.5, 3.0])),
        torch.float64, "cpu",
    )
    out = hz_apply_sigmoid(hz, K=2)
    _containment(hz, out, torch.sigmoid)


def test_tanh_contains_concrete():
    hz = hz_from_bounds(
        Bounds(lb=torch.tensor([-2.0, -0.5, 0.0]),
               ub=torch.tensor([2.0, 0.5, 1.5])),
        torch.float64, "cpu",
    )
    out = hz_apply_tanh(hz, K=2)
    _containment(hz, out, torch.tanh)


def test_sigmoid_eq_mask_propagates_through_reduce():
    """hz_apply_sigmoid emits an all-equality eq_mask; hz_reduce must keep it."""
    hz = hz_from_bounds(
        Bounds(lb=torch.full((4,), -2.0),
               ub=torch.full((4,), 2.0)),
        torch.float64, "cpu",
    )
    out = hz_apply_sigmoid(hz, K=2)
    em = _eq_mask_of(out)
    assert int(em.sum().item()) == em.numel(), "sigmoid output should be all-equality"
    reduced = hz_reduce(out, max_order=1.5)  # force order reduction
    em_r = _eq_mask_of(reduced)
    assert int(em_r.sum().item()) == em_r.numel(), \
        "hz_reduce must keep equality rows tagged True"


def test_tanh_batch_shape_handled():
    """Smoke: tanh on a B=2 stacked HZ (leading 2*n dim) returns matching shape."""
    n_per_batch = 3
    hz = hz_from_bounds(
        Bounds(lb=torch.full((2 * n_per_batch,), -1.0),
               ub=torch.full((2 * n_per_batch,), 1.0)),
        torch.float64, "cpu",
    )
    out = hz_apply_tanh(hz, K=2)
    assert out.c.shape[0] == 2 * n_per_batch


if __name__ == "__main__":
    test_sigmoid_contains_concrete()
    test_tanh_contains_concrete()
    test_sigmoid_eq_mask_propagates_through_reduce()
    test_tanh_batch_shape_handled()
    print("OK: sigmoid/tanh containment + eq_mask propagation")
