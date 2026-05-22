#===- tests/test_hz_eq_mask.py - eq_mask soundness regression test ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Regression test for the HZ eq_mask soundness fix.
#
#   Pre-fix bug: hz_apply_relu constructed the output HZono without passing
#   eq_mask, so the default-None ⇒ all-equalities semantics silently
#   re-typed any inequality rows the input HZ carried (notably the 2n box
#   clipping rows added by hz_intersect_box). The contradictory
#   z ≤ ub ∧ z ≥ lb pair became z = ub ∧ z = lb, the LP went infeasible,
#   and the verdict path returned "verified" on instances whose unsafe set
#   was actually reachable.
#
#   This test reconstructs the exact composition that triggered the bug:
#       intersect_box → hz_apply_relu(external_bounds) → check_unsafe_for_act
#   on a hand-built HZ with a known reachable unsafe point, and asserts the
#   final LP is FEASIBLE (i.e., the verifier does not wrongly verify).
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import numpy as np
import torch

from act.back_end.solver.solver_hz import (
    HZono,
    _eq_mask_of,
    check_unsafe_for_act,
)
from act.back_end.hybridz_tf.algorithms.bounds_tighten import hz_intersect_box
from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu


def _box_hz(lb, ub, *, dtype=torch.float64, device="cpu"):
    """Build an axis-aligned HZ from [lb, ub] (one diagonal continuous
    generator per dim, no binary, no constraints, eq_mask=None ⇒ all-eq).
    """
    lb = torch.tensor(lb, dtype=dtype, device=device).flatten()
    ub = torch.tensor(ub, dtype=dtype, device=device).flatten()
    n = int(lb.numel())
    c = ((lb + ub) / 2).view(n, 1)
    Gc = torch.diag((ub - lb) / 2)
    Gb = torch.zeros(n, 0, dtype=dtype, device=device)
    Ac = torch.zeros(0, n, dtype=dtype, device=device)
    Ab = torch.zeros(0, 0, dtype=dtype, device=device)
    b = torch.zeros(0, 1, dtype=dtype, device=device)
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=b)


class _UnsafeLinearAssert:
    """Minimal mock of an ASSERT layer carrying UNSAFE_LINEAR params."""

    def __init__(self, C, d):
        self.params = {
            "kind": "UNSAFE_LINEAR",
            "c": torch.tensor(C, dtype=torch.float64),
            "d": torch.tensor(d, dtype=torch.float64),
        }


def test_hz_apply_relu_preserves_eq_mask_explicit():
    """Direct: when input HZ has explicit eq_mask containing inequalities,
    hz_apply_relu must propagate it (not drop to None ⇒ all-equality).
    """
    # 2-D input HZ
    hz = _box_hz([-1.0, -1.0], [1.0, 1.0])

    # Tight pre-ReLU bounds (just the input box itself).
    lb = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    ub = torch.tensor([1.0, 1.0], dtype=torch.float64)
    # intersect_box appends 2*n = 4 inequality rows with eq_mask[new] = False.
    clipped = hz_intersect_box(hz, lb, ub)
    em_pre = _eq_mask_of(clipped)
    assert em_pre.numel() == 4, f"intersect_box should add 2n=4 rows; got {em_pre.numel()}"
    assert int(em_pre.sum().item()) == 0, "all 4 new rows must be inequalities (False)"

    # Apply ReLU. The fix: output HZ must keep those 4 rows as inequalities,
    # and add 3*k new equality rows for the k unstable neurons.
    out = hz_apply_relu(clipped, external_bounds=(lb, ub))
    em_post = _eq_mask_of(out)
    n_ineq_after = int((~em_post).sum().item())

    assert n_ineq_after == 4, (
        f"hz_apply_relu must preserve the 4 inequality rows from intersect_box; "
        f"got n_inequalities_after={n_ineq_after} (pre-fix bug: 0)"
    )


def test_hz_apply_relu_with_unsafe_set_lp_finds_reachable_point():
    """Soundness end-to-end: build a small HZ whose post-ReLU range is known
    to intersect an UNSAFE_LINEAR set (Y_0 = max). Pre-fix, LP would return
    'infeasible' (and the verifier would say 'verified'); post-fix LP must
    return 'feasible' since a reachable unsafe point genuinely exists.
    """
    # 2-D HZ where the actual reachable y after ReLU spans [0, 1] x [0, 1].
    hz = _box_hz([-0.5, -0.5], [1.0, 1.0])
    lb = torch.tensor([-0.5, -0.5], dtype=torch.float64)
    ub = torch.tensor([1.0, 1.0], dtype=torch.float64)
    clipped = hz_intersect_box(hz, lb, ub)
    out = hz_apply_relu(clipped, external_bounds=(lb, ub))

    # UNSAFE: Y_1 - Y_0 ≤ 0  (i.e., Y_0 is the max).
    # Reachable: at y = (1, 0), unsafe holds → LP must be feasible.
    assert_layer = _UnsafeLinearAssert(
        C=[[-1.0, 1.0]],   # row: -Y_0 + Y_1 ≤ 0
        d=[0.0],
    )

    status, witness = check_unsafe_for_act(out, assert_layer, timeout_s=10.0)

    assert status == "feasible", (
        f"LP must find the unsafe point (y=(1,0) is reachable + unsafe); "
        f"got status={status}. Pre-fix bug would return 'infeasible' here "
        f"because intersect_box's inequality rows were silently re-typed as "
        f"equalities."
    )
    assert witness is not None, "feasible status must yield a witness"


def test_hz_apply_relu_default_eq_mask_when_input_clean():
    """When input HZ has no inequality rows (clean ReLU encoding), the output
    should still be well-formed with all-equality rows. Smoke test that the
    fix doesn't break the no-intersect_box path.
    """
    hz = _box_hz([-1.0, -1.0], [1.0, 1.0])
    lb = torch.tensor([-1.0, -1.0], dtype=torch.float64)
    ub = torch.tensor([1.0, 1.0], dtype=torch.float64)
    out = hz_apply_relu(hz, external_bounds=(lb, ub))
    em = _eq_mask_of(out)
    # All new rows are equalities (the 3k linking rows per unstable neuron).
    # k=2 here (both [−1, 1] cross zero) ⇒ 6 equality rows.
    assert int(em.sum().item()) == em.numel(), (
        f"With no inequality input rows, output should be all equalities; got "
        f"{int(em.sum().item())}/{em.numel()}"
    )


if __name__ == "__main__":
    test_hz_apply_relu_preserves_eq_mask_explicit()
    test_hz_apply_relu_with_unsafe_set_lp_finds_reachable_point()
    test_hz_apply_relu_default_eq_mask_when_input_clean()
    print("OK: 3 eq_mask regression tests pass")
