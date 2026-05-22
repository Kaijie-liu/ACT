#===- tests/hybridz_tf_port/test_hz_eq_mask.py - eq_mask preservation ====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later (AGPLv3+).
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
#===---------------------------------------------------------------------===#
#
# Purpose:
#   Pin the row-type contract on HZ operators: when the input HZono carries
#   inequality rows in its eq_mask, the operator must propagate those rows
#   (and any new rows it adds must be tagged with the correct type).
#
#   Background: hz_apply_relu used to construct the output HZono without
#   passing ``eq_mask=``, so the default ``None`` was interpreted as
#   "every row is an equality" and any caller-provided inequality rows
#   were silently re-typed. Downstream LP solves treated z ≤ ub and
#   z ≥ lb as z = ub AND z = lb (contradictory), causing wrong verdicts.
#   These tests pin the fix at the operator level so it cannot regress.
#
#===---------------------------------------------------------------------===#

from __future__ import annotations

import torch

from act.back_end.solver.solver_hz import HZono, _eq_mask_of
from act.back_end.hybridz_tf.tf_mlp import hz_apply_relu, hz_apply_leaky_relu


def _hz_with_box_inequalities(lb, ub, *, dtype=torch.float64):
    """Construct an HZ over [lb, ub] with 2n explicit ``≤`` rows.

    The continuous-generator block is the diagonal radius, and we attach
    the two row families ``+G ξ ≤ ub - c`` and ``-G ξ ≤ c - lb`` with
    eq_mask=False to model a HZ that has had box clipping applied.
    """
    lb = torch.tensor(lb, dtype=dtype).flatten()
    ub = torch.tensor(ub, dtype=dtype).flatten()
    n = int(lb.numel())
    c = ((lb + ub) / 2).view(n, 1)
    Gc = torch.diag((ub - lb) / 2)
    Gb = torch.zeros(n, 0, dtype=dtype)
    Ac = torch.cat([Gc, -Gc], dim=0)
    Ab = torch.zeros(2 * n, 0, dtype=dtype)
    rhs = torch.cat([ub.view(-1, 1) - c, c - lb.view(-1, 1)], dim=0)
    em = torch.zeros(2 * n, dtype=torch.bool)
    return HZono(c=c, Gc=Gc, Gb=Gb, Ac=Ac, Ab=Ab, b=rhs, eq_mask=em)


def test_relu_preserves_caller_inequalities():
    """If the input HZ carries inequality rows (eq_mask=False), hz_apply_relu
    must keep those rows tagged as inequalities in the output.
    """
    hz = _hz_with_box_inequalities([-1.0, -1.0], [1.0, 1.0])
    assert int((~_eq_mask_of(hz)).sum().item()) == 4

    out = hz_apply_relu(hz)
    em = _eq_mask_of(out)
    n_le = int((~em).sum().item())
    assert n_le == 4, (
        f"hz_apply_relu must keep the 4 caller-provided inequality rows; "
        f"got {n_le}. Pre-fix the default eq_mask=None re-typed them as "
        f"equalities."
    )


def test_relu_marks_new_rows_equality():
    """The 3k linking/graph rows hz_apply_relu adds per unstable neuron are
    equalities and must be tagged ``True`` in the output eq_mask.
    """
    hz = _hz_with_box_inequalities([-1.0, -1.0], [1.0, 1.0])
    out = hz_apply_relu(hz)
    em = _eq_mask_of(out)
    # 4 caller rows (False) + 3k new equality rows (True), k=2 → 6 True
    n_eq = int(em.sum().item())
    assert n_eq == 6, f"expected 6 new equality rows (k=2, 3 per unstable); got {n_eq}"


def test_leaky_relu_preserves_caller_inequalities():
    hz = _hz_with_box_inequalities([-1.0, -1.0], [1.0, 1.0])
    out = hz_apply_leaky_relu(hz, 0.1)
    em = _eq_mask_of(out)
    n_le = int((~em).sum().item())
    assert n_le == 4, (
        f"hz_apply_leaky_relu must keep caller inequality rows; got {n_le}"
    )


def test_relu_no_inequalities_in_no_inequalities_out():
    """When the input has no inequality rows (eq_mask=None), the output stays
    all-equality. This is the upstream-compatible path.
    """
    n = 2
    hz = HZono(
        c=torch.zeros(n, 1, dtype=torch.float64),
        Gc=torch.eye(n, dtype=torch.float64),
        Gb=torch.zeros(n, 0, dtype=torch.float64),
        Ac=torch.zeros(0, n, dtype=torch.float64),
        Ab=torch.zeros(0, 0, dtype=torch.float64),
        b=torch.zeros(0, 1, dtype=torch.float64),
    )
    out = hz_apply_relu(hz)
    em = _eq_mask_of(out)
    assert int(em.sum().item()) == em.numel(), (
        "all rows in output should be equalities when input had none"
    )


if __name__ == "__main__":
    test_relu_preserves_caller_inequalities()
    test_relu_marks_new_rows_equality()
    test_leaky_relu_preserves_caller_inequalities()
    test_relu_no_inequalities_in_no_inequalities_out()
    print("OK: hz_apply_relu / leaky preserve eq_mask")
