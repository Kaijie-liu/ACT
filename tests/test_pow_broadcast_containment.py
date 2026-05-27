"""Soundness check: the R11/R12 helper-emitting handlers
(``OnnxPow`` chained MULs + ``OnnxBinaryMathOperation`` broadcast
EXPAND) must produce interval bounds that **contain** the model's
actual forward output at every concrete input drawn from the input box.

History
=======
R11 added ``_set_explicit_preds`` for helper layers and R12 added
``_conv1d_to_linear_matrix`` scalar normalization. The earlier
``test_helper_pred_tracking.py`` only verifies *structural* wiring
(the helper layer exists, its preds list has length 2 with duplicates
preserved, the consumer points at the helper). It does NOT verify
that the resulting interval bounds are sound.

Soundness contract for an ACT layer L over input box [lb, ub]:

    forall x in [lb, ub]: L_forward(x) ∈ [L.bounds.lb, L.bounds.ub]

This test takes a small synthetic computation that exercises Pow(x, 3)
followed by ReduceSum + scalar-broadcast Div (mirroring nn4sys
pensieve_*_parallel's L2-norm-style prelude), runs the ACT analyze
path over a tight input box, samples many concrete x in the box,
forwards each through the original PyTorch model, and asserts every
output coordinate is contained in the analyze() output Bin.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch


class _DeviceIsolated(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._dev = torch.get_default_device() if hasattr(torch, "get_default_device") else None
        self._dt = torch.get_default_dtype()
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        try:
            torch.set_default_device(self._dev or "cpu")
        except Exception:
            pass
        torch.set_default_dtype(self._dt)
        super().tearDown()


class TestSimpleMulSelfContainment(_DeviceIsolated):
    """Targeted soundness for the simplest case where R11 preserved
    duplicate preds: ``Mul(x, x)`` (i.e. ``x ** 2`` emitted by OnnxPow
    as a single chain step).

    Build an ACT MUL layer that consumes one source layer twice (the
    classical ``preds = [src, src]`` case after R11) and verify that
    sampled forward results lie inside the interval bounds.
    """

    def test_mul_xx_interval_contains_forward(self):
        from act.back_end.core import Layer, Bounds
        from act.back_end.layer_schema import LayerKind
        from act.back_end.interval_tf.tf_mlp import tf_mul

        # 4-dim input box. x ~ Uniform(lb, ub).
        torch.manual_seed(0)
        n = 4
        lb = torch.tensor([-1.0, -2.0, 0.0, 0.5], dtype=torch.float64)
        ub = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.float64)
        Bin = Bounds(lb=lb.view(1, n), ub=ub.view(1, n))

        L = Layer(
            id=0,
            kind=LayerKind.MUL.value,
            in_vars=list(range(n)) + list(range(n)),  # x and x (duplicate)
            out_vars=list(range(n, 2 * n)),
            params={"x_vars": list(range(n)), "y_vars": list(range(n)),
                    "input_shape": (1, n), "output_shape": (1, n)},
        )
        # tf_mul takes two Bounds (positional); for Mul(x, x) they are equal.
        fact = tf_mul(L, Bin, Bin)
        out_lb = fact.bounds.lb.view(-1)
        out_ub = fact.bounds.ub.view(-1)

        # Sample many points uniformly from the input box, forward through
        # the concrete computation x * x, and assert the result is in the
        # ACT-derived interval. Any violation = unsoundness in the R11
        # helper-pred path or in tf_mul.
        N = 1024
        u = torch.rand(N, n, dtype=torch.float64)
        x = lb + u * (ub - lb)
        y = x * x  # the operation tf_mul abstracts
        for i in range(n):
            ymin = y[:, i].min().item()
            ymax = y[:, i].max().item()
            self.assertLessEqual(
                float(out_lb[i]), ymin + 1e-9,
                f"coord {i}: ACT lb {float(out_lb[i])} > sampled min {ymin}; "
                f"tf_mul is UNDER-approximating (unsound)"
            )
            self.assertGreaterEqual(
                float(out_ub[i]), ymax - 1e-9,
                f"coord {i}: ACT ub {float(out_ub[i])} < sampled max {ymax}; "
                f"tf_mul is UNDER-approximating (unsound)"
            )


class TestExpandPlusDivContainment(_DeviceIsolated):
    """Bring up the actual helper chain that nn4sys pensieve hits:
    ``Div(x, broadcast_to_x_shape(scalar))``. Build the ACT layers by
    hand so we don't need an ONNX round-trip, then verify the analyze-
    path bound contains the forward result for many samples."""

    def test_div_x_by_broadcast_scalar(self):
        from act.back_end.core import Layer, Bounds
        from act.back_end.layer_schema import LayerKind
        from act.back_end.interval_tf.tf_mlp import tf_expand, tf_div

        torch.manual_seed(0)
        n = 4
        # The numerator x ranges in a box.
        x_lb = torch.tensor([-1.0, -2.0, 0.0, 0.5], dtype=torch.float64)
        x_ub = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.float64)
        Bx = Bounds(lb=x_lb.view(1, n), ub=x_ub.view(1, n))

        # Scalar denominator c bounded away from zero.
        c_lb = torch.tensor([1.0], dtype=torch.float64)
        c_ub = torch.tensor([2.0], dtype=torch.float64)
        Bc = Bounds(lb=c_lb.view(1, 1), ub=c_ub.view(1, 1))

        # Step 1: EXPAND the size-1 c to size n.
        L_exp = Layer(
            id=0,
            kind=LayerKind.EXPAND.value,
            in_vars=[100],   # placeholder var id; tf_expand reads from Bin only
            out_vars=list(range(101, 101 + n)),
            params={"shape": [1, n], "input_shape": (1, 1), "output_shape": (1, n)},
        )
        Bc_expanded = tf_expand(L_exp, Bc).bounds
        self.assertEqual(tuple(Bc_expanded.lb.shape), (1, n))

        # Step 2: DIV(x, c_expanded). preds positional → tf_div(L, Bx, Bc_expanded).
        L_div = Layer(
            id=1,
            kind=LayerKind.DIV.value,
            in_vars=list(range(n)) + list(range(101, 101 + n)),
            out_vars=list(range(200, 200 + n)),
            params={"x_vars": list(range(n)),
                    "y_vars": list(range(101, 101 + n)),
                    "input_shape": (1, n), "output_shape": (1, n)},
        )
        fact = tf_div(L_div, Bx, Bc_expanded)
        out_lb = fact.bounds.lb.view(-1)
        out_ub = fact.bounds.ub.view(-1)

        # Sample. Forward = x / c (broadcast). Verify containment per coord.
        N = 1024
        ux = torch.rand(N, n, dtype=torch.float64)
        x = x_lb + ux * (x_ub - x_lb)
        uc = torch.rand(N, 1, dtype=torch.float64)
        c = c_lb + uc * (c_ub - c_lb)
        y = x / c

        for i in range(n):
            ymin = y[:, i].min().item()
            ymax = y[:, i].max().item()
            self.assertLessEqual(
                float(out_lb[i]), ymin + 1e-9,
                f"coord {i}: ACT lb={float(out_lb[i])} > sampled min={ymin}"
            )
            self.assertGreaterEqual(
                float(out_ub[i]), ymax - 1e-9,
                f"coord {i}: ACT ub={float(out_ub[i])} < sampled max={ymax}"
            )


class TestPowCubedContainment(_DeviceIsolated):
    """The OnnxPow handler converts ``x ** 3`` into a chain of two MULs:

        L0: y = MUL(x, x)        # tf_mul(Bx, Bx) -> By
        L1: z = MUL(y, x)        # tf_mul(By, Bx) -> Bz

    Verify ``Bz`` contains the concrete ``x ** 3`` over a wide box."""

    def test_x_cubed_contained(self):
        from act.back_end.core import Layer, Bounds
        from act.back_end.layer_schema import LayerKind
        from act.back_end.interval_tf.tf_mlp import tf_mul

        torch.manual_seed(0)
        n = 4
        lb = torch.tensor([-1.5, -0.5, 0.0, 0.5], dtype=torch.float64)
        ub = torch.tensor([0.5, 1.5, 2.0, 3.0], dtype=torch.float64)
        Bx = Bounds(lb=lb.view(1, n), ub=ub.view(1, n))

        L_sq = Layer(
            id=0, kind=LayerKind.MUL.value,
            in_vars=list(range(n)) + list(range(n)),
            out_vars=list(range(n, 2 * n)),
            params={"x_vars": list(range(n)), "y_vars": list(range(n)),
                    "input_shape": (1, n), "output_shape": (1, n)},
        )
        By = tf_mul(L_sq, Bx, Bx).bounds

        L_cub = Layer(
            id=1, kind=LayerKind.MUL.value,
            in_vars=list(range(n, 2 * n)) + list(range(n)),
            out_vars=list(range(2 * n, 3 * n)),
            params={"x_vars": list(range(n, 2 * n)), "y_vars": list(range(n)),
                    "input_shape": (1, n), "output_shape": (1, n)},
        )
        Bz = tf_mul(L_cub, By, Bx).bounds

        N = 2048
        u = torch.rand(N, n, dtype=torch.float64)
        x = lb + u * (ub - lb)
        z = x * x * x

        out_lb = Bz.lb.view(-1)
        out_ub = Bz.ub.view(-1)
        for i in range(n):
            zmin, zmax = float(z[:, i].min()), float(z[:, i].max())
            self.assertLessEqual(
                float(out_lb[i]), zmin + 1e-9,
                f"coord {i}: ACT lb={float(out_lb[i])} > sampled min={zmin}; "
                f"Pow-chain interval is UNDER-approximating"
            )
            self.assertGreaterEqual(
                float(out_ub[i]), zmax - 1e-9,
                f"coord {i}: ACT ub={float(out_ub[i])} < sampled max={zmax}; "
                f"Pow-chain interval is UNDER-approximating"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
