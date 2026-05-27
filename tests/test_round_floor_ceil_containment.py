"""Sound-interval containment for the new FLOOR / CEIL / ROUND transfer
functions.

ml4acopf_2024 surfaces ONNX Floor/Ceil/Round via onnx2torch's single
``OnnxRound`` class (which dispatches by the stored ``round_function``).
We added ACT LayerKinds FLOOR / CEIL / ROUND plus interval transfers in
``tf_mlp.py``. This test file pins the soundness contract:

    forall x in [lb, ub]: op(x) in [out.lb, out.ub]

for op in {floor, ceil, round}. We sample uniformly from the input
interval AND explicitly include both ends plus half-integer boundary
points for round. A single uncontained sample is a soundness bug.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from act.back_end.core import Layer, Bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_mlp import tf_floor, tf_ceil, tf_round


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


def _make_layer(kind: str, n: int) -> Layer:
    return Layer(
        id=0, kind=kind,
        in_vars=list(range(n)),
        out_vars=list(range(n, 2 * n)),
        params={"input_shape": (1, n), "output_shape": (1, n)},
    )


def _assert_containment(tc: unittest.TestCase, *, op, tf, lb, ub, n_random=512):
    """Run tf on the given (lb, ub) bounds, then assert every sampled
    forward value is contained in the returned [out.lb, out.ub]."""
    n = lb.numel()
    L = _make_layer(kind={torch.floor: LayerKind.FLOOR.value,
                          torch.ceil: LayerKind.CEIL.value,
                          torch.round: LayerKind.ROUND.value}[op], n=n)
    Bin = Bounds(lb=lb.view(1, n), ub=ub.view(1, n))
    fact = tf(L, Bin)
    out_lb = fact.bounds.lb.view(-1)
    out_ub = fact.bounds.ub.view(-1)

    # Always include corners, the midpoints of each coord, and a wide
    # band of random samples.
    samples = [lb.clone(), ub.clone(), 0.5 * (lb + ub)]
    rng = torch.Generator()
    rng.manual_seed(0)
    for _ in range(n_random):
        u = torch.rand(n, generator=rng, dtype=lb.dtype)
        samples.append(lb + u * (ub - lb))
    # Specific half-integer probes to stress round's discontinuity.
    halfs = torch.arange(-3, 4, dtype=lb.dtype) + 0.5
    for h in halfs:
        x = lb.clone()
        # Replace coord 0 with h (if in box) to make it active.
        if (lb[0] <= h <= ub[0]):
            x[0] = h
            samples.append(x.clone())

    for x in samples:
        y = op(x)
        for i in range(n):
            tc.assertGreaterEqual(
                float(y[i]), float(out_lb[i]) - 1e-12,
                f"{op.__name__}({float(x[i])}) = {float(y[i])} < out_lb {float(out_lb[i])}"
            )
            tc.assertLessEqual(
                float(y[i]), float(out_ub[i]) + 1e-12,
                f"{op.__name__}({float(x[i])}) = {float(y[i])} > out_ub {float(out_ub[i])}"
            )


class TestFloorCeilRoundContainment(_DeviceIsolated):
    def test_floor_contains_forward(self):
        lb = torch.tensor([-2.3, -0.5, 0.0, 1.4, 3.7], dtype=torch.float64)
        ub = torch.tensor([-1.0, 0.5, 1.5, 2.9, 4.2], dtype=torch.float64)
        _assert_containment(self, op=torch.floor, tf=tf_floor, lb=lb, ub=ub)

    def test_ceil_contains_forward(self):
        lb = torch.tensor([-2.3, -0.5, 0.0, 1.4, 3.7], dtype=torch.float64)
        ub = torch.tensor([-1.0, 0.5, 1.5, 2.9, 4.2], dtype=torch.float64)
        _assert_containment(self, op=torch.ceil, tf=tf_ceil, lb=lb, ub=ub)

    def test_round_contains_forward_across_half_integers(self):
        # Half-integer points (-0.5, 0.5, 1.5, ...) are discontinuities
        # of banker's rounding, but the function is still monotone.
        lb = torch.tensor([-2.5, -1.5, -0.5, 0.5, 1.5], dtype=torch.float64)
        ub = torch.tensor([-0.5, 0.5, 1.5, 2.5, 3.5], dtype=torch.float64)
        _assert_containment(self, op=torch.round, tf=tf_round, lb=lb, ub=ub)

    def test_zero_width_interval(self):
        """Degenerate case: lb == ub (a single point). Output must be
        the singleton {op(x)}."""
        for op, tf in [(torch.floor, tf_floor), (torch.ceil, tf_ceil), (torch.round, tf_round)]:
            x = torch.tensor([0.0, 1.5, -2.3, 3.0], dtype=torch.float64)
            n = x.numel()
            L = _make_layer(kind={torch.floor: LayerKind.FLOOR.value,
                                  torch.ceil: LayerKind.CEIL.value,
                                  torch.round: LayerKind.ROUND.value}[op], n=n)
            Bin = Bounds(lb=x.view(1, n), ub=x.view(1, n))
            fact = tf(L, Bin)
            y = op(x)
            self.assertTrue(
                torch.all(fact.bounds.lb.view(-1) <= y + 1e-12) and
                torch.all(fact.bounds.ub.view(-1) >= y - 1e-12),
                f"{op.__name__}: zero-width interval result not containment-clean"
            )


class TestRoundTightAtBankerTie(_DeviceIsolated):
    """Banker's ties do not break monotonicity; singleton intervals are
    represented exactly rather than widened around the tie."""

    def test_singleton_half_integer_is_exact(self):
        lb = torch.tensor([0.5], dtype=torch.float64)
        ub = torch.tensor([0.5], dtype=torch.float64)
        L = _make_layer(LayerKind.ROUND.value, n=1)
        Bin = Bounds(lb=lb.view(1, 1), ub=ub.view(1, 1))
        fact = tf_round(L, Bin)
        # torch.round(0.5) is 0 under round-to-even.
        self.assertAlmostEqual(float(fact.bounds.lb), 0.0)
        self.assertAlmostEqual(float(fact.bounds.ub), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
