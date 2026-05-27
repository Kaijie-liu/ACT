"""Containment tests for sound interval SIN/COS transfers."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from act.back_end.core import Bounds, Layer
from act.back_end.interval_tf.tf_mlp import tf_cos, tf_sin
from act.back_end.layer_schema import LayerKind


def _layer(kind: str, n: int) -> Layer:
    return Layer(
        id=1, kind=kind, in_vars=list(range(n)),
        out_vars=list(range(n, 2 * n)),
        params={"input_shape": (1, n), "output_shape": (1, n)},
    )


class TestTrigIntervalContainment(unittest.TestCase):
    def _assert_samples_contained(self, fn, tf, kind, lb, ub):
        bounds = tf(_layer(kind, lb.numel()), Bounds(lb.view(1, -1), ub.view(1, -1))).bounds
        samples = [lb, ub, (lb + ub) / 2]
        gen = torch.Generator().manual_seed(0)
        for _ in range(2048):
            samples.append(lb + torch.rand(lb.shape, generator=gen, dtype=lb.dtype) * (ub - lb))
        for x in samples:
            y = fn(x)
            self.assertTrue(torch.all(y >= bounds.lb.flatten() - 1e-12))
            self.assertTrue(torch.all(y <= bounds.ub.flatten() + 1e-12))

    def test_sin_contains_endpoints_extrema_and_full_cycles(self):
        lb = torch.tensor([-0.1, -torch.pi, 0.0, -8.0], dtype=torch.float64)
        ub = torch.tensor([0.2, torch.pi, torch.pi / 2, 8.0], dtype=torch.float64)
        self._assert_samples_contained(torch.sin, tf_sin, LayerKind.SIN.value, lb, ub)

    def test_cos_contains_endpoints_extrema_and_full_cycles(self):
        lb = torch.tensor([-0.1, 0.1, -torch.pi, -8.0], dtype=torch.float64)
        ub = torch.tensor([0.2, 2 * torch.pi, torch.pi, 8.0], dtype=torch.float64)
        self._assert_samples_contained(torch.cos, tf_cos, LayerKind.COS.value, lb, ub)

    def test_extrema_are_included_exactly(self):
        sin_b = tf_sin(
            _layer(LayerKind.SIN.value, 1),
            Bounds(torch.tensor([[0.0]]), torch.tensor([[torch.pi]])),
        ).bounds
        cos_b = tf_cos(
            _layer(LayerKind.COS.value, 1),
            Bounds(torch.tensor([[0.0]]), torch.tensor([[torch.pi]])),
        ).bounds
        self.assertEqual(float(sin_b.ub.item()), 1.0)
        self.assertEqual(float(cos_b.lb.item()), -1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
