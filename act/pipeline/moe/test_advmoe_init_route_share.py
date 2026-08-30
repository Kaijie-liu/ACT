"""Tests for the AdvMoE initialization route-share diagnostic."""

from __future__ import annotations

import unittest

import numpy as np
import torch

from act.pipeline.moe.advmoe_init_route_share import (
    _cross_route_line_bracket,
    _forward_scores,
)


class _TwoRouteToy(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scalar = value.reshape(len(value), -1).mean(dim=1)
        return torch.stack((0.5 - scalar, scalar - 0.5), dim=1)


class InitRouteShareTests(unittest.TestCase):
    def test_forward_scores_batches_without_reordering(self) -> None:
        inputs = np.array([0.0, 0.25, 0.75, 1.0], dtype=np.float32).reshape(
            4, 1, 1, 1
        )
        scores = _forward_scores(
            _TwoRouteToy(), inputs, device=torch.device("cpu"), batch_size=3
        )
        np.testing.assert_array_equal(scores.argmax(axis=1), [0, 0, 1, 1])

    def test_cross_route_line_returns_concrete_opposite_endpoint(self) -> None:
        row, witness = _cross_route_line_bracket(
            _TwoRouteToy(),
            np.array([[[0.0]]], dtype=np.float32),
            np.array([[[1.0]]], dtype=np.float32),
            0,
            device=torch.device("cpu"),
            iterations=60,
        )
        self.assertEqual(row["status"], "CONCRETE_CROSS_ROUTE_LINE_BRACKET")
        self.assertEqual(row["lower_route"], 0)
        self.assertEqual(row["upper_route"], 1)
        self.assertLessEqual(row["lower_linf"], 0.5)
        self.assertGreaterEqual(row["upper_linf"], 0.5)
        with torch.no_grad():
            route = int(
                _TwoRouteToy()(torch.from_numpy(witness[None])).argmax(dim=1).item()
            )
        self.assertEqual(route, 1)


if __name__ == "__main__":
    unittest.main()
