#!/usr/bin/env python3
"""Controlled audits for heuristic property-tail alpha candidates."""

from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.hybridz_tf.property_tail_candidates import (
    optimize_property_tail_negative_alpha,
)


class PropertyTailCandidateTests(unittest.TestCase):
    @staticmethod
    def _cancellation_inputs() -> dict[str, object]:
        # Two preactivations are the same x in [-1,1].  The positive secant
        # contributes +0.5*x, while alpha=0.5 on the negative coefficient
        # cancels that generator exactly.
        return {
            "preactivation_center": np.zeros(2, dtype=np.float64),
            "preactivation_generators": sp.csr_matrix(
                [[1.0], [1.0]], dtype=np.float64
            ),
            "preactivation_error": np.zeros(2, dtype=np.float64),
            "baseline_planes": np.asarray(
                [[0.5, 0.0]], dtype=np.float64
            ),
            "baseline_intercepts": np.asarray([0.5], dtype=np.float64),
            "property_coefficients": np.asarray(
                [[1.0, -1.0]], dtype=np.float64
            ),
            "lower": np.asarray([-1.0, -1.0], dtype=np.float64),
            "upper": np.asarray([1.0, 1.0], dtype=np.float64),
        }

    def test_controlled_cancellation_improves_proxy(self) -> None:
        result = optimize_property_tail_negative_alpha(
            **self._cancellation_inputs(),
            steps=40,
            time_limit=2.0,
            learning_rate=0.08,
            max_cells=100,
            device="cpu",
        )
        self.assertIn(result.receipt["status"], {"optimized", "time_limit_partial"})
        self.assertFalse(result.receipt["proof_authority"])
        self.assertEqual(result.receipt["proxy_improved_rivals"], 1)
        self.assertEqual(result.alpha.shape, (1, 2))
        self.assertEqual(result.alpha[0, 0], 0.0)
        self.assertGreater(result.alpha[0, 1], 0.25)
        self.assertLess(result.alpha[0, 1], 0.75)
        self.assertGreater(result.receipt["proxy_total_improvement"], 0.4)

    def test_optimizer_works_inside_outer_no_grad(self) -> None:
        with torch.no_grad():
            result = optimize_property_tail_negative_alpha(
                **self._cancellation_inputs(),
                steps=12,
                time_limit=2.0,
                max_cells=100,
                device="cpu",
            )
        self.assertGreater(result.receipt["completed_steps"], 0)
        self.assertGreater(np.count_nonzero(result.alpha), 0)

    def test_projection_mask_and_hash_are_deterministic(self) -> None:
        kwargs = {
            **self._cancellation_inputs(),
            "steps": 20,
            "time_limit": 2.0,
            "learning_rate": 0.05,
            "max_cells": 100,
            "device": "cpu",
        }
        first = optimize_property_tail_negative_alpha(**kwargs)
        second = optimize_property_tail_negative_alpha(**kwargs)
        self.assertTrue(np.array_equal(first.alpha, second.alpha))
        self.assertEqual(
            first.receipt["alpha_sha256"],
            second.receipt["alpha_sha256"],
        )
        self.assertTrue(np.all(first.alpha >= 0.0))
        self.assertTrue(np.all(first.alpha <= 1.0))

    def test_boundary_optima_survive_nondifferentiable_zero(self) -> None:
        common = {
            "preactivation_generators": sp.csr_matrix(
                (1, 0), dtype=np.float64
            ),
            "baseline_planes": np.zeros((1, 1), dtype=np.float64),
            "baseline_intercepts": np.zeros(1, dtype=np.float64),
            "property_coefficients": np.asarray(
                [[-1.0]], dtype=np.float64
            ),
            "lower": np.asarray([-1.0], dtype=np.float64),
            "upper": np.asarray([1.0], dtype=np.float64),
            "steps": 12,
            "time_limit": 1.0,
            "max_cells": 100,
            "device": "cpu",
        }
        # U(alpha)=(-alpha)*.5 + alpha*1 = .5 alpha.
        at_zero = optimize_property_tail_negative_alpha(
            **common,
            preactivation_center=np.asarray([0.5], dtype=np.float64),
            preactivation_error=np.asarray([1.0], dtype=np.float64),
        )
        self.assertEqual(at_zero.alpha[0, 0], 0.0)

        # U(alpha)=(-alpha)*2, so the opposite boundary is optimal.
        at_one = optimize_property_tail_negative_alpha(
            **common,
            preactivation_center=np.asarray([2.0], dtype=np.float64),
            preactivation_error=np.asarray([0.0], dtype=np.float64),
        )
        self.assertEqual(at_one.alpha[0, 0], 1.0)

    def test_sparse_path_never_densifies_generator_matrix(self) -> None:
        with mock.patch.object(
            sp.csr_matrix,
            "toarray",
            side_effect=AssertionError("generator densification forbidden"),
        ):
            result = optimize_property_tail_negative_alpha(
                **self._cancellation_inputs(),
                steps=4,
                time_limit=1.0,
                max_cells=100,
                device="cpu",
            )
        self.assertTrue(result.receipt["sparse_spmm"])
        self.assertGreater(np.count_nonzero(result.alpha), 0)

    def test_stop_losses_return_exact_zero_baseline(self) -> None:
        disabled = optimize_property_tail_negative_alpha(
            **self._cancellation_inputs(),
            steps=0,
            time_limit=2.0,
            max_cells=100,
            device="cpu",
        )
        self.assertEqual(disabled.receipt["status"], "disabled")
        self.assertFalse(np.any(disabled.alpha))

        capped = optimize_property_tail_negative_alpha(
            **self._cancellation_inputs(),
            steps=2,
            time_limit=2.0,
            max_cells=1,
            device="cpu",
        )
        self.assertEqual(
            capped.receipt["status"], "max_cells_fallback_baseline"
        )
        self.assertFalse(np.any(capped.alpha))

        expired = optimize_property_tail_negative_alpha(
            **self._cancellation_inputs(),
            steps=2,
            time_limit=2.0,
            max_cells=100,
            deadline=time.monotonic() - 1.0,
            device="cpu",
        )
        self.assertEqual(
            expired.receipt["status"], "deadline_fallback_baseline"
        )
        self.assertFalse(np.any(expired.alpha))

    def test_malformed_inputs_fail_before_optimization(self) -> None:
        bad = self._cancellation_inputs()
        bad["preactivation_error"] = np.asarray(
            [0.0, -1.0], dtype=np.float64
        )
        with self.assertRaisesRegex(ValueError, "malformed"):
            optimize_property_tail_negative_alpha(
                **bad,
                steps=2,
                time_limit=1.0,
                max_cells=100,
                device="cpu",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
