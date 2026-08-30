#!/usr/bin/env python3
# ===- test_motif_local_sharpness.py - local sharpness toy gate ------====#
# ACT: Abstract Constraint Transformer
# Copyright (C) 2025– ACT Team
#
# Licensed under the GNU Affero General Public License v3.0 or later.
# Distributed without any warranty; see <http://www.gnu.org/licenses/>.
# ===-------------------------------------------------------------------====#
"""Deterministic tests for the complementary-ReLU motif cut.

Run from the repository root:

    python -m act.back_end.hybridz_tf.test_motif_local_sharpness
"""

from __future__ import annotations

from fractions import Fraction
import unittest

from act.back_end.hybridz_tf.motif_local_sharpness import (
    MotifLocalSharpnessError,
    audit_complementary_relu_repeat_motif,
    complementary_relu_repeat_cut,
    motif_local_sharpness_self_test,
)


class MotifLocalSharpnessTest(unittest.TestCase):
    def test_minimal_cut_shape(self) -> None:
        equality = complementary_relu_repeat_cut()
        self.assertEqual(equality.relation, "==")
        self.assertEqual(equality.coefficients, (0, 1, 0, -1))
        self.assertEqual(equality.rhs, 0)
        self.assertEqual(equality.nnz, 2)

        upper = complementary_relu_repeat_cut(one_sided=True)
        self.assertEqual(upper.relation, "<=")
        self.assertEqual(upper.coefficients, equality.coefficients)
        self.assertEqual(upper.nnz, 2)

    def test_fraction_and_phase_soundness(self) -> None:
        receipt = audit_complementary_relu_repeat_motif(max_denominator=64)
        self.assertTrue(receipt.passed)
        self.assertEqual(receipt.phase_patterns_total, 8)
        # Boundary x=0 makes all eight closed phase assignments feasible.
        self.assertEqual(receipt.phase_patterns_feasible, 8)
        self.assertEqual(receipt.phase_patterns_nondegenerate, 2)
        self.assertGreaterEqual(receipt.fraction_samples, 2000)

    def test_exact_lp_tightness_and_stop_loss(self) -> None:
        receipt = audit_complementary_relu_repeat_motif()
        self.assertEqual(
            (receipt.baseline_lower, receipt.baseline_upper),
            (Fraction(-1, 2), Fraction(1, 2)),
        )
        # The obvious complementary identity is valid, but alone has no gain.
        self.assertEqual(
            (receipt.complement_only_lower, receipt.complement_only_upper),
            (Fraction(-1, 2), Fraction(1, 2)),
        )
        self.assertEqual(
            (receipt.upper_cut_lower, receipt.upper_cut_upper),
            (Fraction(-1, 2), Fraction(0)),
        )
        self.assertEqual(
            (receipt.repeat_equality_lower, receipt.repeat_equality_upper),
            (Fraction(0), Fraction(0)),
        )
        self.assertEqual(receipt.rejected_reason, "zero_lp_bound_improvement")

    def test_cost_gate(self) -> None:
        receipt = audit_complementary_relu_repeat_motif(
            max_equality_rows=1,
            max_equality_nnz=2,
        )
        self.assertEqual(receipt.accepted_rows, 1)
        self.assertEqual(receipt.accepted_nnz, 2)
        self.assertEqual(receipt.accepted_new_continuous, 0)
        self.assertEqual(receipt.accepted_new_binary, 0)
        self.assertEqual(receipt.inequality_only_rows, 2)
        self.assertEqual(receipt.inequality_only_nnz, 4)

        with self.assertRaises(MotifLocalSharpnessError):
            audit_complementary_relu_repeat_motif(max_equality_rows=0)
        with self.assertRaises(MotifLocalSharpnessError):
            audit_complementary_relu_repeat_motif(max_equality_nnz=1)

    def test_json_friendly_receipt_is_deterministic(self) -> None:
        first = motif_local_sharpness_self_test()
        second = motif_local_sharpness_self_test()
        self.assertEqual(first, second)
        self.assertEqual(first["schema"], "hybridz_motif_local_sharpness_toy_v1")
        self.assertEqual(first["baseline_lower"], "-1/2")
        self.assertEqual(first["baseline_upper"], "1/2")
        self.assertEqual(first["repeat_equality_lower"], "0")
        self.assertEqual(first["repeat_equality_upper"], "0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
