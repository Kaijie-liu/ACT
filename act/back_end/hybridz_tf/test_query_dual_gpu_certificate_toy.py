"""Exact toys for the rejected GPU-authority shortcut and CPU alternative."""

from __future__ import annotations

from fractions import Fraction
import math
import unittest

import numpy as np

from act.back_end.hybridz_tf.query_dual_gpu_certificate_toy import (
    audit_dense_compressed_roundoff_guard,
    audit_untrusted_dense_claim,
    fraction_relu_upper_line,
    validate_fraction_relu_upper_line,
)


class QueryDualGpuCertificateToyTests(unittest.TestCase):
    def test_compressed_mass_is_exact_rearrangement(self):
        rng = np.random.default_rng(20260728)
        coefficients = rng.normal(size=(7, 5)).astype(np.float64)
        weight = rng.normal(size=(5, 9)).astype(np.float64)
        max_abs = rng.uniform(0.0, 3.0, size=9).astype(np.float64)
        audit = audit_dense_compressed_roundoff_guard(
            coefficients, weight, max_abs
        )
        self.assertFalse(audit.proof_authority)
        self.assertEqual(
            audit.full_weighted_mass,
            audit.compressed_weighted_mass,
        )
        for actual, guard in zip(
            audit.weighted_actual_error, audit.weighted_guard
        ):
            self.assertLessEqual(actual, guard)

    def test_compressed_guard_covers_1e16_cancellation(self):
        audit = audit_dense_compressed_roundoff_guard(
            np.asarray([[1.0e16, 1.0, -1.0e16]], dtype=np.float64),
            np.ones((3, 1), dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
        self.assertEqual(audit.exact_product[0][0], Fraction(1))
        self.assertLessEqual(
            audit.weighted_actual_error[0],
            audit.weighted_guard[0],
        )

    def test_compressed_guard_covers_min_subnormal_underflow(self):
        eta = float(np.nextafter(np.float64(0.0), np.float64(math.inf)))
        audit = audit_dense_compressed_roundoff_guard(
            np.asarray([[eta]], dtype=np.float64),
            np.asarray([[0.5]], dtype=np.float64),
            np.ones(1, dtype=np.float64),
        )
        self.assertEqual(audit.nominal[0, 0], 0.0)
        self.assertEqual(
            audit.exact_product[0][0],
            Fraction.from_float(eta) / 2,
        )
        self.assertLessEqual(
            audit.weighted_actual_error[0],
            audit.weighted_guard[0],
        )

    def test_untrusted_claim_is_sound_but_never_beats_zero_claim(self):
        rng = np.random.default_rng(8841)
        for _ in range(50):
            output_width, input_width = 4, 6
            coefficients = rng.normal(size=output_width)
            weight = rng.normal(size=(output_width, input_width))
            bias = rng.normal(size=output_width)
            lower = rng.normal(size=input_width)
            upper = lower + rng.uniform(0.0, 2.0, size=input_width)
            claim = rng.normal(size=input_width)
            audit = audit_untrusted_dense_claim(
                coefficients, weight, bias, lower, upper, claim
            )
            self.assertFalse(audit.proof_authority)
            self.assertLessEqual(
                audit.claimed_lower, audit.zero_claim_lower
            )
            self.assertLessEqual(
                audit.claimed_lower, audit.exact_box_minimum
            )

    def test_zero_claim_is_optimal_in_triangle_certificate_family(self):
        common = {
            "coefficients": np.asarray([1.0, -0.5], dtype=np.float64),
            "weight": np.asarray(
                [[2.0, -1.0], [0.25, 3.0]], dtype=np.float64
            ),
            "bias": np.asarray([0.1, -0.2], dtype=np.float64),
            "predecessor_lower": np.asarray([-1.0, -2.0], dtype=np.float64),
            "predecessor_upper": np.asarray([2.0, 3.0], dtype=np.float64),
        }
        zero = audit_untrusted_dense_claim(
            **common,
            claimed_predecessor=np.zeros(2, dtype=np.float64),
        )
        nonzero = audit_untrusted_dense_claim(
            **common,
            claimed_predecessor=np.asarray([4.0, -7.0], dtype=np.float64),
        )
        self.assertEqual(zero.claimed_lower, zero.zero_claim_lower)
        self.assertLess(nonzero.claimed_lower, zero.claimed_lower)

    def test_two_equal_wrong_implementations_do_not_prove_a_bound(self):
        # Both hypothetical GPU implementations report coefficient zero and
        # raw lower bound zero for y=x over [-1,1].  Agreement is exact, while
        # the true lower bound is -1.
        gpu_a_raw_lower = Fraction(0)
        gpu_b_raw_lower = Fraction(0)
        self.assertEqual(gpu_a_raw_lower, gpu_b_raw_lower)
        audit = audit_untrusted_dense_claim(
            np.asarray([1.0], dtype=np.float64),
            np.asarray([[1.0]], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
            np.asarray([-1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray([0.0], dtype=np.float64),
        )
        self.assertGreater(gpu_a_raw_lower, audit.exact_box_minimum)
        self.assertLessEqual(audit.claimed_lower, audit.exact_box_minimum)

    def test_fraction_relu_line_handles_1e16_and_subnormal(self):
        for lower, upper in (
            (-1.0e16, 3.0),
            (
                -float(
                    np.nextafter(np.float64(0.0), np.float64(math.inf))
                ),
                float(np.nextafter(np.float64(0.0), np.float64(math.inf))),
            ),
        ):
            line = fraction_relu_upper_line(lower, upper)
            self.assertFalse(line.proof_authority)
            self.assertTrue(
                validate_fraction_relu_upper_line(
                    lower, upper, line.slope, line.intercept
                )
            )
            bad_intercept = float(
                np.nextafter(
                    np.float64(line.intercept), np.float64(-math.inf)
                )
            )
            self.assertFalse(
                validate_fraction_relu_upper_line(
                    lower, upper, line.slope, bad_intercept
                )
            )


if __name__ == "__main__":
    unittest.main()
