"""Unittests for the isolated Dense support-compressed scalar guard."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import json
import math
import time
import unittest
from unittest import mock

import numpy as np

from act.back_end.hybridz_tf import query_dual_scalar_guard as scalar_module
from act.back_end.hybridz_tf.query_dual_replay import (
    _matrix_product_with_error,
)
from act.back_end.hybridz_tf.query_dual_scalar_guard import (
    HashableDiagnostics,
    QueryDualScalarGuardError,
    dense_support_compressed_guard,
    outward_roundoff_parameters,
    prepare_dense_support,
)


_F64 = np.float64
_ETA = float(np.nextafter(_F64(0.0), _F64(math.inf)))


def _fraction_matrix_product(
    coefficients: np.ndarray,
    weight: np.ndarray,
):
    return tuple(
        tuple(
            sum(
                (
                    Fraction.from_float(float(coefficients[q, i]))
                    * Fraction.from_float(float(weight[i, j]))
                    for i in range(coefficients.shape[1])
                ),
                Fraction(0),
            )
            for j in range(weight.shape[1])
        )
        for q in range(coefficients.shape[0])
    )


def _weighted_actual_error(
    nominal: np.ndarray,
    exact,
    max_abs: np.ndarray,
):
    masses = tuple(Fraction.from_float(float(value)) for value in max_abs)
    return tuple(
        sum(
            (
                abs(
                    Fraction.from_float(float(nominal[q, j]))
                    - exact[q][j]
                )
                * masses[j]
                for j in range(nominal.shape[1])
            ),
            Fraction(0),
        )
        for q in range(nominal.shape[0])
    )


def _symmetric_box_lower(coefficients, max_abs: np.ndarray):
    masses = tuple(Fraction.from_float(float(value)) for value in max_abs)
    return tuple(
        -sum(
            (abs(row[j]) * masses[j] for j in range(len(masses))),
            Fraction(0),
        )
        for row in coefficients
    )


def _stored_nominal_rows(nominal: np.ndarray):
    return tuple(
        tuple(Fraction.from_float(float(value)) for value in row)
        for row in nominal
    )


def _componentwise_penalty(
    radius: np.ndarray,
    max_abs: np.ndarray,
):
    masses = tuple(Fraction.from_float(float(value)) for value in max_abs)
    return tuple(
        sum(
            (
                Fraction.from_float(float(radius[q, j])) * masses[j]
                for j in range(radius.shape[1])
            ),
            Fraction(0),
        )
        for q in range(radius.shape[0])
    )


class DenseScalarGuardTests(unittest.TestCase):
    def _audit_against_fraction_and_v3(
        self,
        coefficients: np.ndarray,
        weight: np.ndarray,
        max_abs: np.ndarray,
    ):
        support = prepare_dense_support(
            weight,
            max_abs,
            binding={
                "box_semantics": "predecessor-output",
                "frame_sha256": "f" * 64,
                "layer_id": "2",
            },
        )
        result = dense_support_compressed_guard(
            coefficients, weight, support
        )
        max_abs_fraction = tuple(
            Fraction.from_float(float(value)) for value in max_abs
        )
        exact_support = tuple(
            sum(
                (
                    abs(Fraction.from_float(float(weight[i, j])))
                    * max_abs_fraction[j]
                    for j in range(weight.shape[1])
                ),
                Fraction(0),
            )
            for i in range(weight.shape[0])
        )
        for stored, exact_value in zip(
            support.support_upper, exact_support
        ):
            self.assertGreaterEqual(
                Fraction.from_float(float(stored)), exact_value
            )
        self.assertGreaterEqual(
            Fraction.from_float(support.box_mass_upper),
            sum(max_abs_fraction, Fraction(0)),
        )
        for q, stored in enumerate(result.support_mass_upper):
            exact_mass = sum(
                (
                    abs(Fraction.from_float(float(coefficients[q, i])))
                    * exact_support[i]
                    for i in range(coefficients.shape[1])
                ),
                Fraction(0),
            )
            self.assertGreaterEqual(
                Fraction.from_float(float(stored)), exact_mass
            )
        v3_nominal, v3_radius = _matrix_product_with_error(
            coefficients, weight
        )
        np.testing.assert_array_equal(
            result.nominal.view(np.uint64),
            v3_nominal.view(np.uint64),
        )

        exact = _fraction_matrix_product(coefficients, weight)
        actual_error = _weighted_actual_error(
            result.nominal, exact, max_abs
        )
        v3_penalty = _componentwise_penalty(v3_radius, max_abs)
        exact_lower = _symmetric_box_lower(exact, max_abs)
        nominal_lower = _symmetric_box_lower(
            _stored_nominal_rows(result.nominal), max_abs
        )
        v5_lower = tuple(
            nominal_lower[q]
            - Fraction.from_float(float(result.scalar_guard[q]))
            for q in range(coefficients.shape[0])
        )
        v3_lower = tuple(
            nominal_lower[q] - v3_penalty[q]
            for q in range(coefficients.shape[0])
        )

        for q in range(coefficients.shape[0]):
            self.assertLessEqual(
                actual_error[q],
                Fraction.from_float(float(result.scalar_guard[q])),
            )
            self.assertLessEqual(actual_error[q], v3_penalty[q])
            self.assertLessEqual(v5_lower[q], exact_lower[q])
            self.assertLessEqual(v3_lower[q], exact_lower[q])
        return {
            "support": support,
            "result": result,
            "actual_error": actual_error,
            "v3_penalty": v3_penalty,
            "v5_lower": v5_lower,
            "v3_lower": v3_lower,
            "exact_lower": exact_lower,
        }

    def test_tau_and_gamma_are_outward_fraction_enclosures(self):
        operations = 8
        parameters = outward_roundoff_parameters(operations)
        u = Fraction(1, 2**53)
        eta = Fraction(1, 2**1074)
        denominator = Fraction(1) - operations * u
        exact_gamma = operations * u / denominator
        exact_tau = operations * eta / denominator
        self.assertFalse(parameters.proof_authority)
        self.assertGreaterEqual(
            Fraction.from_float(parameters.gamma_upper), exact_gamma
        )
        self.assertGreaterEqual(
            Fraction.from_float(parameters.tau_upper), exact_tau
        )

    def test_large_cancellation_is_sound_and_nominal_matches_v3_bits(self):
        coefficients = np.asarray(
            [[1.0e16, 1.0, -1.0e16]], dtype=_F64
        )
        weight = np.ones((3, 1), dtype=_F64)
        max_abs = np.ones(1, dtype=_F64)
        audit = self._audit_against_fraction_and_v3(
            coefficients, weight, max_abs
        )
        exact = _fraction_matrix_product(coefficients, weight)
        self.assertEqual(exact[0][0], Fraction(1))
        self.assertEqual(audit["result"].nominal[0, 0], 0.0)
        self.assertGreaterEqual(
            Fraction.from_float(float(audit["result"].scalar_guard[0])),
            Fraction(1),
        )
        # The extra outward support-precompute stage can cost a few ulps versus
        # V3.  Compare tightness with an explicit controlled-toy budget while
        # Fraction—not V3—remains the soundness oracle.
        self.assertLessEqual(
            Fraction.from_float(float(audit["result"].scalar_guard[0])),
            2 * audit["v3_penalty"][0],
        )

    def test_subnormal_nonzero_product_is_not_an_exact_zero_shortcut(self):
        coefficients = np.asarray([[_ETA]], dtype=_F64)
        weight = np.asarray([[0.5]], dtype=_F64)
        max_abs = np.ones(1, dtype=_F64)
        audit = self._audit_against_fraction_and_v3(
            coefficients, weight, max_abs
        )
        exact = _fraction_matrix_product(coefficients, weight)
        self.assertEqual(
            exact[0][0], Fraction.from_float(_ETA) / Fraction(2)
        )
        self.assertEqual(audit["result"].nominal[0, 0], 0.0)
        self.assertGreater(audit["result"].support_mass_upper[0], 0.0)
        self.assertGreater(audit["result"].scalar_guard[0], 0.0)

    def test_degenerate_exact_zero_rows_and_boxes_have_zero_guard(self):
        weight = np.asarray(
            [[2.0, -3.0], [4.0, 5.0]], dtype=_F64
        )
        coefficients = np.asarray(
            [[0.0, -0.0], [1.0, -2.0]], dtype=_F64
        )

        zero_box = np.zeros(2, dtype=_F64)
        zero_box_audit = self._audit_against_fraction_and_v3(
            coefficients, weight, zero_box
        )
        np.testing.assert_array_equal(
            zero_box_audit["result"].scalar_guard,
            np.zeros(2, dtype=_F64),
        )

        max_abs = np.asarray([1.0, 2.0], dtype=_F64)
        zero_query_audit = self._audit_against_fraction_and_v3(
            coefficients[:1], weight, max_abs
        )
        self.assertEqual(zero_query_audit["result"].scalar_guard[0], 0.0)

        zero_weight = np.zeros_like(weight)
        zero_weight_audit = self._audit_against_fraction_and_v3(
            coefficients[1:], zero_weight, max_abs
        )
        self.assertEqual(zero_weight_audit["result"].scalar_guard[0], 0.0)

    def test_signed_zero_is_an_exact_zero_operand_not_nominal_zero_guess(self):
        coefficients = np.asarray(
            [[0.0, -0.0], [-0.0, 0.0]], dtype=_F64
        )
        weight = np.asarray(
            [[-3.0, 5.0], [7.0, -11.0]], dtype=_F64
        )
        max_abs = np.asarray([-0.0, 2.0], dtype=_F64)
        audit = self._audit_against_fraction_and_v3(
            coefficients, weight, max_abs
        )
        np.testing.assert_array_equal(
            audit["result"].scalar_guard,
            np.zeros(2, dtype=_F64),
        )
        self.assertEqual(audit["actual_error"], (Fraction(0), Fraction(0)))

    def test_random_fraction_soundness_and_v3_tightness_comparison(self):
        rng = np.random.default_rng(20260728)
        comparisons = []
        for _ in range(60):
            query_count = int(rng.integers(1, 5))
            output_width = int(rng.integers(1, 8))
            input_width = int(rng.integers(1, 9))
            coefficients = np.ascontiguousarray(
                rng.normal(size=(query_count, output_width)), dtype=_F64
            )
            weight = np.ascontiguousarray(
                rng.normal(size=(output_width, input_width)), dtype=_F64
            )
            max_abs = np.ascontiguousarray(
                rng.uniform(0.0, 3.0, size=input_width), dtype=_F64
            )
            audit = self._audit_against_fraction_and_v3(
                coefficients, weight, max_abs
            )
            for q in range(query_count):
                v5_gap = (
                    audit["exact_lower"][q] - audit["v5_lower"][q]
                )
                v3_gap = (
                    audit["exact_lower"][q] - audit["v3_lower"][q]
                )
                self.assertGreaterEqual(v5_gap, 0)
                self.assertGreaterEqual(v3_gap, 0)
                comparisons.append((v5_gap, v3_gap))

        # Tightness is compared, not assumed as a proof obligation.  On this
        # bounded normal-valued corpus, the extra outward stage must stay
        # within a 2x gap budget relative to V3.
        for v5_gap, v3_gap in comparisons:
            self.assertGreater(v3_gap, 0)
            self.assertLessEqual(v5_gap, 2 * v3_gap)

    def test_hashable_diagnostics_and_immutable_support_binding(self):
        weight = np.asarray(
            [[1.0, -2.0], [3.0, 4.0]], dtype=_F64
        )
        max_abs = np.asarray([2.0, 5.0], dtype=_F64)
        coefficients = np.asarray([[0.5, -0.25]], dtype=_F64)
        support = prepare_dense_support(
            weight,
            max_abs,
            binding={"frame_sha256": "a" * 64, "layer_id": "7"},
        )
        result = dense_support_compressed_guard(
            coefficients, weight, support
        )
        self.assertFalse(support.proof_authority)
        self.assertFalse(result.proof_authority)
        self.assertIsInstance(hash(support.diagnostics), int)
        self.assertIsInstance(hash(result.diagnostics), int)
        self.assertEqual(
            result.diagnostics.as_dict()["integration_gate"],
            "not-authoritative",
        )
        with self.assertRaises(ValueError):
            support.support_upper.setflags(write=True)
        with self.assertRaises(ValueError):
            HashableDiagnostics(
                items=result.diagnostics.items,
                sha256="0" * 64,
            )

        changed_weight = weight.copy()
        changed_weight[0, 0] = np.nextafter(
            changed_weight[0, 0], _F64(math.inf)
        )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard(
                coefficients, changed_weight, support
            )

        changed_values = support.support_upper.copy()
        changed_values[0] = np.nextafter(
            changed_values[0], _F64(math.inf)
        )
        immutable_changed = np.frombuffer(
            changed_values.tobytes(), dtype=_F64
        )
        changed_support = replace(
            support, support_upper=immutable_changed
        )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard(
                coefficients, weight, changed_support
            )

        forged_mass = replace(
            support,
            box_mass_upper=float(
                np.nextafter(
                    _F64(support.box_mass_upper), _F64(math.inf)
                )
            ),
        )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard(
                coefficients, weight, forged_mass
            )

        forged_max_abs_hash = replace(
            support, max_abs_sha256="0" * 64
        )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard(
                coefficients, weight, forged_max_abs_hash
            )

    def test_deadline_is_fail_closed_before_and_during_work(self):
        weight = np.eye(2, dtype=_F64)
        max_abs = np.ones(2, dtype=_F64)
        coefficients = np.ones((1, 2), dtype=_F64)
        expired = time.monotonic() - 1.0
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "DEADLINE_EXPIRED"
        ):
            prepare_dense_support(
                weight, max_abs, deadline=expired
            )
        support = prepare_dense_support(weight, max_abs)
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "DEADLINE_EXPIRED"
        ):
            dense_support_compressed_guard(
                coefficients, weight, support, deadline=expired
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "INVALID_DEADLINE"
        ):
            prepare_dense_support(
                weight, max_abs, deadline=math.inf
            )

        # First check admits the work; the second check denies publication.
        with mock.patch.object(
            scalar_module.time,
            "monotonic",
            side_effect=[0.0, 2.0],
        ):
            with self.assertRaisesRegex(
                QueryDualScalarGuardError, "DEADLINE_EXPIRED"
            ):
                prepare_dense_support(
                    weight, max_abs, deadline=1.0
                )

    def test_controlled_production_shape_timing_and_artifact_sha(self):
        # Synthetic only: a classifier-sized Dense shape, no model/data load.
        rng = np.random.default_rng(73491)
        query_count, output_width, input_width = 64, 512, 1000
        coefficients = np.ascontiguousarray(
            rng.normal(size=(query_count, output_width)), dtype=_F64
        )
        weight = np.ascontiguousarray(
            rng.normal(size=(output_width, input_width)), dtype=_F64
        )
        max_abs = np.ascontiguousarray(
            rng.uniform(0.0, 2.0, size=input_width), dtype=_F64
        )
        deadline = time.monotonic() + 30.0

        started = time.perf_counter()
        support = prepare_dense_support(
            weight,
            max_abs,
            binding={
                "artifact": "synthetic-64x512x1000",
                "box_semantics": "predecessor-output",
            },
            deadline=deadline,
        )
        after_precompute = time.perf_counter()
        result = dense_support_compressed_guard(
            coefficients,
            weight,
            support,
            deadline=deadline,
        )
        after_runtime = time.perf_counter()
        v3_started = time.perf_counter()
        v3_nominal, _ = _matrix_product_with_error(
            coefficients, weight
        )
        v3_finished = time.perf_counter()

        np.testing.assert_array_equal(
            result.nominal.view(np.uint64),
            v3_nominal.view(np.uint64),
        )
        self.assertTrue(np.all(np.isfinite(result.scalar_guard)))
        precompute_seconds = after_precompute - started
        runtime_seconds = after_runtime - after_precompute
        total_seconds = after_runtime - started
        v3_seconds = v3_finished - v3_started
        for value in (
            precompute_seconds,
            runtime_seconds,
            total_seconds,
            v3_seconds,
        ):
            self.assertGreater(value, 0.0)
            self.assertTrue(math.isfinite(value))
        self.assertLess(total_seconds, 30.0)

        artifact = {
            "artifact_sha256": result.diagnostics.sha256,
            "precompute_diagnostics_sha256": support.diagnostics.sha256,
            "shape": [query_count, output_width, input_width],
            "precompute_seconds": precompute_seconds,
            "runtime_seconds": runtime_seconds,
            "total_with_precompute_seconds": total_seconds,
            "v3_componentwise_seconds": v3_seconds,
            "observed_v3_over_v5_runtime": (
                v3_seconds / runtime_seconds
            ),
            "observed_v3_over_v5_total": v3_seconds / total_seconds,
        }
        print(
            "CONTROLLED_DENSE_SCALAR_GUARD "
            + json.dumps(artifact, sort_keys=True)
        )

    def test_fail_closed_inputs_and_platform(self):
        weight = np.eye(2, dtype=_F64)
        max_abs = np.ones(2, dtype=_F64)
        coefficients = np.ones((1, 2), dtype=_F64)

        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "NEGATIVE_MASS"
        ):
            prepare_dense_support(
                weight, np.asarray([1.0, -1.0], dtype=_F64)
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "NONFINITE"
        ):
            prepare_dense_support(
                np.asarray([[1.0, math.inf], [0.0, 1.0]], dtype=_F64),
                max_abs,
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "INVALID_DTYPE"
        ):
            prepare_dense_support(weight.astype(np.float32), max_abs)
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "INVALID_LAYOUT"
        ):
            prepare_dense_support(weight[:, ::-1], max_abs)
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "SHAPE_MISMATCH"
        ):
            prepare_dense_support(weight, np.ones(3, dtype=_F64))

        support = prepare_dense_support(weight, max_abs)
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "NONFINITE"
        ):
            dense_support_compressed_guard(
                np.asarray([[math.nan, 1.0]], dtype=_F64),
                weight,
                support,
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardError, "SHAPE_MISMATCH"
        ):
            dense_support_compressed_guard(
                np.ones((1, 3), dtype=_F64), weight, support
            )
        with mock.patch.object(
            scalar_module, "_has_wide_longdouble", return_value=False
        ):
            with self.assertRaisesRegex(
                QueryDualScalarGuardError, "NUMERIC_PLATFORM"
            ):
                prepare_dense_support(weight, max_abs)


if __name__ == "__main__":
    unittest.main()
