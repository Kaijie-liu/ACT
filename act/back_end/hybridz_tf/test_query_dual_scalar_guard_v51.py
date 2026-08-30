"""Controlled unittests for the isolated V5.1a Dense scalar guard."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import math
import time
import unittest

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as v3
from act.back_end.hybridz_tf.query_dual_scalar_guard_v51 import (
    DenseV51Support,
    QueryDualScalarGuardV51Error,
    check_v51_platform,
    dense_support_compressed_guard_v51,
    dot_up_longdouble,
    prepare_dense_support_v51,
)


_F64 = np.float64
_ETA = float(np.nextafter(_F64(0.0), _F64(math.inf)))


def _fraction(value: float) -> Fraction:
    return Fraction.from_float(float(value))


def _exact_matrix_product(
    coefficients: np.ndarray, weight: np.ndarray
):
    return tuple(
        tuple(
            sum(
                (
                    _fraction(coefficients[q, i])
                    * _fraction(weight[i, j])
                    for i in range(coefficients.shape[1])
                ),
                Fraction(0),
            )
            for j in range(weight.shape[1])
        )
        for q in range(coefficients.shape[0])
    )


def _weighted_actual_error(
    nominal: np.ndarray, exact, max_abs: np.ndarray
):
    masses = tuple(_fraction(value) for value in max_abs)
    return tuple(
        sum(
            (
                abs(_fraction(nominal[q, j]) - exact[q][j])
                * masses[j]
                for j in range(nominal.shape[1])
            ),
            Fraction(0),
        )
        for q in range(nominal.shape[0])
    )


def _v3_penalty(
    coefficients: np.ndarray, weight: np.ndarray, max_abs: np.ndarray
):
    nominal, radius = v3._matrix_product_with_error(coefficients, weight)
    if not np.any(radius):
        return nominal, np.zeros(coefficients.shape[0], dtype=np.float64)
    _, raw_error = v3._row_dots_with_error(radius, max_abs)
    absorbed = np.asarray(radius @ max_abs, dtype=np.float64)
    penalty = v3._upper_nonnegative_sum(absorbed, raw_error)
    zero_rows = ~np.any(
        (radius != 0.0) & (max_abs.reshape(1, -1) != 0.0), axis=1
    )
    penalty[zero_rows] = 0.0
    return nominal, penalty


def _audit_fraction(
    case: unittest.TestCase,
    coefficients: np.ndarray,
    weight: np.ndarray,
    max_abs: np.ndarray,
    *,
    tile_width: int = 256,
):
    support = prepare_dense_support_v51(
        weight,
        max_abs,
        binding={
            "frame_sha256": "f" * 64,
            "layer_id": "2",
            "semantics": "predecessor-output",
        },
    )
    result = dense_support_compressed_guard_v51(
        coefficients,
        weight,
        max_abs,
        support,
        tile_width=tile_width,
    )
    v3_nominal, v3_guard = _v3_penalty(
        coefficients, weight, max_abs
    )
    np.testing.assert_array_equal(
        result.nominal.view(np.uint64),
        v3_nominal.view(np.uint64),
    )

    exact = _exact_matrix_product(coefficients, weight)
    errors = _weighted_actual_error(result.nominal, exact, max_abs)
    for query_index, error in enumerate(errors):
        case.assertGreaterEqual(
            _fraction(result.final_guard[query_index]), error
        )
        case.assertLessEqual(
            result.final_guard[query_index], v3_guard[query_index]
        )
        old_lower = v3._down_add(
            np.zeros(1, dtype=np.float64),
            -v3_guard[query_index : query_index + 1],
            where="V5.1 test old lower",
        )[0]
        new_lower = v3._down_add(
            np.zeros(1, dtype=np.float64),
            -result.final_guard[query_index : query_index + 1],
            where="V5.1 test new lower",
        )[0]
        case.assertGreaterEqual(new_lower, old_lower)

    masses = tuple(_fraction(value) for value in max_abs)
    exact_support = tuple(
        sum(
            (
                abs(_fraction(weight[i, j])) * masses[j]
                for j in range(weight.shape[1])
            ),
            Fraction(0),
        )
        for i in range(weight.shape[0])
    )
    for stored, exact_value in zip(
        support.support_upper, exact_support
    ):
        case.assertGreaterEqual(_fraction(stored), exact_value)
    case.assertGreaterEqual(
        _fraction(support.box_mass_upper),
        sum(masses, Fraction(0)),
    )
    for query_index, stored in enumerate(result.support_mass_upper):
        exact_mass = sum(
            (
                abs(_fraction(coefficients[query_index, i]))
                * exact_support[i]
                for i in range(coefficients.shape[1])
            ),
            Fraction(0),
        )
        case.assertGreaterEqual(_fraction(stored), exact_mass)
    return support, result, v3_guard, exact, errors


class DenseScalarGuardV51Tests(unittest.TestCase):
    def test_platform_and_dot_up_fraction_enclosures(self):
        diagnostics = check_v51_platform()
        values = diagnostics.as_dict()
        self.assertEqual(values["proof_authority"], "False")
        self.assertEqual(values["integration_gate"], "not-authoritative")
        self.assertGreaterEqual(
            int(values["longdouble_nmant"])
            - int(values["binary64_nmant"]),
            8,
        )

        left = np.asarray(
            [
                [1.0e16, 1.0, 1.0e16],
                [_ETA, 0.5, 0.0],
                [0.0, -0.0, 0.0],
            ],
            dtype=_F64,
        )
        right = np.asarray([1.0, _ETA, 2.0], dtype=_F64)
        upper = dot_up_longdouble(left, right)
        for row, stored in zip(left, upper):
            exact = sum(
                (
                    _fraction(x) * _fraction(y)
                    for x, y in zip(row, right)
                ),
                Fraction(0),
            )
            self.assertGreaterEqual(_fraction(stored), exact)
        self.assertEqual(upper[2], 0.0)

    def test_fixed_large_cancellation_is_sound_and_tighter_than_v3(self):
        coefficients = np.asarray(
            [[1.0e16, 1.0, -1.0e16]], dtype=_F64
        )
        weight = np.ones((3, 1), dtype=_F64)
        max_abs = np.ones(1, dtype=_F64)
        _, result, v3_guard, exact, errors = _audit_fraction(
            self, coefficients, weight, max_abs
        )
        self.assertEqual(exact[0][0], Fraction(1))
        self.assertEqual(result.nominal[0, 0], 0.0)
        self.assertEqual(errors[0], Fraction(1))
        self.assertFalse(result.fallback_mask[0])
        self.assertLess(result.final_guard[0], v3_guard[0])
        # The pre-registered conditional ceil keeps an already-upward
        # binary64 value; it must not add the rejected V5 unconditional
        # successor.  The prose observation ``...009`` came from that older
        # prototype, while the fixed formula evaluates to ``...008`` here.
        self.assertEqual(
            result.final_guard[0].hex(),
            "0x1.1c37937e08008p+4",
        )
        self.assertEqual(v3_guard[0].hex(), "0x1.1c37937e0800fp+4")

    def test_two_coordinate_large_cancellation(self):
        coefficients = np.asarray(
            [[1.0e16, 1.0, -1.0e16]], dtype=_F64
        )
        weight = np.asarray(
            [[1.0, -1.0], [1.0, 1.0], [1.0, -1.0]],
            dtype=_F64,
        )
        max_abs = np.asarray([1.0, 2.0], dtype=_F64)
        _, result, v3_guard, exact, errors = _audit_fraction(
            self, coefficients, weight, max_abs
        )
        self.assertEqual(exact[0], (Fraction(1), Fraction(1)))
        np.testing.assert_array_equal(
            result.nominal, np.zeros((1, 2), dtype=_F64)
        )
        self.assertEqual(errors[0], Fraction(3))
        self.assertLess(result.final_guard[0], v3_guard[0])

    def test_signed_zero_mixed_rows_and_disjoint_support(self):
        coefficients = np.asarray(
            [
                [0.0, -0.0],
                [1.0, -2.0],
                [0.0, 3.0],
            ],
            dtype=_F64,
        )
        weight = np.asarray(
            [[0.0, 5.0], [0.0, 0.0]], dtype=_F64
        )
        max_abs = np.asarray([7.0, 0.0], dtype=_F64)
        support, result, _, _, errors = _audit_fraction(
            self, coefficients, weight, max_abs, tile_width=1
        )
        np.testing.assert_array_equal(
            support.support_upper, np.zeros(2, dtype=_F64)
        )
        np.testing.assert_array_equal(
            result.final_guard, np.zeros(3, dtype=_F64)
        )
        np.testing.assert_array_equal(
            result.active_mask, np.zeros(3, dtype=np.bool_)
        )
        np.testing.assert_array_equal(
            result.fallback_mask, np.zeros(3, dtype=np.bool_)
        )
        self.assertEqual(errors, (Fraction(0),) * 3)

    def test_minimum_subnormal_uses_streamed_fallback(self):
        coefficients = np.asarray([[_ETA]], dtype=_F64)
        weight = np.asarray([[0.5]], dtype=_F64)
        max_abs = np.ones(1, dtype=_F64)
        _, result, v3_guard, exact, errors = _audit_fraction(
            self, coefficients, weight, max_abs, tile_width=1
        )
        self.assertEqual(exact[0][0], _fraction(_ETA) / 2)
        self.assertEqual(result.nominal[0, 0], 0.0)
        self.assertEqual(errors[0], _fraction(_ETA) / 2)
        self.assertTrue(result.fallback_mask[0])
        self.assertIn(
            "coefficient_subnormal", result.fallback_reasons[0]
        )
        self.assertLessEqual(result.final_guard[0], v3_guard[0])
        self.assertGreater(result.final_guard[0], 0.0)

    def test_normal_operands_with_potential_underflow_use_fallback(self):
        minimum_normal = float(np.finfo(np.float64).tiny)
        coefficients = np.asarray([[minimum_normal]], dtype=_F64)
        weight = np.asarray([[0.5]], dtype=_F64)
        max_abs = np.ones(1, dtype=_F64)
        _, result, _, _, _ = _audit_fraction(
            self, coefficients, weight, max_abs, tile_width=1
        )
        self.assertTrue(result.fallback_mask[0])
        self.assertTrue(
            {
                "nominal_product_underflow_risk",
                "support_mass_underflow_risk",
            }
            & set(result.fallback_reasons[0])
        )

    def test_active_row_with_disjoint_box_mass_uses_fallback(self):
        minimum_normal = float(np.finfo(np.float64).tiny)
        coefficients = np.asarray([[minimum_normal]], dtype=_F64)
        weight = np.asarray(
            [[1.0, 0.0, -2.0]], dtype=_F64
        )
        max_abs = np.asarray(
            [minimum_normal, 1.0e16, 1.0], dtype=_F64
        )
        support, result, _, _, _ = _audit_fraction(
            self, coefficients, weight, max_abs, tile_width=1
        )
        self.assertTrue(support.disjoint_box_mass)
        self.assertTrue(result.fallback_mask[0])
        self.assertIn(
            "disjoint_box_mass", result.fallback_reasons[0]
        )

    def test_bytes_binding_deadline_and_overflow_rejection(self):
        weight = np.asarray(
            [[1.0, -2.0], [3.0, 4.0]], dtype=_F64
        )
        max_abs = np.asarray([2.0, 5.0], dtype=_F64)
        coefficients = np.asarray([[0.5, -0.25]], dtype=_F64)
        support = prepare_dense_support_v51(
            weight, max_abs, binding={"frame": "a" * 64}
        )
        result = dense_support_compressed_guard_v51(
            coefficients, weight, max_abs, support
        )
        for array in (
            support.support_upper,
            result.nominal,
            result.support_mass_upper,
            result.wide_guard,
            result.streamed_v3_guard,
            result.final_guard,
            result.active_mask,
            result.fallback_mask,
        ):
            self.assertFalse(array.flags.writeable)
            with self.assertRaises(ValueError):
                array.setflags(write=True)
        self.assertIsInstance(hash(support.diagnostics), int)
        self.assertIsInstance(hash(result.diagnostics), int)
        self.assertFalse(result.proof_authority)

        changed_weight = weight.copy()
        changed_weight[0, 0] = 2.0
        with self.assertRaisesRegex(
            QueryDualScalarGuardV51Error, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard_v51(
                coefficients, changed_weight, max_abs, support
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardV51Error, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard_v51(
                coefficients, weight, max_abs + 1.0, support
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardV51Error, "DEADLINE_EXPIRED"
        ):
            prepare_dense_support_v51(
                weight, max_abs, deadline=time.monotonic() - 1.0
            )
        with self.assertRaisesRegex(
            QueryDualScalarGuardV51Error, "INVALID_DEADLINE"
        ):
            prepare_dense_support_v51(
                weight, max_abs, deadline=math.inf
            )

        huge = float(np.finfo(np.float64).max)
        overflow_weight = np.asarray([[huge]], dtype=_F64)
        overflow_mass = np.asarray([0.5], dtype=_F64)
        overflow_support = prepare_dense_support_v51(
            overflow_weight, overflow_mass
        )
        with np.errstate(over="ignore", invalid="ignore"):
            with self.assertRaisesRegex(
                QueryDualScalarGuardV51Error, "NONFINITE"
            ):
                dense_support_compressed_guard_v51(
                    np.asarray([[2.0]], dtype=_F64),
                    overflow_weight,
                    overflow_mass,
                    overflow_support,
                )

        forged = replace(
            support,
            support_upper=np.asarray(
                support.support_upper, dtype=_F64
            ).copy(),
        )
        with self.assertRaisesRegex(
            QueryDualScalarGuardV51Error, "INVALID_SUPPORT"
        ):
            dense_support_compressed_guard_v51(
                coefficients, weight, max_abs, forged
            )
        forged_mass = replace(support, box_mass_upper=0.0)
        with self.assertRaisesRegex(
            QueryDualScalarGuardV51Error, "BINDING_MISMATCH"
        ):
            dense_support_compressed_guard_v51(
                coefficients, weight, max_abs, forged_mass
            )

    def test_fixed_5000_fraction_rows_and_v3_tightness(self):
        rng = np.random.default_rng(20260728051)
        audited_rows = 0
        fallback_rows = 0
        for case_index in range(100):
            output_width = int(rng.integers(1, 9))
            input_width = int(rng.integers(1, 9))
            coefficients = np.ascontiguousarray(
                rng.normal(size=(50, output_width)), dtype=_F64
            )
            weight = np.ascontiguousarray(
                rng.normal(size=(output_width, input_width)), dtype=_F64
            )
            max_abs = np.ascontiguousarray(
                rng.uniform(0.0, 3.0, size=input_width), dtype=_F64
            )
            with self.subTest(case=case_index):
                _, result, _, _, _ = _audit_fraction(
                    self,
                    coefficients,
                    weight,
                    max_abs,
                    tile_width=3,
                )
            audited_rows += coefficients.shape[0]
            fallback_rows += int(np.count_nonzero(result.fallback_mask))
        self.assertEqual(audited_rows, 5_000)
        self.assertEqual(fallback_rows, 0)

    def test_fixed_1000_mixed_underflow_fraction_rows(self):
        rng = np.random.default_rng(510051)
        values = np.asarray(
            [
                0.0,
                -0.0,
                _ETA,
                -_ETA,
                np.finfo(np.float64).tiny,
                -np.finfo(np.float64).tiny,
                0.5,
                -0.5,
                1.0,
                -1.0,
                2.0,
                -2.0,
                1.0e16,
                -1.0e16,
            ],
            dtype=_F64,
        )
        audited_rows = 0
        fallback_rows = 0
        for case_index in range(200):
            output_width = int(rng.integers(1, 7))
            input_width = int(rng.integers(1, 7))
            coefficients = np.ascontiguousarray(
                rng.choice(values, size=(5, output_width)),
                dtype=_F64,
            )
            weight = np.ascontiguousarray(
                rng.choice(values, size=(output_width, input_width)),
                dtype=_F64,
            )
            max_abs = np.ascontiguousarray(
                np.abs(rng.choice(values, size=input_width)),
                dtype=_F64,
            )
            with self.subTest(case=case_index):
                _, result, _, _, _ = _audit_fraction(
                    self,
                    coefficients,
                    weight,
                    max_abs,
                    tile_width=2,
                )
            audited_rows += coefficients.shape[0]
            fallback_rows += int(np.count_nonzero(result.fallback_mask))
        self.assertEqual(audited_rows, 1_000)
        self.assertGreater(fallback_rows, 0)

    def test_controlled_classifier_shape_zero_fallback(self):
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
        support = prepare_dense_support_v51(
            weight,
            max_abs,
            binding={"artifact": "synthetic-64x512x1000"},
            deadline=deadline,
        )
        result = dense_support_compressed_guard_v51(
            coefficients,
            weight,
            max_abs,
            support,
            deadline=deadline,
        )
        elapsed = time.perf_counter() - started
        v3_nominal, _ = v3._matrix_product_with_error(
            coefficients, weight
        )
        np.testing.assert_array_equal(
            result.nominal.view(np.uint64),
            v3_nominal.view(np.uint64),
        )
        self.assertEqual(np.count_nonzero(result.fallback_mask), 0)
        self.assertEqual(
            result.diagnostics.as_dict()["fallback_rows"], "0"
        )
        self.assertLess(elapsed, 30.0)


if __name__ == "__main__":
    unittest.main()
