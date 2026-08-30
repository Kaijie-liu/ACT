"""Bit-identity gates for low-risk query-dual replay micro-optimizations."""

from __future__ import annotations

import unittest

import numpy as np

from act.back_end.hybridz_tf import query_dual_replay as replay


def _candidate_nonnegative_row_dots_with_error(
    a: np.ndarray,
    b: np.ndarray,
):
    """Rejected V4 candidate, isolated here so frozen proof code is unchanged."""

    if a.ndim != 2 or b.ndim != 1 or b.size != a.shape[1]:
        replay._fail("SHAPE_MISMATCH", "invalid nonnegative row-dot operands")
    if np.any(a < 0.0) or np.any(b < 0.0):
        replay._fail(
            "NUMERIC_GUARD", "nonnegative row-dot operand is negative"
        )
    if not np.any(a) or not np.any(b):
        zeros = np.zeros(a.shape[0], dtype=np.float64)
        return zeros, zeros.copy()
    operations = 2 * int(a.shape[1]) + 2
    gamma = replay._gamma(operations)
    underflow = replay._underflow_allowance(operations)
    nominal = np.asarray(a @ b, dtype=np.float64)
    replay._require_finite(nominal, where="candidate nonnegative row dots")
    sum_upper = replay._upper_gamma_enclosure(
        nominal, gamma, underflow
    )
    radius = replay._upper_error_from_mass(
        sum_upper, gamma, underflow
    )
    radius[
        ~np.any((a != 0.0) & (b.reshape(1, -1) != 0.0), axis=1)
    ] = 0.0
    replay._require_finite(
        radius, where="candidate nonnegative row-dot error"
    )
    return np.ascontiguousarray(nominal), np.ascontiguousarray(radius)


def _candidate_absorb_radius(
    scalar: np.ndarray,
    radius: np.ndarray,
    box: replay._Box,
    stats: replay._ReplayStats,
) -> np.ndarray:
    if not np.any(radius):
        return scalar
    max_abs = np.maximum(np.abs(box.lb), np.abs(box.ub))
    nominal, raw_error = _candidate_nonnegative_row_dots_with_error(
        radius, max_abs
    )
    penalty = replay._upper_nonnegative_sum(nominal, raw_error)
    zero_rows = ~np.any(
        (radius != 0.0) & (max_abs.reshape(1, -1) != 0.0), axis=1
    )
    penalty[zero_rows] = 0.0
    replay._require_finite(
        penalty, where="candidate coefficient-error box absorption"
    )
    stats.record_guard(penalty, coefficient=True)
    if not np.any(penalty):
        return scalar
    return replay._down_add(
        scalar, -penalty, where="candidate coefficient-error absorption"
    )


def _legacy_absorb_radius(
    scalar: np.ndarray,
    radius: np.ndarray,
    box: replay._Box,
    stats: replay._ReplayStats,
) -> np.ndarray:
    """The pre-V4 implementation retained as an exact test oracle."""

    if not np.any(radius):
        return scalar
    max_abs = np.maximum(np.abs(box.lb), np.abs(box.ub))
    _, raw_error = replay._row_dots_with_error(radius, max_abs)
    nominal = np.asarray(radius @ max_abs, dtype=np.float64)
    penalty = replay._upper_nonnegative_sum(nominal, raw_error)
    zero_rows = ~np.any(
        (radius != 0.0) & (max_abs.reshape(1, -1) != 0.0), axis=1
    )
    penalty[zero_rows] = 0.0
    replay._require_finite(
        penalty, where="legacy coefficient-error box absorption"
    )
    stats.record_guard(penalty, coefficient=True)
    if not np.any(penalty):
        return scalar
    return replay._down_add(
        scalar, -penalty, where="legacy coefficient-error absorption"
    )


class QueryDualReplayRejectedV4MicroTests(unittest.TestCase):
    def test_rejected_nonnegative_row_dot_was_bit_identical(self):
        rng = np.random.default_rng(20260728)
        cases = [
            (
                np.abs(rng.normal(size=(7, 257))).astype(np.float64),
                np.abs(rng.normal(size=257)).astype(np.float64),
            ),
            (
                np.abs(rng.normal(size=(16, 2048))).astype(np.float64)
                * np.float64(1.0e-13),
                np.abs(rng.normal(size=2048)).astype(np.float64),
            ),
            (
                np.asarray(
                    [
                        [0.0, np.nextafter(0.0, 1.0), 2.0**-1022, 1.0],
                        [2.0**-1073, 0.0, 2.0**-1021, 2.0**500],
                    ],
                    dtype=np.float64,
                ),
                np.asarray(
                    [2.0**500, 1.0, 2.0**-500, 2.0**-500],
                    dtype=np.float64,
                ),
            ),
        ]
        for left, right in cases:
            with self.subTest(shape=left.shape):
                old_nominal, old_radius = replay._row_dots_with_error(
                    left, right
                )
                repeated_nominal = np.asarray(left @ right, dtype=np.float64)
                new_nominal, new_radius = (
                    _candidate_nonnegative_row_dots_with_error(left, right)
                )
                self.assertTrue(np.array_equal(old_nominal, repeated_nominal))
                self.assertTrue(np.array_equal(new_nominal, old_nominal))
                self.assertTrue(np.array_equal(new_radius, old_radius))
                self.assertEqual(
                    [float(value).hex() for value in new_radius],
                    [float(value).hex() for value in old_radius],
                )

    def test_rejected_absorb_radius_was_bit_identical_to_legacy_path(self):
        rng = np.random.default_rng(17)
        radius = (
            np.abs(rng.normal(size=(11, 513))).astype(np.float64)
            * np.float64(1.0e-14)
        )
        radius[0] = 0.0
        radius[:, ::67] = 0.0
        box = replay._Box(
            lb=-np.abs(rng.normal(size=513)).astype(np.float64),
            ub=np.abs(rng.normal(size=513)).astype(np.float64),
        )
        scalar = rng.normal(size=11).astype(np.float64)
        old_stats = replay._ReplayStats()
        new_stats = replay._ReplayStats()
        old_stats.configure_queries(11)
        new_stats.configure_queries(11)
        old_stats.begin_block(0, 11)
        new_stats.begin_block(0, 11)

        old = _legacy_absorb_radius(scalar.copy(), radius, box, old_stats)
        new = _candidate_absorb_radius(
            scalar.copy(), radius, box, new_stats
        )

        self.assertTrue(np.array_equal(new, old))
        self.assertEqual(
            [float(value).hex() for value in new],
            [float(value).hex() for value in old],
        )
        self.assertEqual(
            new_stats.coefficient_guards, old_stats.coefficient_guards
        )
        self.assertEqual(new_stats.scalar_guards, old_stats.scalar_guards)
        self.assertEqual(
            float(new_stats.guard_total).hex(),
            float(old_stats.guard_total).hex(),
        )
        self.assertEqual(
            float(new_stats.guard_max).hex(),
            float(old_stats.guard_max).hex(),
        )
        self.assertTrue(
            np.array_equal(
                new_stats.guard_by_query, old_stats.guard_by_query
            )
        )

    def test_rejected_specialization_failed_closed_on_negative_input(self):
        with self.assertRaisesRegex(
            replay.QueryDualReplayError, "NUMERIC_GUARD"
        ):
            _candidate_nonnegative_row_dots_with_error(
                np.asarray([[1.0, -1.0]], dtype=np.float64),
                np.asarray([1.0, 1.0], dtype=np.float64),
            )


if __name__ == "__main__":
    unittest.main()
