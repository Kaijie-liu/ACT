#!/usr/bin/env python3
"""Exact arithmetic gates for the HybridZ binary RHS transform."""

from __future__ import annotations

from fractions import Fraction
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

from act.back_end.solver import solver_hz as solver_hz_module
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _base_milp_matrices_from_blocks,
    _binary_shift_rhs_exact_or_outward,
    _tagged_upper_band_compaction_plan,
    hz_mark_constructively_nonempty,
    hz_objbound_decide,
)


def _fraction_reference(
    base: np.ndarray,
    matrix: sp.csr_matrix,
    *,
    equality: bool,
) -> np.ndarray:
    base = np.asarray(base, dtype=np.float64).reshape(-1)
    matrix = sp.csr_matrix(matrix, dtype=np.float64)
    out = np.empty(base.size, dtype=np.float64)
    for row in range(base.size):
        exact = Fraction.from_float(float(base[row]))
        start = int(matrix.indptr[row])
        end = int(matrix.indptr[row + 1])
        for value in matrix.data[start:end]:
            exact += Fraction.from_float(float(value))
        rounded = float(exact)
        rounded_exact = Fraction.from_float(rounded)
        if equality and rounded_exact != exact:
            raise ValueError("inexact equality")
        if not equality and rounded_exact < exact:
            rounded = float(np.nextafter(rounded, np.inf))
        out[row] = rounded
    return out


class SolverHZBinaryShiftTests(unittest.TestCase):
    def test_zero_and_one_binary_rows_match_fraction_bitwise(self) -> None:
        rng = np.random.default_rng(0x2A520260811)
        edge = np.asarray(
            [
                0.0,
                -0.0,
                np.nextafter(0.0, 1.0),
                -np.nextafter(0.0, 1.0),
                np.finfo(np.float64).tiny,
                -np.finfo(np.float64).tiny,
                1.0,
                -1.0,
                np.nextafter(1.0, 2.0),
                np.finfo(np.float64).max / 2.0,
                -np.finfo(np.float64).max / 2.0,
            ],
            dtype=np.float64,
        )
        pairs = [(left, right) for left in edge for right in edge]
        raw = rng.integers(
            0, 2**64, size=(8192, 2), dtype=np.uint64
        ).view(np.float64)
        raw = raw[np.all(np.isfinite(raw), axis=1)]
        pairs.extend((row[0], row[1]) for row in raw[:4096])

        base = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
        coefficient = np.asarray(
            [pair[1] for pair in pairs], dtype=np.float64
        )
        finite_sum = np.isfinite(base + coefficient)
        base = base[finite_sum]
        coefficient = coefficient[finite_sum]
        rows = np.arange(base.size, dtype=np.int32)
        matrix = sp.csr_matrix(
            (coefficient, (rows, rows)),
            shape=(base.size, base.size),
            dtype=np.float64,
        )
        expected = _fraction_reference(base, matrix, equality=False)
        actual = _binary_shift_rhs_exact_or_outward(
            base, matrix, equality=False
        )
        np.testing.assert_array_equal(
            actual.view(np.uint64), expected.view(np.uint64)
        )

        empty = sp.csr_matrix((base.size, 0), dtype=np.float64)
        unchanged = _binary_shift_rhs_exact_or_outward(
            base, empty, equality=False
        )
        empty_reference = _fraction_reference(
            base, empty, equality=False
        )
        np.testing.assert_array_equal(
            unchanged.view(np.uint64), empty_reference.view(np.uint64)
        )

    def test_equality_accepts_exact_and_rejects_inexact_two_sum(self) -> None:
        exact_base = np.asarray([0.5, -1.0, -0.0], dtype=np.float64)
        exact_data = np.asarray([0.25, 0.5, -0.0], dtype=np.float64)
        rows = np.arange(3, dtype=np.int32)
        exact_matrix = sp.csr_matrix(
            (exact_data, (rows, rows)), shape=(3, 3)
        )
        actual = _binary_shift_rhs_exact_or_outward(
            exact_base, exact_matrix, equality=True
        )
        expected = _fraction_reference(
            exact_base, exact_matrix, equality=True
        )
        np.testing.assert_array_equal(
            actual.view(np.uint64), expected.view(np.uint64)
        )

        inexact_base = np.asarray([1.0], dtype=np.float64)
        inexact_matrix = sp.csr_matrix(
            np.asarray([[2.0**-54]], dtype=np.float64)
        )
        with self.assertRaisesRegex(ValueError, "not exactly representable"):
            _binary_shift_rhs_exact_or_outward(
                inexact_base, inexact_matrix, equality=True
            )

    def test_multiple_binary_terms_keep_fraction_fallback(self) -> None:
        base = np.asarray([1.0, -0.25, 0.0], dtype=np.float64)
        matrix = sp.csr_matrix(
            np.asarray(
                [
                    [2.0**-54, -2.0**-55, 0.0],
                    [0.5, -0.25, 2.0**-52],
                    [2.0**48, 2.0**-5, -2.0**48],
                ],
                dtype=np.float64,
            )
        )
        expected = _fraction_reference(base, matrix, equality=False)
        actual = _binary_shift_rhs_exact_or_outward(
            base, matrix, equality=False
        )
        np.testing.assert_array_equal(
            actual.view(np.uint64), expected.view(np.uint64)
        )

    def test_one_term_overflow_fails_closed(self) -> None:
        maximum = np.finfo(np.float64).max
        with np.errstate(over="raise"):
            with self.assertRaisesRegex(ValueError, "overflows binary64"):
                _binary_shift_rhs_exact_or_outward(
                    np.asarray([maximum]),
                    sp.csr_matrix(np.asarray([[maximum]])),
                    equality=False,
                )

    def test_tagged_forward_reverse_band_compacts_exactly(self) -> None:
        forward_cont = np.asarray(
            [[1.0, -2.0], [3.0, 4.0]], dtype=np.float64
        )
        forward_bin = np.asarray([[0.5], [-0.25]], dtype=np.float64)
        Acl = sp.csr_matrix(
            np.vstack([forward_cont, -forward_cont])
        )
        Abl = sp.csr_matrix(
            np.vstack([forward_bin, -forward_bin])
        )
        upper = np.asarray([5.0, 6.0, 7.0, 8.0], dtype=np.float64)
        tags = (
            "affine_chain_cut:20:forward",
            "affine_chain_cut:20:forward",
            "affine_chain_cut:20:reverse",
            "affine_chain_cut:20:reverse",
        )
        plan = _tagged_upper_band_compaction_plan(Acl, Abl, tags)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["pair_count"], 2)
        self.assertEqual(
            plan["compacted_tags"],
            (
                "affine_chain_cut:20:range",
                "affine_chain_cut:20:range",
            ),
        )

        empty_eq_cont = sp.csr_matrix((0, 2), dtype=np.float64)
        empty_eq_bin = sp.csr_matrix((0, 1), dtype=np.float64)
        Gc = sp.csr_matrix((1, 2), dtype=np.float64)
        Gb = sp.csr_matrix((1, 1), dtype=np.float64)
        A, lower, ranged_upper, lb, ub, integ = (
            _base_milp_matrices_from_blocks(
                Gc,
                Gb,
                empty_eq_cont,
                empty_eq_bin,
                np.zeros(0, dtype=np.float64),
                Acl,
                Abl,
                upper,
                upper_compaction_plan=plan,
            )
        )
        shifted = _binary_shift_rhs_exact_or_outward(
            upper, Abl, equality=False
        )
        np.testing.assert_array_equal(
            lower.view(np.uint64), (-shifted[2:]).view(np.uint64)
        )
        np.testing.assert_array_equal(
            ranged_upper.view(np.uint64), shifted[:2].view(np.uint64)
        )
        np.testing.assert_array_equal(A.toarray()[:, :2], forward_cont)
        np.testing.assert_array_equal(
            A.toarray()[:, 2:], 2.0 * forward_bin
        )
        np.testing.assert_array_equal(lb, np.asarray([-1.0, -1.0, 0.0]))
        np.testing.assert_array_equal(ub, np.ones(3, dtype=np.float64))
        np.testing.assert_array_equal(integ, np.asarray([0, 0, 1]))

        for x0 in (-1.0, 0.0, 1.0):
            for x1 in (-1.0, 0.0, 1.0):
                for z in (0.0, 1.0):
                    point = np.asarray([x0, x1, z], dtype=np.float64)
                    compact_value = np.asarray(A @ point).reshape(-1)
                    xi = np.asarray([x0, x1, 2.0 * z - 1.0])
                    source_value = np.asarray(
                        sp.hstack([Acl, Abl], format="csr") @ xi
                    ).reshape(-1)
                    source_ok = bool(np.all(source_value <= upper))
                    compact_ok = bool(
                        np.all(compact_value >= lower)
                        and np.all(compact_value <= ranged_upper)
                    )
                    self.assertEqual(source_ok, compact_ok)

    def test_tagged_band_rejects_any_nonnegative_pair_or_orphan(self) -> None:
        forward = sp.csr_matrix(
            np.asarray([[1.0, -2.0]], dtype=np.float64)
        )
        wrong_reverse = sp.csr_matrix(
            np.asarray([[-1.0, 3.0]], dtype=np.float64)
        )
        matrix = sp.vstack([forward, wrong_reverse], format="csr")
        empty_binary = sp.csr_matrix((2, 0), dtype=np.float64)
        with self.assertRaisesRegex(ValueError, "bitwise coefficient"):
            _tagged_upper_band_compaction_plan(
                matrix,
                empty_binary,
                (
                    "affine_chain_cut:3:forward",
                    "affine_chain_cut:3:reverse",
                ),
            )
        with self.assertRaisesRegex(ValueError, "orphan"):
            _tagged_upper_band_compaction_plan(
                forward,
                sp.csr_matrix((1, 0), dtype=np.float64),
                ("affine_chain_cut:3:reverse",),
            )

    def test_stable_active_relu_equality_uses_the_same_range_path(self) -> None:
        forward_cont = sp.csr_matrix(
            np.asarray([[1.0, -1.0], [2.0, -2.0]], dtype=np.float64)
        )
        forward_bin = sp.csr_matrix(
            np.asarray([[0.5], [-0.25]], dtype=np.float64)
        )
        Acl = sp.vstack([forward_cont, -forward_cont], format="csr")
        Abl = sp.vstack([forward_bin, -forward_bin], format="csr")
        tags = (
            "relu_active:5:forward",
            "relu_active:5:forward",
            "relu_active:5:reverse",
            "relu_active:5:reverse",
        )
        plan = _tagged_upper_band_compaction_plan(Acl, Abl, tags)
        self.assertIsNotNone(plan)
        self.assertEqual(plan["pair_count"], 2)
        self.assertEqual(
            plan["compacted_tags"],
            ("relu_active:5:range", "relu_active:5:range"),
        )
        np.testing.assert_array_equal(plan["keep_rows"], [0, 1])
        np.testing.assert_array_equal(plan["reverse_rows"], [2, 3])
        np.testing.assert_array_equal(plan["lower_positions"], [0, 1])

    def test_unrelated_upper_tags_leave_primary_matrix_unchanged(self) -> None:
        Acl = sp.csr_matrix(np.asarray([[1.0], [-1.0]]))
        Abl = sp.csr_matrix((2, 0), dtype=np.float64)
        self.assertIsNone(
            _tagged_upper_band_compaction_plan(
                Acl,
                Abl,
                ("relu_exact_lower:2", "relu_exact_x_branch:2"),
            )
        )

    def test_objbound_entry_uses_one_native_ranged_row(self) -> None:
        hz = SparseHZono(
            c=np.asarray([0.0], dtype=np.float64),
            Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
            Gb=sp.csr_matrix((1, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, 1), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix([[1.0], [-1.0]], dtype=np.float64),
            Aub=sp.csr_matrix((2, 0), dtype=np.float64),
            ub=np.asarray([0.5, 0.5], dtype=np.float64),
            col_ids=np.asarray([11], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        hz_mark_constructively_nonempty(hz, "range_integration_test")
        hz._solver_constraint_row_tags = (
            "affine_chain_cut:7:forward",
            "affine_chain_cut:7:reverse",
        )

        real_highs = solver_hz_module._highspy.Highs
        observed_time_limits = []

        class RecordingHighs:
            def __init__(self):
                self._inner = real_highs()
                self.time_limits = []
                observed_time_limits.append(self.time_limits)

            def setOptionValue(self, name, value):
                if name == "time_limit":
                    self.time_limits.append(float(value))
                return self._inner.setOptionValue(name, value)

            def __getattr__(self, name):
                return getattr(self._inner, name)

        with patch.object(
            solver_hz_module._highspy,
            "Highs",
            side_effect=RecordingHighs,
        ):
            verdict, witness = hz_objbound_decide(
                hz,
                np.asarray([[1.0]], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
                is_unsafe_linear=True,
                time_limit=2.0,
            )
        self.assertEqual(verdict, "UNSAFE")
        self.assertIsNotNone(witness)
        self.assertLessEqual(abs(float(witness[0])), 0.5)
        stats = hz._solver_objbound_stats
        self.assertTrue(stats["parent_ranged_row_compaction_applied"])
        self.assertEqual(stats["parent_ranged_row_pair_count"], 1)
        self.assertEqual(stats["parent_base_matrix_rows"], 1)
        self.assertEqual(stats["parent_base_matrix_nnz"], 1)
        self.assertEqual(len(observed_time_limits), 1)
        self.assertEqual(len(observed_time_limits[0]), 2)
        self.assertGreater(observed_time_limits[0][0], 0.0)
        self.assertGreater(observed_time_limits[0][1], 0.0)
        self.assertLessEqual(
            observed_time_limits[0][1], observed_time_limits[0][0]
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
