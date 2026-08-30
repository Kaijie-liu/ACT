from __future__ import annotations

from fractions import Fraction
import inspect
import math
import time
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import phase_projection_incremental_repair as _repair
from act.back_end.hybridz_tf.phase_projection_highs_owner import (
    FrozenRows,
    _project_new_column_tiny,
)


def _csr(values: np.ndarray) -> sp.csr_matrix:
    matrix = sp.csr_matrix(np.asarray(values, dtype=np.float64))
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    matrix.indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    return matrix


def _fraction_down(value: Fraction) -> float:
    rounded = float(value)
    if Fraction.from_float(rounded) > value:
        rounded = float(np.nextafter(rounded, -np.inf))
    return rounded


def _base_rows(
    logical: sp.csr_matrix,
    centers: np.ndarray,
    active: np.ndarray,
    keep: np.ndarray,
    row_ids: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
) -> FrozenRows:
    orientation = np.where(active, -1.0, 1.0)
    rhs = -orientation * centers
    selected = np.flatnonzero(keep)
    matrix = logical[selected].tocsr()
    matrix.sum_duplicates()
    matrix.eliminate_zeros()
    matrix.sort_indices()
    matrix.indptr = np.ascontiguousarray(matrix.indptr, dtype=np.int32)
    matrix.indices = np.ascontiguousarray(matrix.indices, dtype=np.int32)
    matrix.data = np.ascontiguousarray(matrix.data, dtype=np.float64)
    return FrozenRows.from_csr(
        matrix,
        row_lower=np.full(selected.size, -np.inf, dtype=np.float64),
        row_upper=np.ascontiguousarray(rhs[selected], dtype=np.float64),
        row_ids=np.ascontiguousarray(row_ids[selected], dtype=np.int64),
        column_lower=np.ascontiguousarray(lower, dtype=np.float64),
        column_upper=np.ascontiguousarray(upper, dtype=np.float64),
    )


def _build(
    logical: sp.csr_matrix,
    centers: np.ndarray,
    active: np.ndarray,
    keep: np.ndarray,
    row_ids: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    selected: np.ndarray,
    delta: np.ndarray,
    objective: np.ndarray,
):
    base = _base_rows(
        logical, centers, active, keep, row_ids, lower, upper
    )
    plan = _repair.build_incremental_repair(
        full_oriented_rows=logical,
        phase_centers=np.ascontiguousarray(centers, dtype=np.float64),
        base_active=np.ascontiguousarray(active, dtype=np.bool_),
        keep=np.ascontiguousarray(keep, dtype=np.bool_),
        full_row_ids=np.ascontiguousarray(row_ids, dtype=np.int64),
        base_rows=base,
        x_lower=np.ascontiguousarray(lower, dtype=np.float64),
        x_upper=np.ascontiguousarray(upper, dtype=np.float64),
        selected_ordinals=np.ascontiguousarray(selected, dtype=np.int64),
        delta=np.ascontiguousarray(delta, dtype=np.float64),
        objective_delta=np.ascontiguousarray(objective, dtype=np.float64),
        deadline_monotonic=float(time.monotonic() + 10.0),
    )
    return base, plan


class IncrementalRepairBuilderTests(unittest.TestCase):
    def test_up_down_cross_layer_and_same_layer_definitions(self):
        logical = _csr(
            np.asarray(
                [
                    [1.0, 0.0],
                    [2.0, -1.0],
                    [-1.0, 0.5],
                    [0.25, 1.5],
                ]
            )
        )
        centers = np.asarray([-2.0, 0.2, -0.3, 0.1], dtype=np.float64)
        active = np.asarray([False, False, True, False], dtype=np.bool_)
        keep = np.ones(4, dtype=np.bool_)
        row_ids = np.asarray([41, 83, 107, 211], dtype=np.int64)
        selected = np.asarray([1, 2, 3], dtype=np.int64)
        delta = np.zeros((4, 3), dtype=np.float64)
        # Column zero is an earlier-layer correction.  Rows two and three are
        # treated as same-layer in this toy, hence the row-three/column-one
        # influence stays exactly zero.
        delta[2, 0] = 0.5
        delta[3, 0] = 0.25
        objective = np.asarray([0.2, 0.0, -0.3], dtype=np.float64)
        base, plan = _build(
            logical,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0, -2.0]),
            np.asarray([1.0, 2.0]),
            selected,
            delta,
            objective,
        )

        plan.assert_intact()
        self.assertEqual(plan.missing_rows_appended, 0)
        self.assertEqual(plan.definition_rows_appended, 3)
        self.assertTrue(np.array_equal(plan.selected_base_row_positions, selected))
        self.assertTrue(
            np.array_equal(plan.updated_active, np.asarray([False, True, False, True]))
        )
        self.assertTrue(np.array_equal(plan.objective_minimize_cost, -objective))
        expected_existing = np.zeros((4, 3), dtype=np.float64)
        expected_existing[2, 0] = -0.5
        expected_existing[3, 0] = 0.25
        self.assertTrue(
            np.array_equal(
                sp.csc_matrix(
                    (
                        plan.new_columns.data,
                        plan.new_columns.indices,
                        plan.new_columns.indptr,
                    ),
                    shape=(4, 3),
                ).toarray(),
                expected_existing,
            )
        )
        dense = sp.csr_matrix(
            (
                plan.appended_rows.data,
                plan.appended_rows.indices,
                plan.appended_rows.indptr,
            ),
            shape=(plan.appended_rows.rows, plan.appended_rows.columns),
        ).toarray()
        expected_definitions = np.asarray(
            [
                [-2.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, -0.5, 0.0, 1.0, 0.0],
                [-0.25, -1.5, -0.25, 0.0, 1.0],
            ]
        )
        self.assertTrue(np.array_equal(dense, expected_definitions))
        self.assertTrue(
            np.array_equal(
                plan.appended_rows.lower, np.asarray([0.2, 0.3, 0.1])
            )
        )
        self.assertTrue(np.array_equal(plan.appended_rows.lower, plan.appended_rows.upper))
        self.assertTrue(np.array_equal(plan.existing_row_upper[[0]], base.upper[[0]]))
        self.assertTrue(np.all(np.isposinf(plan.existing_row_upper[selected])))

    def test_missing_screen_drops_only_strictly_redundant_row(self):
        logical = _csr(
            np.asarray(
                [
                    [1.0],
                    [-1.0],
                    [0.0],
                    [1.0],
                ]
            )
        )
        centers = np.asarray([0.0, 0.0, -1.0, -1.0], dtype=np.float64)
        active = np.zeros(4, dtype=np.bool_)
        keep = np.asarray([True, True, False, False], dtype=np.bool_)
        row_ids = np.asarray([10, 11, 12, 13], dtype=np.int64)
        _base, plan = _build(
            logical,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
            np.asarray([0], dtype=np.int64),
            np.zeros((4, 1), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )
        # Row two has exact upper zero < rhs one.  Row three reaches the rhs;
        # equality is critical and is conservatively appended.
        self.assertTrue(np.array_equal(plan.missing_ordinals, np.asarray([3])))
        self.assertEqual(plan.missing_rows_appended, 1)
        self.assertEqual(int(plan.appended_rows.row_ids[0]), 13)
        self.assertEqual(int(plan.appended_rows.row_ids[1]), 14)

    def test_x_and_y_tiny_projection_are_applied_once_each(self):
        x_tiny = float(5.0e-13)
        y_tiny = float(4.0e-13)
        logical = _csr(np.asarray([[1.0], [x_tiny]], dtype=np.float64))
        centers = np.asarray([0.0, 0.25], dtype=np.float64)
        active = np.asarray([False, True], dtype=np.bool_)
        keep = np.ones(2, dtype=np.bool_)
        row_ids = np.asarray([7, 9], dtype=np.int64)
        delta = np.zeros((2, 2), dtype=np.float64)
        delta[1, 0] = y_tiny
        _base, plan = _build(
            logical,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
            np.asarray([0, 1], dtype=np.int64),
            delta,
            np.zeros(2, dtype=np.float64),
        )
        position = int(plan.selected_base_row_positions[1])
        x_only_exact = Fraction.from_float(0.25) - Fraction.from_float(x_tiny)
        self.assertEqual(
            float(plan.existing_row_lower[position]), _fraction_down(x_only_exact)
        )
        projected = _project_new_column_tiny(
            plan.new_columns,
            plan.existing_row_lower,
            plan.existing_row_upper,
        )
        # The existing coefficient is orientation(-1) * D, and y0 encloses
        # [-1, 1], so its maximum deleted contribution is +y_tiny.
        # The owner starts from the caller's already outward-rounded x-only
        # bound, then performs its own exact-dyadic y projection.
        final_exact = Fraction.from_float(
            float(plan.existing_row_lower[position])
        ) - Fraction.from_float(y_tiny)
        self.assertEqual(
            float(projected.row_lower[position]), _fraction_down(final_exact)
        )
        self.assertEqual(plan.new_columns.data.size, 1)
        self.assertEqual(float(plan.new_columns.data[0]), -y_tiny)

    def test_duplicate_missing_coefficients_keep_distinct_row_ids(self):
        logical = _csr(np.asarray([[1.0], [1.0], [1.0]], dtype=np.float64))
        centers = np.asarray([-1.0, -1.0, -1.0], dtype=np.float64)
        active = np.zeros(3, dtype=np.bool_)
        keep = np.asarray([True, False, False], dtype=np.bool_)
        row_ids = np.asarray([101, 205, 999], dtype=np.int64)
        _base, plan = _build(
            logical,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
            np.asarray([0], dtype=np.int64),
            np.zeros((3, 1), dtype=np.float64),
            np.zeros(1, dtype=np.float64),
        )
        self.assertTrue(np.array_equal(plan.missing_ordinals, np.asarray([1, 2])))
        self.assertTrue(
            np.array_equal(plan.appended_rows.row_ids[:2], np.asarray([205, 999]))
        )
        dense = sp.csr_matrix(
            (
                plan.appended_rows.data,
                plan.appended_rows.indices,
                plan.appended_rows.indptr,
            ),
            shape=(plan.appended_rows.rows, plan.appended_rows.columns),
        ).toarray()
        self.assertTrue(np.array_equal(dense[0], dense[1]))
        self.assertEqual(np.unique(plan.appended_rows.row_ids).size, 3)

    def test_shape_deadline_and_aba_fail_closed(self):
        logical = _csr(np.asarray([[1.0], [2.0]], dtype=np.float64))
        centers = np.asarray([0.0, -1.0], dtype=np.float64)
        active = np.zeros(2, dtype=np.bool_)
        keep = np.ones(2, dtype=np.bool_)
        row_ids = np.asarray([1, 2], dtype=np.int64)
        base = _base_rows(
            logical,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
        )
        common = dict(
            full_oriented_rows=logical,
            phase_centers=np.ascontiguousarray(centers),
            base_active=np.ascontiguousarray(active),
            keep=np.ascontiguousarray(keep),
            full_row_ids=np.ascontiguousarray(row_ids),
            base_rows=base,
            x_lower=np.asarray([-1.0], dtype=np.float64),
            x_upper=np.asarray([1.0], dtype=np.float64),
            selected_ordinals=np.asarray([0], dtype=np.int64),
            delta=np.zeros((2, 1), dtype=np.float64),
            objective_delta=np.zeros(1, dtype=np.float64),
        )
        with self.assertRaises(_repair.IncrementalRepairUnknown):
            _repair.build_incremental_repair(
                **common, deadline_monotonic=float(time.monotonic() - 1.0)
            )
        malformed = dict(common)
        malformed["delta"] = np.zeros((1, 1), dtype=np.float64)
        with self.assertRaises(_repair.IncrementalRepairUnknown):
            _repair.build_incremental_repair(
                **malformed, deadline_monotonic=float(time.monotonic() + 10.0)
            )

        raw_delta = np.zeros((2, 1), dtype=np.float64)
        aba = dict(common)
        aba["delta"] = raw_delta
        original_deadline = _repair._check_deadline

        delta_changed = False

        def mutate_then_restore_delta(deadline: float, stage: str) -> None:
            nonlocal delta_changed
            if stage == "existing-row auxiliary CSC" and not delta_changed:
                raw_delta[1, 0] = 0.125
                delta_changed = True
            elif stage == "final ABA validation":
                raw_delta[1, 0] = 0.0
            original_deadline(deadline, stage)

        with mock.patch.object(
            _repair, "_check_deadline", side_effect=mutate_then_restore_delta
        ):
            delta_plan = _repair.build_incremental_repair(
                **aba, deadline_monotonic=float(time.monotonic() + 10.0)
            )
        self.assertTrue(delta_changed)
        self.assertEqual(float(raw_delta[1, 0]), 0.0)
        # The late caller alias value never enters the sealed CSC.
        self.assertEqual(delta_plan.new_columns.data.size, 0)
        delta_plan.assert_intact()

        logical_aba = _csr(np.asarray([[1.0], [2.0]], dtype=np.float64))
        matrix_aba = dict(common)
        matrix_aba["full_oriented_rows"] = logical_aba
        matrix_aba["base_rows"] = _base_rows(
            logical_aba,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
        )

        logical_changed = False

        def mutate_then_restore_matrix(deadline: float, stage: str) -> None:
            nonlocal logical_changed
            if stage == "append correction definitions" and not logical_changed:
                logical_aba.data[0] = 9.0
                logical_changed = True
            elif stage == "final ABA validation":
                logical_aba.data[0] = 1.0
            original_deadline(deadline, stage)

        with mock.patch.object(
            _repair, "_check_deadline", side_effect=mutate_then_restore_matrix
        ):
            matrix_plan = _repair.build_incremental_repair(
                **matrix_aba,
                deadline_monotonic=float(time.monotonic() + 10.0),
            )
        self.assertTrue(logical_changed)
        self.assertEqual(float(logical_aba.data[0]), 1.0)
        definition = sp.csr_matrix(
            (
                matrix_plan.appended_rows.data,
                matrix_plan.appended_rows.indices,
                matrix_plan.appended_rows.indptr,
            ),
            shape=(
                matrix_plan.appended_rows.rows,
                matrix_plan.appended_rows.columns,
            ),
        ).toarray()[-1]
        self.assertEqual(float(definition[0]), -1.0)
        matrix_plan.assert_intact()

        def expire_after_plan_assert(deadline: float, stage: str) -> None:
            if stage == "return":
                raise _repair.IncrementalRepairUnknown(
                    "incremental repair deadline expired at return"
                )
            original_deadline(deadline, stage)

        with mock.patch.object(
            _repair, "_check_deadline", side_effect=expire_after_plan_assert
        ):
            with self.assertRaisesRegex(
                _repair.IncrementalRepairUnknown, "expired at return"
            ):
                _repair.build_incremental_repair(
                    **common,
                    deadline_monotonic=float(time.monotonic() + 10.0),
                )

    def test_hostile_selector_numeric_and_output_aliases_fail_closed(self):
        logical = _csr(np.asarray([[1.0], [2.0]], dtype=np.float64))
        centers = np.asarray([0.0, -1.0], dtype=np.float64)
        active = np.zeros(2, dtype=np.bool_)
        keep = np.ones(2, dtype=np.bool_)
        row_ids = np.asarray([31, 47], dtype=np.int64)
        base = _base_rows(
            logical,
            centers,
            active,
            keep,
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
        )
        common = dict(
            full_oriented_rows=logical,
            phase_centers=centers,
            base_active=active,
            keep=keep,
            full_row_ids=row_ids,
            base_rows=base,
            x_lower=np.asarray([-1.0], dtype=np.float64),
            x_upper=np.asarray([1.0], dtype=np.float64),
            selected_ordinals=np.asarray([0], dtype=np.int64),
            delta=np.zeros((2, 1), dtype=np.float64),
            objective_delta=np.zeros(1, dtype=np.float64),
            deadline_monotonic=float(time.monotonic() + 10.0),
        )
        noncausal = dict(common)
        noncausal_delta = np.zeros((2, 1), dtype=np.float64)
        noncausal_delta[0, 0] = 0.125
        noncausal["delta"] = noncausal_delta
        with self.assertRaisesRegex(
            _repair.IncrementalRepairUnknown, "causality"
        ):
            _repair.build_incremental_repair(**noncausal)

        tiny_objective = dict(common)
        tiny_objective["objective_delta"] = np.asarray([5.0e-13])
        with self.assertRaises(_repair.IncrementalRepairUnknown):
            _repair.build_incremental_repair(**tiny_objective)

        omitted_selector = dict(common)
        omitted_selector["keep"] = np.asarray([False, True], dtype=np.bool_)
        omitted_selector["base_rows"] = _base_rows(
            logical,
            centers,
            active,
            omitted_selector["keep"],
            row_ids,
            np.asarray([-1.0]),
            np.asarray([1.0]),
        )
        with self.assertRaises(_repair.IncrementalRepairUnknown):
            _repair.build_incremental_repair(**omitted_selector)

        plan = _repair.build_incremental_repair(**common)
        centers[:] = 99.0
        row_ids[:] = -99
        common["delta"][1, 0] = 0.5
        self.assertTrue(np.array_equal(plan.selected_ordinals, np.asarray([0])))
        self.assertTrue(np.array_equal(plan.new_columns.row_ids, np.asarray([31, 47])))
        plan.assert_intact()
        with self.assertRaises(ValueError):
            plan.updated_active.setflags(write=True)

    def test_source_has_no_solver_or_existing_x_reassembly_path(self):
        source = inspect.getsource(_repair)
        self.assertNotIn("Highs()", source)
        self.assertNotIn("linprog", source)
        self.assertNotIn("sp.hstack", source)
        self.assertNotIn("changeCoeff", source)
        self.assertNotIn("clearModel", source)


if __name__ == "__main__":
    unittest.main()
