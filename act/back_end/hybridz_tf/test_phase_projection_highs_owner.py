#!/usr/bin/env python3
"""Focused CPU tests for the private phase-projection HiGHS owner."""

from __future__ import annotations

from fractions import Fraction
import time
import unittest
from unittest import mock

import highspy
import numpy as np
import scipy.sparse as sp

from act.back_end.hybridz_tf import phase_projection_highs_owner as _owner


_REAL_HIGHS = highspy.Highs


def _vector(values: list[float]) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.float64)


def _ids(values: list[int]) -> np.ndarray:
    return np.ascontiguousarray(values, dtype=np.int64)


def _csr(values: list[list[float]]) -> sp.csr_matrix:
    matrix = sp.csr_matrix(np.asarray(values, dtype=np.float64))
    matrix.sort_indices()
    assert matrix.indptr.dtype == np.dtype(np.int32)
    assert matrix.indices.dtype == np.dtype(np.int32)
    return matrix


def _csc(values: list[list[float]]) -> sp.csc_matrix:
    matrix = sp.csc_matrix(np.asarray(values, dtype=np.float64))
    matrix.sort_indices()
    assert matrix.indptr.dtype == np.dtype(np.int32)
    assert matrix.indices.dtype == np.dtype(np.int32)
    return matrix


def _rows(
    values: list[list[float]],
    upper: list[float],
    column_lower: list[float],
    column_upper: list[float],
    *,
    row_ids: list[int] | None = None,
    lower: list[float] | None = None,
) -> _owner.FrozenRows:
    count = len(values)
    return _owner.FrozenRows.from_csr(
        _csr(values),
        row_lower=_vector([-np.inf] * count if lower is None else lower),
        row_upper=_vector(upper),
        row_ids=_ids(list(range(count)) if row_ids is None else row_ids),
        column_lower=_vector(column_lower),
        column_upper=_vector(column_upper),
    )


def _new_columns(
    values: list[list[float]],
    cost: list[float],
    column_lower: list[float],
    column_upper: list[float],
    existing_row_ids: list[int],
) -> _owner.FrozenNewColumns:
    return _owner.FrozenNewColumns.from_csc(
        _csc(values),
        cost=_vector(cost),
        column_lower=_vector(column_lower),
        column_upper=_vector(column_upper),
        existing_row_ids=_ids(existing_row_ids),
    )


class _RecordingHighs:
    def __init__(self) -> None:
        self.inner = _REAL_HIGHS()
        self.clear_calls = 0
        self.clear_model_calls = 0
        self.add_cols_calls = 0
        self.add_rows_calls = 0
        self.run_calls = 0
        self.get_solution_calls = 0
        self.get_ray_exist_calls = 0
        self.get_ray_calls = 0
        self.change_rows_bounds_calls = 0

    def __getattr__(self, name: str):
        return getattr(self.inner, name)

    def clear(self):
        self.clear_calls += 1
        return self.inner.clear()

    def clearModel(self):
        self.clear_model_calls += 1
        return self.inner.clearModel()

    def addCols(self, *args):
        self.add_cols_calls += 1
        return self.inner.addCols(*args)

    def addRows(self, *args):
        self.add_rows_calls += 1
        return self.inner.addRows(*args)

    def run(self):
        self.run_calls += 1
        return self.inner.run()

    def getSolution(self):
        self.get_solution_calls += 1
        return self.inner.getSolution()

    def getDualRayExist(self):
        self.get_ray_exist_calls += 1
        return self.inner.getDualRayExist()

    def getDualRay(self):
        self.get_ray_calls += 1
        return self.inner.getDualRay()

    def changeRowsBounds(self, *args):
        self.change_rows_bounds_calls += 1
        return self.inner.changeRowsBounds(*args)


class PhaseProjectionHighsOwnerTests(unittest.TestCase):
    def test_one_owner_base_then_incremental_warm_update(self) -> None:
        backends: list[_RecordingHighs] = []

        def factory():
            backend = _RecordingHighs()
            backends.append(backend)
            return backend

        base_rows = _rows([[1.0]], [0.75], [-1.0], [1.0], row_ids=[10])
        new_columns = _new_columns([[1.0]], [-1.0], [-1.0], [1.0], [10])
        appended_rows = _rows(
            [[-1.0, 1.0]],
            [0.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            row_ids=[30],
            lower=[0.0],
        )
        with mock.patch.object(_owner.highspy, "Highs", side_effect=factory):
            owner = _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            )
            base = owner.solve_base(
                cost=_vector([-2.0]),
                column_lower=_vector([-1.0]),
                column_upper=_vector([1.0]),
                rows=base_rows,
            )
            self.assertIsInstance(base, _owner.OptimalSelector)
            np.testing.assert_array_equal(base.factors, [0.75])
            np.testing.assert_array_equal(base.row_value, [0.75])
            np.testing.assert_array_equal(base.row_dual, [-2.0])
            np.testing.assert_array_equal(base.row_ids, [10])
            self.assertEqual(owner.state, "BASE_SOLVED")

            updated = owner.apply_incremental_update(
                new_columns=new_columns,
                existing_row_lower=_vector([-np.inf]),
                existing_row_upper=_vector([1.0]),
                appended_rows=appended_rows,
            )
            self.assertIsInstance(updated, _owner.OptimalCandidate)
            np.testing.assert_allclose(updated.factors, [0.5, 0.5], atol=1.0e-12)
            self.assertAlmostEqual(updated.minimized_objective, -1.5, places=12)
            self.assertEqual(owner.state, "UPDATE_SOLVED")
            owner.close()

        self.assertEqual(len(backends), 1)
        backend = backends[0]
        self.assertEqual(backend.run_calls, 2)
        self.assertEqual(backend.clear_model_calls, 0)
        self.assertEqual(backend.change_rows_bounds_calls, 1)
        self.assertEqual(backend.get_solution_calls, 2)
        self.assertEqual(backend.get_ray_exist_calls, 0)
        self.assertEqual(backend.get_ray_calls, 0)
        self.assertEqual(backend.clear_calls, 1)
        self.assertEqual(owner.state, "CLOSED")
        with self.assertRaises(_owner.HighsOwnerUnknown):
            owner.solve_base(
                cost=_vector([-1.0]),
                column_lower=_vector([-1.0]),
                column_upper=_vector([1.0]),
                rows=base_rows,
            )

    def test_updated_infeasible_returns_unknown_without_second_ray(self) -> None:
        backend = _RecordingHighs()
        base_rows = _rows([[1.0]], [0.75], [-1.0], [1.0], row_ids=[10])
        new_columns = _new_columns([[1.0]], [0.0], [-1.0], [1.0], [10])
        impossible_definition = _rows(
            [[-1.0, 1.0]],
            [3.0],
            [-1.0, -1.0],
            [1.0, 1.0],
            row_ids=[31],
            lower=[3.0],
        )
        with mock.patch.object(_owner.highspy, "Highs", return_value=backend):
            with _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            ) as owner:
                base = owner.solve_base(
                    cost=_vector([-1.0]),
                    column_lower=_vector([-1.0]),
                    column_upper=_vector([1.0]),
                    rows=base_rows,
                )
                self.assertIsInstance(base, _owner.OptimalSelector)
                updated = owner.apply_incremental_update(
                    new_columns=new_columns,
                    existing_row_lower=_vector([-np.inf]),
                    existing_row_upper=_vector([1.0]),
                    appended_rows=impossible_definition,
                )
        self.assertIsInstance(updated, _owner.Unresolved)
        self.assertEqual(updated.model_status, highspy.HighsModelStatus.kInfeasible)
        self.assertEqual(backend.run_calls, 2)
        self.assertEqual(backend.clear_model_calls, 0)
        self.assertEqual(backend.get_ray_exist_calls, 0)
        self.assertEqual(backend.get_ray_calls, 0)

    def test_base_infeasible_ray_is_selector_only_and_mapped_once(self) -> None:
        backend = _RecordingHighs()
        rows = _rows(
            [[1.0], [1.0]],
            [-1.0, -2.0],
            [0.0],
            [1.0],
            row_ids=[101, 202],
        )
        with mock.patch.object(_owner.highspy, "Highs", return_value=backend):
            with _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            ) as owner:
                result = owner.solve_base(
                    cost=_vector([0.0]),
                    column_lower=_vector([0.0]),
                    column_upper=_vector([1.0]),
                    rows=rows,
                )
        self.assertIsInstance(result, _owner.InfeasibleRaySelector)
        self.assertEqual(backend.get_solution_calls, 0)
        self.assertEqual(backend.get_ray_exist_calls, 1)
        self.assertEqual(backend.get_ray_calls, 1)
        self.assertTrue(np.all(result.row_ray <= 0.0))
        self.assertTrue(np.any(result.row_ray != 0.0))
        self.assertEqual(
            result.support_row_ids,
            tuple(
                int(result.row_ids[index])
                for index in np.flatnonzero(result.row_ray != 0.0)
            ),
        )
        self.assertFalse(result.row_ray.flags.writeable)
        self.assertFalse(result.row_ids.flags.writeable)

    def test_base_infeasible_can_receive_the_same_incremental_transaction(self) -> None:
        backend = _RecordingHighs()
        base_rows = _rows([[1.0]], [-1.0], [0.0], [1.0], row_ids=[101])
        new_columns = _new_columns([[1.0]], [0.0], [-2.0], [0.0], [101])
        definition = _rows(
            [[0.0, 1.0]],
            [-1.0],
            [0.0, -2.0],
            [1.0, 0.0],
            row_ids=[300],
            lower=[-1.0],
        )
        with mock.patch.object(_owner.highspy, "Highs", return_value=backend):
            with _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            ) as owner:
                base = owner.solve_base(
                    cost=_vector([0.0]),
                    column_lower=_vector([0.0]),
                    column_upper=_vector([1.0]),
                    rows=base_rows,
                )
                self.assertIsInstance(base, _owner.InfeasibleRaySelector)
                updated = owner.apply_incremental_update(
                    new_columns=new_columns,
                    existing_row_lower=_vector([-np.inf]),
                    existing_row_upper=_vector([-1.0]),
                    appended_rows=definition,
                )
        self.assertIsInstance(updated, _owner.OptimalCandidate)
        np.testing.assert_allclose(updated.factors, [0.0, -1.0], atol=1.0e-12)
        self.assertEqual(backend.run_calls, 2)
        self.assertEqual(backend.get_ray_exist_calls, 1)
        self.assertEqual(backend.get_ray_calls, 1)
        self.assertEqual(backend.get_solution_calls, 1)

    def test_general_tiny_projection_is_exactly_outward_and_box_bound(self) -> None:
        source = _csr([[1.0, 1.0e-12]])
        column_lower = _vector([-1.0, -3.0])
        column_upper = _vector([1.0, 2.0])
        frozen = _owner.FrozenRows.from_csr(
            source,
            row_lower=_vector([-0.5]),
            row_upper=_vector([0.5]),
            row_ids=_ids([7]),
            column_lower=column_lower,
            column_upper=column_upper,
        )
        coefficient = Fraction.from_float(1.0e-12)
        deleted_min = min(coefficient * -3, coefficient * 2)
        deleted_max = max(coefficient * -3, coefficient * 2)
        exact_lower = Fraction.from_float(-0.5) - deleted_max
        exact_upper = Fraction.from_float(0.5) - deleted_min
        self.assertEqual(frozen.deleted_tiny_nnz, 1)
        self.assertEqual(frozen.logical_nnz, 2)
        np.testing.assert_array_equal(frozen.indices, [0])
        np.testing.assert_array_equal(frozen.data, [1.0])
        self.assertLessEqual(Fraction.from_float(float(frozen.lower[0])), exact_lower)
        self.assertGreaterEqual(Fraction.from_float(float(frozen.upper[0])), exact_upper)
        self.assertEqual(float(frozen.lower[0]), -0.5000000000020001)
        self.assertEqual(float(frozen.upper[0]), 0.500000000003)

        new_columns = _new_columns(
            [[1.0e-12]], [0.0], [-3.0], [2.0], [7]
        )
        projected = _owner._project_new_column_tiny(
            new_columns, _vector([-0.5]), _vector([0.5])
        )
        self.assertEqual(projected.data.size, 0)
        self.assertEqual(float(projected.row_lower[0]), -0.5000000000020001)
        self.assertEqual(float(projected.row_upper[0]), 0.500000000003)

        with _owner.SafeHighsOwner(
            deadline_monotonic=time.monotonic() + 10.0
        ) as owner:
            with self.assertRaises(_owner.HighsOwnerUnknown):
                owner.solve_base(
                    cost=_vector([-1.0, -1.0]),
                    column_lower=_vector([-1.0, -2.0]),
                    column_upper=column_upper,
                    rows=frozen,
                )
        self.assertEqual(owner.state, "CLOSED")

    def test_canonical_csr_and_sealed_mapping_reject_mutation(self) -> None:
        original_ids = _ids([9])
        frozen = _owner.FrozenRows.from_csr(
            _csr([[2.0]]),
            row_lower=_vector([-np.inf]),
            row_upper=_vector([3.0]),
            row_ids=original_ids,
            column_lower=_vector([-1.0]),
            column_upper=_vector([1.0]),
        )
        original_ids[0] = 99
        np.testing.assert_array_equal(frozen.row_ids, [9])
        self.assertFalse(frozen.row_ids.flags.writeable)

        duplicate = sp.csr_matrix(
            (
                np.asarray([1.0, 2.0], dtype=np.float64),
                np.asarray([0, 0], dtype=np.int32),
                np.asarray([0, 2], dtype=np.int32),
            ),
            shape=(1, 1),
        )
        with self.assertRaises(_owner.HighsOwnerUnknown):
            _owner.FrozenRows.from_csr(
                duplicate,
                row_lower=_vector([-np.inf]),
                row_upper=_vector([1.0]),
                row_ids=_ids([1]),
                column_lower=_vector([-1.0]),
                column_upper=_vector([1.0]),
            )

        frozen.row_ids.setflags(write=True)
        frozen.row_ids[0] = 10
        frozen.row_ids.setflags(write=False)
        with self.assertRaises(_owner.HighsOwnerUnknown):
            frozen.assert_intact()

    def test_canonical_csc_and_post_base_aba_fail_before_mutation(self) -> None:
        malformed = {
            "duplicate": (
                [1.0, 2.0],
                [0, 0],
                [0, 2],
            ),
            "unsorted": (
                [1.0, 2.0],
                [1, 0],
                [0, 2],
            ),
            "explicit_zero": (
                [0.0],
                [0],
                [0, 1],
            ),
            "nonfinite": (
                [np.nan],
                [0],
                [0, 1],
            ),
            "oversized": (
                [_owner._LARGE_MATRIX_VALUE],
                [0],
                [0, 1],
            ),
        }
        for name, (data, indices, indptr) in malformed.items():
            with self.subTest(name=name):
                matrix = sp.csc_matrix(
                    (
                        np.asarray(data, dtype=np.float64),
                        np.asarray(indices, dtype=np.int32),
                        np.asarray(indptr, dtype=np.int32),
                    ),
                    shape=(2, 1),
                )
                with self.assertRaises(_owner.HighsOwnerUnknown):
                    _owner.FrozenNewColumns.from_csc(
                        matrix,
                        cost=_vector([0.0]),
                        column_lower=_vector([-1.0]),
                        column_upper=_vector([1.0]),
                        existing_row_ids=_ids([1, 2]),
                    )

        backend = _RecordingHighs()
        base_rows = _rows([[1.0]], [1.0], [0.0], [1.0], row_ids=[5])
        new_columns = _new_columns([[1.0]], [0.0], [-1.0], [1.0], [5])
        appended = _rows(
            [[-1.0, 1.0]],
            [0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            row_ids=[6],
            lower=[0.0],
        )
        with mock.patch.object(_owner.highspy, "Highs", return_value=backend):
            owner = _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            )
            owner.solve_base(
                cost=_vector([-1.0]),
                column_lower=_vector([0.0]),
                column_upper=_vector([1.0]),
                rows=base_rows,
            )
            replacement = np.array(new_columns.data, dtype=np.float64, copy=True)
            replacement[0] = 2.0
            replacement.setflags(write=False)
            object.__setattr__(new_columns, "data", replacement)
            with self.assertRaises(_owner.HighsOwnerUnknown):
                owner.apply_incremental_update(
                    new_columns=new_columns,
                    existing_row_lower=_vector([-np.inf]),
                    existing_row_upper=_vector([1.0]),
                    appended_rows=appended,
                )
        self.assertEqual(backend.add_cols_calls, 1)
        self.assertEqual(backend.add_rows_calls, 1)
        self.assertEqual(backend.change_rows_bounds_calls, 0)
        self.assertEqual(backend.run_calls, 1)
        self.assertEqual(backend.clear_calls, 1)
        self.assertEqual(owner.state, "POISONED")

    def test_invalid_objectives_stop_before_backend_construction(self) -> None:
        rows = _rows([[1.0]], [1.0], [0.0], [1.0])
        for value in (1.0e-12, np.nan, _owner._INFINITE_BOUND):
            with self.subTest(value=value):
                factory = mock.Mock(side_effect=AssertionError("must not construct"))
                with mock.patch.object(_owner.highspy, "Highs", factory):
                    owner = _owner.SafeHighsOwner(
                        deadline_monotonic=time.monotonic() + 10.0
                    )
                    with self.assertRaises(_owner.HighsOwnerUnknown):
                        owner.solve_base(
                            cost=_vector([value]),
                            column_lower=_vector([0.0]),
                            column_upper=_vector([1.0]),
                            rows=rows,
                        )
                factory.assert_not_called()
                self.assertEqual(owner.state, "POISONED")

    def test_non_ok_during_partial_incremental_mutation_poison_clears(self) -> None:
        class WarningSecondAddCols(_RecordingHighs):
            def addCols(self, *args):
                self.add_cols_calls += 1
                status = self.inner.addCols(*args)
                if self.add_cols_calls == 2:
                    self.assert_native_ok = status == highspy.HighsStatus.kOk
                    return highspy.HighsStatus.kWarning
                return status

        backend = WarningSecondAddCols()
        rows = _rows([[1.0]], [1.0], [0.0], [1.0], row_ids=[5])
        new_columns = _new_columns([[1.0]], [0.0], [-1.0], [1.0], [5])
        appended = _rows(
            [[-1.0, 1.0]],
            [0.0],
            [0.0, -1.0],
            [1.0, 1.0],
            row_ids=[6],
            lower=[0.0],
        )
        with mock.patch.object(_owner.highspy, "Highs", return_value=backend):
            owner = _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            )
            owner.solve_base(
                cost=_vector([-1.0]),
                column_lower=_vector([0.0]),
                column_upper=_vector([1.0]),
                rows=rows,
            )
            with self.assertRaises(_owner.HighsOwnerUnknown):
                owner.apply_incremental_update(
                    new_columns=new_columns,
                    existing_row_lower=_vector([-np.inf]),
                    existing_row_upper=_vector([1.0]),
                    appended_rows=appended,
                )
        self.assertTrue(backend.assert_native_ok)
        self.assertEqual(backend.add_cols_calls, 2)
        self.assertEqual(backend.change_rows_bounds_calls, 0)
        self.assertEqual(backend.run_calls, 1)
        self.assertEqual(backend.clear_calls, 1)
        self.assertEqual(backend.inner.getNumRow(), 0)
        self.assertEqual(owner.state, "POISONED")

    def test_deadline_discards_late_primal_and_late_ray_existence(self) -> None:
        class Clock:
            now = 1.0

        clock = Clock()

        class LateRun(_RecordingHighs):
            def run(self):
                self.run_calls += 1
                status = self.inner.run()
                clock.now = 3.0
                return status

        optimal_backend = LateRun()
        optimal_rows = _rows([[1.0]], [1.0], [0.0], [1.0])
        with mock.patch.object(_owner.time, "monotonic", side_effect=lambda: clock.now), mock.patch.object(
            _owner.highspy, "Highs", return_value=optimal_backend
        ):
            owner = _owner.SafeHighsOwner(deadline_monotonic=2.0)
            with self.assertRaises(_owner.HighsOwnerDeadline):
                owner.solve_base(
                    cost=_vector([-1.0]),
                    column_lower=_vector([0.0]),
                    column_upper=_vector([1.0]),
                    rows=optimal_rows,
                )
        self.assertEqual(optimal_backend.get_solution_calls, 0)
        self.assertEqual(optimal_backend.clear_calls, 1)
        self.assertEqual(owner.state, "POISONED")

        clock.now = 1.0

        class LateExist(_RecordingHighs):
            def getDualRayExist(self):
                self.get_ray_exist_calls += 1
                result = self.inner.getDualRayExist()
                clock.now = 3.0
                return result

        ray_backend = LateExist()
        infeasible_rows = _rows([[1.0]], [-1.0], [0.0], [1.0], row_ids=[44])
        with mock.patch.object(_owner.time, "monotonic", side_effect=lambda: clock.now), mock.patch.object(
            _owner.highspy, "Highs", return_value=ray_backend
        ):
            owner = _owner.SafeHighsOwner(deadline_monotonic=2.0)
            with self.assertRaises(_owner.HighsOwnerDeadline):
                owner.solve_base(
                    cost=_vector([0.0]),
                    column_lower=_vector([0.0]),
                    column_upper=_vector([1.0]),
                    rows=infeasible_rows,
                )
        self.assertEqual(ray_backend.get_solution_calls, 0)
        self.assertEqual(ray_backend.get_ray_exist_calls, 1)
        self.assertEqual(ray_backend.get_ray_calls, 0)
        self.assertEqual(ray_backend.clear_calls, 1)

    def test_baseexception_and_cleanup_failure_preserve_primary_identity(self) -> None:
        class HostilePrimary(BaseException):
            def __str__(self):
                raise RuntimeError("primary must not be stringified")

            def __repr__(self):
                raise RuntimeError("primary must not be represented")

        class HostileCleanup(BaseException):
            def __str__(self):
                raise RuntimeError("cleanup must not be stringified")

            def __repr__(self):
                raise RuntimeError("cleanup must not be represented")

        primary = HostilePrimary()

        class ConfigureFault(_RecordingHighs):
            def setOptionValue(self, *_args):
                raise primary

            def clear(self):
                self.clear_calls += 1
                raise HostileCleanup()

        backend = ConfigureFault()
        rows = _rows([[1.0]], [1.0], [0.0], [1.0])
        caught = None
        with mock.patch.object(_owner.highspy, "Highs", return_value=backend):
            owner = _owner.SafeHighsOwner(
                deadline_monotonic=time.monotonic() + 10.0
            )
            try:
                owner.solve_base(
                    cost=_vector([-1.0]),
                    column_lower=_vector([0.0]),
                    column_upper=_vector([1.0]),
                    rows=rows,
                )
            except BaseException as error:
                caught = error
        self.assertIs(caught, primary)
        self.assertEqual(backend.clear_calls, 1)
        self.assertEqual(owner.state, "POISONED")
        self.assertTrue(
            any(
                "HostileCleanup" in note
                for note in getattr(caught, "__notes__", ())
            )
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
