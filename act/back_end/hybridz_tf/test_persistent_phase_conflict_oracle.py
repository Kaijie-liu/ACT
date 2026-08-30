#!/usr/bin/env python3
"""Exact and adversarial toys for the persistent PC-PCC pair oracle."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import gc
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from scipy.optimize import linprog
import scipy.sparse as sp

import act.back_end.hybridz_tf.persistent_phase_conflict_oracle as pco
from act.back_end.hybridz_tf.adaptive_phase_forest import (
    ordered_property_digest,
    sparse_hz_semantic_digest,
)
from act.back_end.hybridz_tf.persistent_phase_conflict_oracle import (
    PersistentConflictOracleError,
    NativeSplitRowObjectiveDualProposal,
    exact_certificate_from_highs_dual_ray_candidate,
    make_persistent_pc_pcc_invocation_spec,
    propose_native_split_row_objective_duals,
    revoke_persistent_pc_pcc_result,
    run_persistent_conflict_oracle_candidate,
    run_persistent_pc_pcc_candidate,
    verify_exact_dual_ray_conflict_certificate,
    verify_persistent_conflict_oracle_result,
    verify_persistent_pc_pcc_result,
    verify_persistent_pc_pcc_structural_result,
)
from act.back_end.hybridz_tf.property_phase_conflict_clique import (
    PhaseLiteral,
    _derive_property_literals,
)
from act.back_end.hybridz_tf.test_property_phase_conflict_clique import (
    _clone_hz,
    _complete_c49,
    _lp_upper,
    _rivals,
)
from act.back_end.solver.solver_hz import (
    SparseHZono,
    _hz_independent_split_block_lp_lagrangian_upper,
    hz_fresh_col_ids,
)


def _fresh_ids(count: int) -> np.ndarray:
    return (
        hz_fresh_col_ids(count, device="cpu")
        .detach()
        .cpu()
        .numpy()
        .astype(np.int64, copy=False)
    )


def _three_source_cancellation_hz() -> SparseHZono:
    """All three rows are needed after the two positive phases are fixed."""

    return SparseHZono(
        c=np.asarray([1.0], dtype=np.float64),
        Gc=sp.csr_matrix(np.zeros((1, 2), dtype=np.float64)),
        Gb=sp.csr_matrix(
            np.asarray([[0.5, 0.5]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, 2), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]],
                dtype=np.float64,
            )
        ),
        Aub=sp.csr_matrix(
            np.asarray(
                [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]],
                dtype=np.float64,
            )
        ),
        ub=np.zeros(3, dtype=np.float64),
        col_ids=_fresh_ids(2),
        bcol_ids=_fresh_ids(2),
    )


def _equality_cancellation_hz() -> SparseHZono:
    """x+s1+s2=0 is impossible at positive/positive within x in [-1,1]."""

    return SparseHZono(
        c=np.asarray([1.0], dtype=np.float64),
        Gc=sp.csr_matrix(np.zeros((1, 1), dtype=np.float64)),
        Gb=sp.csr_matrix(
            np.asarray([[0.5, 0.5]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix(
            np.asarray([[1.0]], dtype=np.float64)
        ),
        Ab=sp.csr_matrix(
            np.asarray([[1.0, 1.0]], dtype=np.float64)
        ),
        b=np.zeros(1, dtype=np.float64),
        Auc=sp.csr_matrix((0, 1), dtype=np.float64),
        Aub=sp.csr_matrix((0, 2), dtype=np.float64),
        ub=np.zeros(0, dtype=np.float64),
        col_ids=_fresh_ids(1),
        bcol_ids=_fresh_ids(2),
    )


def _redundant_dust_hz() -> SparseHZono:
    """Rows zero and two conflict; row one is numerically irrelevant."""

    return SparseHZono(
        c=np.asarray([1.0], dtype=np.float64),
        Gc=sp.csr_matrix(np.zeros((1, 1), dtype=np.float64)),
        Gb=sp.csr_matrix(
            np.asarray([[0.5, 0.5]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, 1), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            np.asarray([[1.0], [1.0], [-1.0]], dtype=np.float64)
        ),
        Aub=sp.csr_matrix(
            np.asarray(
                [[1.0, 0.0], [0.0, 0.0], [0.0, 1.0]],
                dtype=np.float64,
            )
        ),
        ub=np.asarray([0.0, 2.0, 0.0], dtype=np.float64),
        col_ids=_fresh_ids(1),
        bcol_ids=_fresh_ids(2),
    )


def _many_source_chain_hz(source_count: int = 129) -> SparseHZono:
    """Every row in one long continuous cancellation chain is necessary."""

    n_continuous = source_count - 1
    continuous = np.zeros(
        (source_count, n_continuous), dtype=np.float64
    )
    binary = np.zeros((source_count, 2), dtype=np.float64)
    continuous[0, 0] = 1.0
    binary[0, 0] = 1.0
    for row in range(1, source_count - 1):
        continuous[row, row - 1] = -1.0
        continuous[row, row] = 1.0
    continuous[-1, -1] = -1.0
    binary[-1, 1] = 1.0
    return SparseHZono(
        c=np.asarray([1.0], dtype=np.float64),
        Gc=sp.csr_matrix(
            np.zeros((1, n_continuous), dtype=np.float64)
        ),
        Gb=sp.csr_matrix(
            np.asarray([[0.5, 0.5]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix(
            (0, n_continuous), dtype=np.float64
        ),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(continuous),
        Aub=sp.csr_matrix(binary),
        ub=np.zeros(source_count, dtype=np.float64),
        col_ids=_fresh_ids(n_continuous),
        bcol_ids=_fresh_ids(2),
    )


def _wide_single_source_hz(nonzeros: int = 4096) -> SparseHZono:
    """One deliberately wide exact row for deadline/cap stop-loss tests."""

    return SparseHZono(
        c=np.asarray([1.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, nonzeros), dtype=np.float64),
        Gb=sp.csr_matrix(
            np.asarray([[0.5, 0.5]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix((0, nonzeros), dtype=np.float64),
        Ab=sp.csr_matrix((0, 2), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            (
                np.ones(nonzeros, dtype=np.float64),
                np.arange(nonzeros, dtype=np.int32),
                np.asarray([0, nonzeros], dtype=np.int32),
            ),
            shape=(1, nonzeros),
        ),
        Aub=sp.csr_matrix(
            np.asarray([[1.0, 1.0]], dtype=np.float64)
        ),
        ub=np.zeros(1, dtype=np.float64),
        col_ids=_fresh_ids(nonzeros),
        bcol_ids=_fresh_ids(2),
    )


def _reseal(certificate):
    placeholder = replace(certificate, certificate_sha256="")
    return replace(
        placeholder,
        certificate_sha256=pco._certificate_digest(placeholder),
    )


def _bound_pair(
    parent: SparseHZono,
    phases: tuple[int, int],
) -> tuple[PhaseLiteral, PhaseLiteral]:
    parent_digest = sparse_hz_semantic_digest(parent)
    property_digest = "d" * 64
    return tuple(
        PhaseLiteral(
            stable_bcol_id=int(parent.bcol_ids[index]),
            phase=phase,
            binding_digest=pco._literal_binding_digest(
                parent_digest=parent_digest,
                property_digest=property_digest,
                stable_bcol_id=int(parent.bcol_ids[index]),
                phase=phase,
            ),
        )
        for index, phase in enumerate(phases)
    )


def _full_invocation(
    parent: SparseHZono,
    rivals,
    *,
    gate_id: str,
    seconds: float = 10.0,
    **caps,
):
    return make_persistent_pc_pcc_invocation_spec(
        parent,
        rivals,
        deadline=time.monotonic() + seconds,
        gate_id=gate_id,
        **caps,
    )


def _mixed_candidate_load_hz() -> SparseHZono:
    above_dust = np.nextafter(1.0e-12, np.inf)
    return SparseHZono(
        c=np.asarray([0.0], dtype=np.float64),
        Gc=sp.csr_matrix((1, 2), dtype=np.float64),
        Gb=sp.csr_matrix((1, 2), dtype=np.float64),
        Ac=sp.csr_matrix(
            np.asarray(
                [[-5.0, 0.0], [1.0e-13, 6.0]],
                dtype=np.float64,
            )
        ),
        Ab=sp.csr_matrix(
            np.asarray(
                [[7.0, 8.0], [9.0, 0.0]],
                dtype=np.float64,
            )
        ),
        b=np.asarray([10.0, 11.0], dtype=np.float64),
        Auc=sp.csr_matrix(
            np.asarray(
                [[1.0, 1.0e-12], [above_dust, -2.0]],
                dtype=np.float64,
            )
        ),
        Aub=sp.csr_matrix(
            np.asarray(
                [[3.0, -1.0e-12], [0.0, 4.0]],
                dtype=np.float64,
            )
        ),
        ub=np.asarray([12.0, 13.0], dtype=np.float64),
        col_ids=_fresh_ids(2),
        bcol_ids=_fresh_ids(2),
    )


def _legacy_candidate_matrix(hz: SparseHZono) -> sp.csr_matrix:
    upper = sp.hstack([hz.Auc, hz.Aub], format="csr")
    equality = sp.hstack([hz.Ac, hz.Ab], format="csr")
    matrix = sp.vstack([upper, equality], format="csr")
    matrix.sum_duplicates()
    matrix.sort_indices()
    if matrix.nnz:
        keep = np.abs(matrix.data) > 1.0e-12
        matrix.data[~keep] = 0.0
        matrix.eliminate_zeros()
    return matrix


def _native_objective_certificate_hz() -> SparseHZono:
    """Exact optimum is 7/8 for the represented scalar output."""

    # Equality gives xc=xb.  The first upper row then gives xc<=1/4.
    # The output is 1/8 + 2*xc + xb, hence max=1/8+3/4=7/8.
    return SparseHZono(
        c=np.asarray([1.0 / 8.0], dtype=np.float64),
        Gc=sp.csr_matrix(
            np.asarray([[2.0]], dtype=np.float64)
        ),
        Gb=sp.csr_matrix(
            np.asarray([[1.0]], dtype=np.float64)
        ),
        Ac=sp.csr_matrix(
            np.asarray([[1.0]], dtype=np.float64)
        ),
        Ab=sp.csr_matrix(
            np.asarray([[-1.0]], dtype=np.float64)
        ),
        b=np.asarray([0.0], dtype=np.float64),
        Auc=sp.csr_matrix(
            np.asarray([[1.0], [-1.0]], dtype=np.float64)
        ),
        Aub=sp.csr_matrix(
            np.asarray([[1.0], [0.0]], dtype=np.float64)
        ),
        ub=np.asarray([0.5, 0.75], dtype=np.float64),
        col_ids=_fresh_ids(1),
        bcol_ids=_fresh_ids(1),
    )


def _zero_equality_binary_objective_hz() -> SparseHZono:
    return SparseHZono(
        c=np.asarray([0.0], dtype=np.float64),
        Gc=sp.csr_matrix(
            np.asarray([[1.0, -0.5]], dtype=np.float64)
        ),
        Gb=sp.csr_matrix((1, 0), dtype=np.float64),
        Ac=sp.csr_matrix((0, 2), dtype=np.float64),
        Ab=sp.csr_matrix((0, 0), dtype=np.float64),
        b=np.zeros(0, dtype=np.float64),
        Auc=sp.csr_matrix(
            np.asarray([[1.0, 0.0], [0.0, 1.0]])
        ),
        Aub=sp.csr_matrix((2, 0), dtype=np.float64),
        ub=np.asarray([0.75, 0.5], dtype=np.float64),
        col_ids=_fresh_ids(2),
        bcol_ids=_fresh_ids(0),
    )


class _FakeHighsStatus:
    kOk = "ok"
    kError = "error"


class _FakeHighsModelStatus:
    kOptimal = "optimal"
    kInfeasible = "infeasible"
    kUnknown = "unknown"


class _FakeHighs:
    instances = []
    fail_add_rows_call = None
    fail_change_coeff_call = None
    fail_change_cost = False
    fail_clear = False
    run_status = _FakeHighsStatus.kOk
    model_status = _FakeHighsModelStatus.kOptimal
    dual_valid = True
    row_dual = None
    objective_value = -1.0
    expire_after_run = False
    expire_clock = None

    def __init__(self):
        type(self).instances.append(self)
        self.options = {}
        self.column_count = 0
        self.column_lower = np.zeros(0, dtype=np.float64)
        self.column_upper = np.zeros(0, dtype=np.float64)
        self.rows = []
        self.add_rows_calls = 0
        self.change_coeff_calls = 0
        self.cleared = False
        self.run_time = 0.0
        self.column_cost = np.zeros(0, dtype=np.float64)

    @classmethod
    def reset(cls):
        cls.instances = []
        cls.fail_add_rows_call = None
        cls.fail_change_coeff_call = None
        cls.fail_change_cost = False
        cls.fail_clear = False
        cls.run_status = _FakeHighsStatus.kOk
        cls.model_status = _FakeHighsModelStatus.kOptimal
        cls.dual_valid = True
        cls.row_dual = None
        cls.objective_value = -1.0
        cls.expire_after_run = False
        cls.expire_clock = None

    def setOptionValue(self, name, value):
        self.options[name] = value
        return _FakeHighsStatus.kOk

    def changeColsCost(self, count, columns, cost):
        if type(self).fail_change_cost:
            return _FakeHighsStatus.kError
        count = int(count)
        columns = np.asarray(columns)
        cost = np.asarray(cost)
        if (
            count != self.column_count
            or columns.dtype != np.int32
            or columns.shape != (count,)
            or cost.dtype != np.float64
            or cost.shape != (count,)
            or not np.array_equal(
                columns, np.arange(count, dtype=np.int32)
            )
        ):
            return _FakeHighsStatus.kError
        self.column_cost = np.array(cost, copy=True)
        return _FakeHighsStatus.kOk

    def run(self):
        if (
            type(self).expire_after_run
            and type(self).expire_clock is not None
        ):
            type(self).expire_clock.expired = True
        return type(self).run_status

    def getModelStatus(self):
        return type(self).model_status

    def getSolution(self):
        row_dual = type(self).row_dual
        if row_dual is None:
            row_dual = -np.ones(len(self.rows), dtype=np.float64)
        return SimpleNamespace(
            dual_valid=type(self).dual_valid,
            row_dual=np.array(row_dual, copy=True),
        )

    def getInfo(self):
        return SimpleNamespace(
            objective_function_value=type(self).objective_value
        )

    def addCols(
        self,
        count,
        objective,
        lower,
        upper,
        nonzeros,
        starts,
        indices,
        data,
    ):
        del objective, starts, indices, data
        if int(nonzeros) != 0:
            return _FakeHighsStatus.kError
        self.column_count = int(count)
        self.column_lower = np.array(lower, copy=True)
        self.column_upper = np.array(upper, copy=True)
        return _FakeHighsStatus.kOk

    def addRows(
        self,
        count,
        lower,
        upper,
        nonzeros,
        starts,
        indices,
        data,
    ):
        call = self.add_rows_calls
        self.add_rows_calls += 1
        if call == type(self).fail_add_rows_call:
            return _FakeHighsStatus.kError
        count = int(count)
        nonzeros = int(nonzeros)
        starts = np.asarray(starts)
        indices = np.asarray(indices)
        data = np.asarray(data)
        if (
            starts.dtype != np.int32
            or indices.dtype != np.int32
            or data.dtype != np.float64
            or starts.size != count + 1
            or int(starts[0]) != 0
            or int(starts[-1]) != nonzeros
            or indices.size != nonzeros
            or data.size != nonzeros
        ):
            return _FakeHighsStatus.kError
        for row in range(count):
            start = int(starts[row])
            stop = int(starts[row + 1])
            coefficients = {
                int(indices[position]): float(data[position])
                for position in range(start, stop)
            }
            self.rows.append(
                {
                    "lower": float(lower[row]),
                    "upper": float(upper[row]),
                    "coefficients": coefficients,
                }
            )
        return _FakeHighsStatus.kOk

    def changeCoeff(self, row, column, value):
        call = self.change_coeff_calls
        self.change_coeff_calls += 1
        if call == type(self).fail_change_coeff_call:
            return _FakeHighsStatus.kError
        if (
            int(row) < 0
            or int(row) >= len(self.rows)
            or int(column) < 0
            or int(column) >= self.column_count
        ):
            return _FakeHighsStatus.kError
        if float(value) == 0.0:
            self.rows[int(row)]["coefficients"].pop(
                int(column), None
            )
        else:
            self.rows[int(row)]["coefficients"][int(column)] = (
                float(value)
            )
        if type(self).expire_clock is not None:
            type(self).expire_clock.expired = True
        return _FakeHighsStatus.kOk

    def getNumCol(self):
        return self.column_count

    def getNumRow(self):
        return len(self.rows)

    def getNumNz(self):
        return sum(len(row["coefficients"]) for row in self.rows)

    def getRunTime(self):
        return self.run_time

    def clear(self):
        self.cleared = True
        self.rows.clear()
        return (
            _FakeHighsStatus.kError
            if type(self).fail_clear
            else _FakeHighsStatus.kOk
        )


class _FakeHighspy:
    Highs = _FakeHighs
    HighsStatus = _FakeHighsStatus
    HighsModelStatus = _FakeHighsModelStatus
    kHighsInf = 1.0e30
    HIGHS_VERSION_MAJOR = 1
    HIGHS_VERSION_MINOR = 14
    HIGHS_VERSION_PATCH = 0


class _ExpiryClock:
    def __init__(self):
        self.expired = False

    def monotonic(self):
        return 2.0 if self.expired else 0.0


def _fake_loaded_matrix(fake: _FakeHighs) -> np.ndarray:
    matrix = np.zeros(
        (len(fake.rows), fake.column_count), dtype=np.float64
    )
    for row_index, row in enumerate(fake.rows):
        for column, value in row["coefficients"].items():
            matrix[row_index, column] = value
    return matrix


class PersistentHighsLowPeakLoadTests(unittest.TestCase):
    def setUp(self) -> None:
        _FakeHighs.reset()

    def _assert_legacy_model(
        self, parent: SparseHZono, fake: _FakeHighs
    ) -> None:
        legacy = _legacy_candidate_matrix(parent)
        np.testing.assert_array_equal(
            _fake_loaded_matrix(fake), legacy.toarray()
        )
        expected_lower = np.concatenate(
            [
                np.full(parent.n_ub, -_FakeHighspy.kHighsInf),
                np.asarray(parent.b, dtype=np.float64),
            ]
        )
        expected_upper = np.concatenate(
            [
                np.asarray(parent.ub, dtype=np.float64),
                np.asarray(parent.b, dtype=np.float64),
            ]
        )
        np.testing.assert_array_equal(
            np.asarray([row["lower"] for row in fake.rows]),
            expected_lower,
        )
        np.testing.assert_array_equal(
            np.asarray([row["upper"] for row in fake.rows]),
            expected_upper,
        )
        self.assertEqual(fake.getNumNz(), legacy.nnz)

    def test_split_loader_matches_legacy_rows_bounds_dust_and_equalities(
        self,
    ) -> None:
        parent = _mixed_candidate_load_hz()
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            mock.patch.object(
                pco, "_highs_process_threads", return_value=3
            ),
        ):
            with pco._PersistentHighsPairLP(
                parent,
                deadline=time.monotonic() + 10.0,
                solve_base_relaxation=False,
                candidate_presolve=True,
            ) as oracle:
                fake = _FakeHighs.instances[-1]
                self._assert_legacy_model(parent, fake)
                self.assertEqual(fake.add_rows_calls, 2)
                self.assertEqual(fake.change_coeff_calls, 5)
                self.assertEqual(fake.options["presolve"], "on")
                self.assertEqual(fake.options["threads"], 3)
                self.assertFalse(hasattr(oracle, "_matrix"))
                self.assertEqual(
                    oracle.telemetry["candidate_load_mode"],
                    "split_continuous_rows_binary_change_coeff_v1",
                )
                self.assertEqual(
                    oracle.telemetry["candidate_rows"],
                    parent.n_ub + parent.n_eq,
                )
                self.assertEqual(
                    oracle.telemetry["candidate_nonzeros"],
                    _legacy_candidate_matrix(parent).nnz,
                )
            self.assertTrue(fake.cleared)
            self.assertIsNone(oracle._highs)
            with self.assertRaisesRegex(
                PersistentConflictOracleError,
                "highs_model_closed",
            ):
                oracle.probe(
                    _bound_pair(parent, (1, 1)),
                    deadline=time.monotonic() + 1.0,
                )

    def test_split_loader_handles_zero_equalities_and_binary_injection(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        with mock.patch.object(pco, "_highspy", _FakeHighspy):
            oracle = pco._PersistentHighsPairLP(
                parent,
                deadline=time.monotonic() + 10.0,
                solve_base_relaxation=False,
            )
            fake = _FakeHighs.instances[-1]
            self._assert_legacy_model(parent, fake)
            self.assertEqual(fake.add_rows_calls, 1)
            self.assertEqual(fake.change_coeff_calls, 2)
            oracle.close()
            self.assertTrue(fake.cleared)

    def test_noncanonical_sum_then_dust_is_legacy_bit_exact(self) -> None:
        parent = _mixed_candidate_load_hz()
        parent.Auc = sp.csr_matrix(
            (
                np.asarray(
                    [0.6e-12, 0.6e-12, -2.0],
                    dtype=np.float64,
                ),
                np.asarray([0, 0, 1], dtype=np.int32),
                np.asarray([0, 2, 3], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        with mock.patch.object(pco, "_highspy", _FakeHighspy):
            oracle = pco._PersistentHighsPairLP(
                parent,
                deadline=time.monotonic() + 10.0,
                solve_base_relaxation=False,
            )
            self._assert_legacy_model(
                parent, _FakeHighs.instances[-1]
            )
            oracle.close()

    def test_binary_cap_uses_one_equivalent_merged_fallback(self) -> None:
        parent = _mixed_candidate_load_hz()
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            mock.patch.object(
                pco, "_MAX_BINARY_CHANGE_COEFFICIENTS", 0
            ),
        ):
            oracle = pco._PersistentHighsPairLP(
                parent,
                deadline=time.monotonic() + 10.0,
                solve_base_relaxation=False,
            )
            fake = _FakeHighs.instances[-1]
            self._assert_legacy_model(parent, fake)
            self.assertEqual(fake.add_rows_calls, 1)
            self.assertEqual(fake.change_coeff_calls, 0)
            self.assertEqual(
                oracle.telemetry["candidate_load_mode"],
                "single_merged_csr_binary_cap_fallback_v1",
            )
            oracle.close()

    def test_change_coefficient_error_clears_and_fails_closed(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        _FakeHighs.fail_change_coeff_call = 0
        with mock.patch.object(pco, "_highspy", _FakeHighspy):
            with self.assertRaisesRegex(
                PersistentConflictOracleError,
                "highs_change_Aub_coefficient_failed",
            ):
                pco._PersistentHighsPairLP(
                    parent,
                    deadline=time.monotonic() + 10.0,
                    solve_base_relaxation=False,
                )
        self.assertEqual(len(_FakeHighs.instances), 1)
        self.assertTrue(_FakeHighs.instances[0].cleared)

    def test_binary_injection_timeout_clears_and_fails_closed(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        clock = _ExpiryClock()
        _FakeHighs.expire_clock = clock
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            mock.patch.object(
                pco.time, "monotonic", side_effect=clock.monotonic
            ),
        ):
            with self.assertRaisesRegex(
                PersistentConflictOracleError,
                "deadline_expired_during_inject_Aub",
            ):
                pco._PersistentHighsPairLP(
                    parent,
                    deadline=1.0,
                    solve_base_relaxation=False,
                )
        self.assertEqual(len(_FakeHighs.instances), 1)
        self.assertTrue(_FakeHighs.instances[0].cleared)


class NativeSplitRowObjectiveDualProposalTests(unittest.TestCase):
    _RECEIPT_KEYS = {
        "schema",
        "status",
        "candidate_only",
        "proof_authority",
        "verdict_authority",
        "backend",
        "highs_version",
        "presolve",
        "row_order",
        "candidate_load_mode",
        "binary_change_coefficient_cap",
        "candidate_rows",
        "candidate_columns",
        "candidate_nonzeros",
        "n_continuous",
        "n_binary",
        "n_upper",
        "n_equality",
        "objective_convention",
        "maximization_factor_objective_size",
        "maximization_factor_objective_sha256",
        "solver_cost_sha256",
        "upper_row_dual_size",
        "equality_row_dual_size",
        "upper_row_dual_sha256",
        "equality_row_dual_sha256",
        "solver_minimization_objective_hex",
        "pair_solve_calls",
        "objective_solve_calls",
        "native_model_closed_before_return",
        "uses_sparse_hstack",
        "uses_sparse_vstack",
        "used_merged_sparse_frame",
        "receipt_sha256",
    }

    def setUp(self) -> None:
        _FakeHighs.reset()

    def _fake_proposal(
        self,
        parent: SparseHZono,
        objective: np.ndarray,
    ) -> NativeSplitRowObjectiveDualProposal:
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            mock.patch.object(
                pco.sp,
                "hstack",
                side_effect=AssertionError("hstack forbidden"),
            ),
            mock.patch.object(
                pco.sp,
                "vstack",
                side_effect=AssertionError("vstack forbidden"),
            ),
        ):
            return propose_native_split_row_objective_duals(
                parent,
                objective,
                deadline=time.monotonic() + 10.0,
            )

    def test_fake_success_preserves_sign_row_order_cost_receipt_and_close(
        self,
    ) -> None:
        parent = _mixed_candidate_load_hz()
        objective = np.asarray(
            [1.25, -2.5, 3.75, -4.0], dtype=np.float64
        )
        raw_row_dual = np.asarray(
            [-1.0, -2.0, 3.0, 4.0], dtype=np.float64
        )
        _FakeHighs.row_dual = raw_row_dual
        _FakeHighs.objective_value = -9.25
        proposal = self._fake_proposal(parent, objective)
        fake = _FakeHighs.instances[-1]

        self.assertIsInstance(
            proposal, NativeSplitRowObjectiveDualProposal
        )
        np.testing.assert_array_equal(
            fake.column_cost, -objective
        )
        np.testing.assert_array_equal(
            proposal.upper_row_dual, raw_row_dual[:2]
        )
        np.testing.assert_array_equal(
            proposal.equality_row_dual, raw_row_dual[2:]
        )
        self.assertFalse(proposal.upper_row_dual.flags.writeable)
        self.assertFalse(proposal.equality_row_dual.flags.writeable)
        self.assertIsNone(proposal.upper_row_dual.base)
        self.assertIsNone(proposal.equality_row_dual.base)
        self.assertFalse(
            np.shares_memory(
                proposal.upper_row_dual, raw_row_dual
            )
        )
        self.assertFalse(
            np.shares_memory(
                proposal.equality_row_dual, raw_row_dual
            )
        )
        with self.assertRaises(ValueError):
            proposal.upper_row_dual[0] = 0.0
        self.assertEqual(
            proposal.solver_minimization_objective, -9.25
        )
        self.assertFalse(proposal.proof_authority)
        self.assertFalse(proposal.verdict_authority)
        self.assertTrue(fake.cleared)
        self.assertEqual(fake.options["presolve"], "on")
        self.assertEqual(fake.add_rows_calls, 2)
        self.assertEqual(fake.change_coeff_calls, 5)

        receipt = proposal.receipt
        self.assertIs(type(receipt), dict)
        self.assertEqual(set(receipt), self._RECEIPT_KEYS)
        self.assertEqual(
            receipt["schema"],
            "act.hybridz.native_split_row_objective_dual_proposal.v1",
        )
        self.assertEqual(receipt["status"], "optimal_dual_candidate")
        self.assertEqual(
            receipt["backend"],
            "highspy_one_shot_simplex_presolve_split_rows_v1",
        )
        self.assertEqual(receipt["presolve"], "on")
        self.assertEqual(receipt["row_order"], "upper_then_equality")
        self.assertEqual(
            receipt["candidate_load_mode"],
            "split_continuous_rows_binary_change_coeff_v1",
        )
        self.assertEqual(receipt["pair_solve_calls"], 0)
        self.assertEqual(receipt["objective_solve_calls"], 1)
        self.assertTrue(receipt["native_model_closed_before_return"])
        self.assertFalse(receipt["uses_sparse_hstack"])
        self.assertFalse(receipt["uses_sparse_vstack"])
        self.assertFalse(receipt["used_merged_sparse_frame"])
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256")
        self.assertEqual(claimed, pco._canonical_sha256(unsigned))

    def test_zero_equality_and_binary_blocks_return_readonly_empty_dual(
        self,
    ) -> None:
        parent = _zero_equality_binary_objective_hz()
        objective = np.asarray([2.0, -1.0], dtype=np.float64)
        _FakeHighs.row_dual = np.asarray(
            [-0.5, -0.25], dtype=np.float64
        )
        proposal = self._fake_proposal(parent, objective)
        fake = _FakeHighs.instances[-1]
        self.assertEqual(proposal.upper_row_dual.shape, (2,))
        self.assertEqual(proposal.equality_row_dual.shape, (0,))
        self.assertFalse(proposal.equality_row_dual.flags.writeable)
        self.assertIsNone(proposal.equality_row_dual.base)
        self.assertEqual(fake.add_rows_calls, 1)
        self.assertEqual(fake.change_coeff_calls, 0)
        self.assertEqual(proposal.receipt["n_binary"], 0)
        self.assertEqual(proposal.receipt["n_equality"], 0)
        self.assertTrue(fake.cleared)

    def test_binary_cap_fails_before_any_merged_or_row_materialization(
        self,
    ) -> None:
        parent = _mixed_candidate_load_hz()
        objective = np.ones(
            parent.n_cont + parent.n_bin, dtype=np.float64
        )
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            mock.patch.object(
                pco, "_MAX_BINARY_CHANGE_COEFFICIENTS", 0
            ),
            mock.patch.object(
                pco,
                "_merged_candidate_arrays",
                side_effect=AssertionError("merged route forbidden"),
            ),
            self.assertRaisesRegex(
                PersistentConflictOracleError,
                "binary_nonzeros_exceed_split_load_cap",
            ),
        ):
            propose_native_split_row_objective_duals(
                parent,
                objective,
                deadline=time.monotonic() + 10.0,
            )
        fake = _FakeHighs.instances[-1]
        self.assertEqual(fake.add_rows_calls, 0)
        self.assertEqual(fake.change_coeff_calls, 0)
        self.assertTrue(fake.cleared)

    def test_fake_solver_failure_paths_close_and_never_return_candidate(
        self,
    ) -> None:
        parent = _zero_equality_binary_objective_hz()
        objective = np.ones(2, dtype=np.float64)
        cases = (
            ("fail_change_cost", True, "change_objective_cost_failed"),
            ("run_status", _FakeHighsStatus.kError, "run_objective_failed"),
            (
                "model_status",
                _FakeHighsModelStatus.kUnknown,
                "model_not_optimal",
            ),
            ("dual_valid", False, "dual_invalid"),
            (
                "row_dual",
                np.asarray([-1.0]),
                "original_row_shape_invalid",
            ),
            (
                "row_dual",
                np.asarray([-1.0, np.nan]),
                "row_dual_nonfinite",
            ),
            (
                "objective_value",
                np.inf,
                "solver_objective_nonfinite",
            ),
        )
        for attribute, value, pattern in cases:
            with self.subTest(attribute=attribute, pattern=pattern):
                _FakeHighs.reset()
                setattr(_FakeHighs, attribute, value)
                with (
                    mock.patch.object(pco, "_highspy", _FakeHighspy),
                    self.assertRaisesRegex(
                        PersistentConflictOracleError, pattern
                    ),
                ):
                    propose_native_split_row_objective_duals(
                        parent,
                        objective,
                        deadline=time.monotonic() + 10.0,
                    )
                self.assertTrue(_FakeHighs.instances[-1].cleared)

    def test_deadline_after_run_and_close_failure_both_fail_closed(
        self,
    ) -> None:
        parent = _zero_equality_binary_objective_hz()
        objective = np.ones(2, dtype=np.float64)
        clock = _ExpiryClock()
        _FakeHighs.expire_clock = clock
        _FakeHighs.expire_after_run = True
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            mock.patch.object(
                pco.time, "monotonic", side_effect=clock.monotonic
            ),
            self.assertRaisesRegex(
                PersistentConflictOracleError,
                "deadline_expired_during_objective_solver_run",
            ),
        ):
            propose_native_split_row_objective_duals(
                parent, objective, deadline=1.0
            )
        self.assertTrue(_FakeHighs.instances[-1].cleared)

        _FakeHighs.reset()
        _FakeHighs.row_dual = np.asarray([-1.0, -1.0])
        _FakeHighs.fail_clear = True
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            self.assertRaisesRegex(
                PersistentConflictOracleError,
                "highs_clear_failed",
            ),
        ):
            propose_native_split_row_objective_duals(
                parent,
                objective,
                deadline=time.monotonic() + 10.0,
            )
        self.assertTrue(_FakeHighs.instances[-1].cleared)

    def test_noncanonical_parent_fails_before_backend_construction(self):
        parent = _zero_equality_binary_objective_hz()
        parent.Auc = sp.csr_matrix(
            (
                np.asarray([0.5, 0.5, 1.0], dtype=np.float64),
                np.asarray([0, 0, 1], dtype=np.int32),
                np.asarray([0, 2, 3], dtype=np.int32),
            ),
            shape=(2, 2),
        )
        self.assertFalse(parent.Auc.has_canonical_format)
        with (
            mock.patch.object(pco, "_highspy", _FakeHighspy),
            self.assertRaisesRegex(
                PersistentConflictOracleError,
                "Auc_not_canonical_binary64_csr",
            ),
        ):
            propose_native_split_row_objective_duals(
                parent,
                np.ones(2, dtype=np.float64),
                deadline=time.monotonic() + 10.0,
            )
        self.assertEqual(_FakeHighs.instances, [])

    def test_real_tiny_producer_checker_scipy_and_fraction_agree(self):
        parent = _native_objective_certificate_hz()
        objective = np.asarray([2.0, 1.0], dtype=np.float64)
        proposal = propose_native_split_row_objective_duals(
            parent,
            objective,
            deadline=time.monotonic() + 10.0,
        )
        self.assertLess(proposal.upper_row_dual[0], 0.0)
        self.assertEqual(proposal.upper_row_dual.shape, (2,))
        self.assertEqual(proposal.equality_row_dual.shape, (1,))
        self.assertAlmostEqual(
            proposal.solver_minimization_objective, -0.75, places=9
        )

        checked_upper, checker_receipt = (
            _hz_independent_split_block_lp_lagrangian_upper(
                c=parent.c,
                Gc=parent.Gc,
                Gb=parent.Gb,
                C_row=np.asarray([1.0], dtype=np.float64),
                threshold=0.0,
                Auc=parent.Auc,
                Aub=parent.Aub,
                Ac=parent.Ac,
                Ab=parent.Ab,
                ub=parent.ub,
                b=parent.b,
                continuous_lb=-np.ones(1, dtype=np.float64),
                continuous_ub=np.ones(1, dtype=np.float64),
                binary_lb=-np.ones(1, dtype=np.float64),
                binary_ub=np.ones(1, dtype=np.float64),
                upper_row_dual=proposal.upper_row_dual,
                equality_row_dual=proposal.equality_row_dual,
                deadline=time.monotonic() + 10.0,
            )
        )
        self.assertEqual(checker_receipt["status"], "verified_upper")
        self.assertFalse(checker_receipt["uses_sparse_hstack"])
        self.assertFalse(checker_receipt["uses_sparse_vstack"])
        self.assertEqual(checker_receipt["assembled_sparse_nnz"], 0)

        scipy_result = linprog(
            -objective,
            A_ub=np.asarray([[1.0, 1.0], [-1.0, 0.0]]),
            b_ub=np.asarray([0.5, 0.75]),
            A_eq=np.asarray([[1.0, -1.0]]),
            b_eq=np.asarray([0.0]),
            bounds=[(-1.0, 1.0), (-1.0, 1.0)],
            method="highs",
        )
        self.assertTrue(scipy_result.success, scipy_result.message)
        scipy_upper = Fraction(1, 8) + Fraction(
            *float(-scipy_result.fun).as_integer_ratio()
        )
        exact_upper = Fraction(7, 8)
        self.assertEqual(scipy_upper, exact_upper)
        checked_exact = Fraction(
            *checked_upper.as_integer_ratio()
        )
        self.assertGreaterEqual(checked_exact, exact_upper)
        self.assertLess(
            float(checked_upper) - float(exact_upper), 1.0e-12
        )


class PersistentConflictOracleExactTests(unittest.TestCase):
    def test_candidate_presolve_materializes_lazy_ray_then_exactly_replays(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        oracle = pco._PersistentHighsPairLP(
            parent,
            deadline=time.monotonic() + 10.0,
            solve_base_relaxation=False,
            candidate_presolve=True,
        )

        class LazyRayProxy:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def getDualRayExist(self):
                # Reproduce HiGHS 1.14's post-presolve lazy-ray state.
                return pco._highspy.HighsStatus.kOk, False

        oracle._highs = LazyRayProxy(oracle._highs)
        pair = _bound_pair(parent, (1, 1))
        status, ray = oracle.probe(
            pair, deadline=time.monotonic() + 10.0
        )
        self.assertEqual(status, "infeasible_with_ray")
        self.assertIsNotNone(ray)
        self.assertEqual(oracle.telemetry["presolve"], "on")
        self.assertEqual(oracle.telemetry["dual_ray_calls"], 1)

        parent_digest = sparse_hz_semantic_digest(parent)
        certificate = exact_certificate_from_highs_dual_ray_candidate(
            parent,
            pair,
            ray,
            parent_digest=parent_digest,
            property_digest="d" * 64,
            source_frame_digest=pco._ordered_source_frame_digest(
                parent,
                parent_digest=parent_digest,
                deadline=time.monotonic() + 10.0,
            ),
            deadline=time.monotonic() + 10.0,
        )
        self.assertIsNotNone(certificate)
        self.assertLess(certificate.contradiction, 0)

    def test_candidate_first_skips_base_and_mixed_pairs_remain_exact(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        oracle = pco._PersistentHighsPairLP(
            parent,
            deadline=time.monotonic() + 10.0,
            solve_base_relaxation=False,
        )
        self.assertEqual(oracle.telemetry["model_builds"], 1)
        self.assertEqual(oracle.telemetry["base_solve_calls"], 0)
        conflict_pair = _bound_pair(parent, (1, 1))
        status, ray = oracle.probe(
            conflict_pair, deadline=time.monotonic() + 10.0
        )
        self.assertEqual(status, "infeasible_with_ray")
        self.assertIsNotNone(ray)
        parent_digest = sparse_hz_semantic_digest(parent)
        certificate = exact_certificate_from_highs_dual_ray_candidate(
            parent,
            conflict_pair,
            ray,
            parent_digest=parent_digest,
            property_digest="d" * 64,
            source_frame_digest=pco._ordered_source_frame_digest(
                parent,
                parent_digest=parent_digest,
                deadline=time.monotonic() + 10.0,
            ),
            deadline=time.monotonic() + 10.0,
        )
        self.assertIsNotNone(certificate)
        feasible_pair = _bound_pair(parent, (1, -1))
        feasible_status, feasible_ray = oracle.probe(
            feasible_pair, deadline=time.monotonic() + 10.0
        )
        self.assertEqual(feasible_status, "feasible_or_unknown")
        self.assertIsNone(feasible_ray)
        self.assertEqual(oracle.telemetry["base_solve_calls"], 0)
        self.assertEqual(oracle.telemetry["solve_calls"], 2)

    def test_candidate_first_parent_infeasible_is_only_a_proposal(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        parent.Auc = sp.vstack(
            [
                parent.Auc,
                sp.csr_matrix([[1.0, 0.0]]),
                sp.csr_matrix([[-1.0, 0.0]]),
            ],
            format="csr",
        )
        parent.Aub = sp.vstack(
            [parent.Aub, sp.csr_matrix((2, 2))], format="csr"
        )
        parent.ub = np.concatenate(
            [parent.ub, np.asarray([-2.0, -2.0])]
        )
        pair = _bound_pair(parent, (-1, -1))
        oracle = pco._PersistentHighsPairLP(
            parent,
            deadline=time.monotonic() + 10.0,
            solve_base_relaxation=False,
        )
        status, ray = oracle.probe(
            pair, deadline=time.monotonic() + 10.0
        )
        self.assertEqual(status, "infeasible_with_ray")
        self.assertIsNotNone(ray)
        parent_digest = sparse_hz_semantic_digest(parent)
        certificate = exact_certificate_from_highs_dual_ray_candidate(
            parent,
            pair,
            ray,
            parent_digest=parent_digest,
            property_digest="d" * 64,
            source_frame_digest=pco._ordered_source_frame_digest(
                parent,
                parent_digest=parent_digest,
                deadline=time.monotonic() + 10.0,
            ),
            deadline=time.monotonic() + 10.0,
        )
        self.assertIsNotNone(certificate)
        self.assertLess(certificate.contradiction, 0)
        self.assertFalse(certificate.proof_authority)

    def test_candidate_first_infeasible_without_ray_fails_closed(
        self,
    ) -> None:
        parent = _three_source_cancellation_hz()
        oracle = pco._PersistentHighsPairLP(
            parent,
            deadline=time.monotonic() + 10.0,
            solve_base_relaxation=False,
        )

        class NoRayProxy:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def getDualRayExist(self):
                return pco._highspy.HighsStatus.kOk, False

        oracle._highs = NoRayProxy(oracle._highs)
        status, ray = oracle.probe(
            _bound_pair(parent, (1, 1)),
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(status, "infeasible_without_ray")
        self.assertIsNone(ray)
        self.assertEqual(oracle.telemetry["dual_ray_calls"], 0)

    def test_base_outcomes_classify_deadline_and_nonoptimal(self) -> None:
        parent = _three_source_cancellation_hz()
        oracle = pco._PersistentHighsPairLP(
            parent,
            deadline=time.monotonic() + 10.0,
            solve_base_relaxation=False,
        )
        expired = time.monotonic() - 1.0
        with self.assertRaisesRegex(
            PersistentConflictOracleError,
            "deadline_expired_during_base_relaxation",
        ):
            oracle._require_base_optimal(
                pco._highspy.HighsStatus.kOk,
                deadline=expired,
            )

        class NonoptimalProxy:
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def getModelStatus(self):
                return pco._highspy.HighsModelStatus.kUnknown

        oracle._highs = NonoptimalProxy(oracle._highs)
        with self.assertRaisesRegex(
            PersistentConflictOracleError,
            "base_relaxation_not_optimal",
        ):
            oracle._require_base_optimal(
                pco._highspy.HighsStatus.kOk,
                deadline=time.monotonic() + 10.0,
            )

    def test_k7_one_model_yields_complete_exact_graph(self) -> None:
        parent = _complete_c49(7)
        rivals = _rivals()
        result = run_persistent_conflict_oracle_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(
            result.status, "complete_conflict_graph_candidate"
        )
        self.assertEqual(len(result.records), 21)
        self.assertEqual(len(result.certificates), 21)
        self.assertTrue(
            verify_persistent_conflict_oracle_result(
                parent, rivals, result
            )
        )
        telemetry = result.telemetry
        self.assertEqual(telemetry["model_builds"], 1)
        self.assertEqual(telemetry["solve_calls"], 21)
        self.assertEqual(telemetry["bound_update_calls"], 42)
        self.assertEqual(telemetry["dual_ray_calls"], 21)
        self.assertEqual(telemetry["phase_children_minted"], 0)
        self.assertTrue(
            all(
                1 <= len(certificate.source_terms) <= 5
                for certificate in result.certificates
            )
        )
        self.assertTrue(
            any(
                term.multiplier != 1
                for certificate in result.certificates
                for term in certificate.source_terms
            )
        )

    def test_three_source_ray_is_necessary_and_exact(self) -> None:
        parent = _three_source_cancellation_hz()
        rivals = _rivals()
        result = run_persistent_conflict_oracle_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(
            result.status, "complete_conflict_graph_candidate"
        )
        certificate = result.certificates[0]
        self.assertEqual(len(certificate.source_terms), 3)
        self.assertLess(certificate.contradiction, 0)
        for omitted in range(3):
            terms = tuple(
                term
                for index, term in enumerate(
                    certificate.source_terms
                )
                if index != omitted
            )
            malformed = _reseal(
                replace(certificate, source_terms=terms)
            )
            self.assertFalse(
                verify_exact_dual_ray_conflict_certificate(
                    parent,
                    malformed,
                    property_digest=ordered_property_digest(
                        rivals
                    ),
                )
            )

    def test_equality_ray_orientation_replays_exactly(self) -> None:
        parent = _equality_cancellation_hz()
        rivals = _rivals()
        result = run_persistent_conflict_oracle_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(len(result.certificates), 1)
        certificate = result.certificates[0]
        self.assertIn(
            certificate.source_terms[0].kind,
            {"equality_pos", "equality_neg"},
        )
        self.assertTrue(
            verify_exact_dual_ray_conflict_certificate(
                parent,
                certificate,
                property_digest=ordered_property_digest(rivals),
            )
        )

    def test_missing_graph_edge_never_becomes_complete(self) -> None:
        parent = _complete_c49(4, missing=((0, 1),))
        rivals = _rivals()
        result = run_persistent_conflict_oracle_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(
            result.status, "incomplete_conflict_graph_candidate"
        )
        self.assertEqual(len(result.records), 6)
        self.assertEqual(len(result.certificates), 5)
        self.assertTrue(
            verify_persistent_conflict_oracle_result(
                parent, rivals, result
            )
        )

    def test_source_cap_rejects_three_row_ray_without_edge(self) -> None:
        parent = _three_source_cancellation_hz()
        result = run_persistent_conflict_oracle_candidate(
            parent,
            _rivals(),
            deadline=time.monotonic() + 10.0,
            max_source_terms=2,
        )
        self.assertEqual(
            result.status, "incomplete_conflict_graph_candidate"
        )
        self.assertEqual(len(result.certificates), 0)
        self.assertEqual(
            result.records[0].status, "exact_replay_rejected"
        )

    def test_129_source_invocation_caps_replay_consistently(self) -> None:
        parent = _many_source_chain_hz()
        rivals = _rivals()
        oracle = run_persistent_conflict_oracle_candidate(
            parent,
            rivals,
            deadline=time.monotonic() + 10.0,
            max_source_terms=200,
        )
        self.assertEqual(
            oracle.status, "complete_conflict_graph_candidate"
        )
        self.assertEqual(
            len(oracle.certificates[0].source_terms), 129
        )
        self.assertTrue(
            verify_persistent_conflict_oracle_result(
                parent, rivals, oracle
            )
        )

        invocation = _full_invocation(
            parent,
            rivals,
            gate_id="test_129_source_caps",
            max_source_terms=200,
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            rivals,
            invocation=invocation,
        )
        self.assertTrue(
            verify_persistent_pc_pcc_structural_result(
                parent, rivals, result
            )
        )
        self.assertTrue(
            verify_persistent_pc_pcc_result(
                parent,
                rivals,
                result,
                invocation=invocation,
            )
        )

    def test_full_k7_tightness_wrapper_reaches_exact_one(self) -> None:
        parent = _complete_c49(7)
        rivals = _rivals()
        invocation = _full_invocation(
            parent, rivals, gate_id="test_full_k7"
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            rivals,
            invocation=invocation,
        )
        self.assertEqual(
            result.status, "unknown_to_safe_candidate"
        )
        self.assertAlmostEqual(_lp_upper(parent), 7.0 / 3.0)
        self.assertAlmostEqual(_lp_upper(result.hz), 1.0)
        self.assertTrue(
            verify_persistent_pc_pcc_result(
                parent,
                rivals,
                result,
                invocation=invocation,
            )
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                rivals,
                result,
                invocation=invocation,
            )
        )
        self.assertTrue(
            verify_persistent_pc_pcc_structural_result(
                parent, rivals, result
            )
        )
        self.assertFalse(result.proof_authority)

    def test_full_missing_edge_wrapper_emits_no_cut(self) -> None:
        parent = _complete_c49(4, missing=((0, 1),))
        rivals = _rivals()
        invocation = _full_invocation(
            parent, rivals, gate_id="test_full_missing"
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            rivals,
            invocation=invocation,
        )
        self.assertEqual(
            result.status, "incomplete_conflict_graph_candidate"
        )
        self.assertIsNone(result.hz)
        self.assertTrue(
            verify_persistent_pc_pcc_result(
                parent,
                rivals,
                result,
                invocation=invocation,
            )
        )
        self.assertTrue(
            verify_persistent_pc_pcc_structural_result(
                parent, rivals, result
            )
        )


class PersistentConflictOracleAdversarialTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parent = _three_source_cancellation_hz()
        self.rivals = _rivals()
        self.property_digest = ordered_property_digest(self.rivals)
        self.result = run_persistent_conflict_oracle_candidate(
            self.parent,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        self.certificate = self.result.certificates[0]

    def _verify(self, certificate) -> bool:
        return verify_exact_dual_ray_conflict_certificate(
            self.parent,
            certificate,
            property_digest=self.property_digest,
        )

    def test_reorder_duplicate_and_noncanonical_fraction_fail(self) -> None:
        terms = self.certificate.source_terms
        reordered = _reseal(
            replace(
                self.certificate,
                source_terms=(terms[1], terms[0], terms[2]),
            )
        )
        self.assertFalse(self._verify(reordered))

        duplicate = _reseal(
            replace(
                self.certificate,
                source_terms=(terms[0], terms[0], terms[2]),
            )
        )
        self.assertFalse(self._verify(duplicate))

        first = replace(
            terms[0],
            numerator=2 * terms[0].numerator,
            denominator=2 * terms[0].denominator,
        )
        noncanonical = _reseal(
            replace(
                self.certificate,
                source_terms=(first, *terms[1:]),
            )
        )
        self.assertFalse(self._verify(noncanonical))

    def test_wrong_row_digest_and_contradiction_fail(self) -> None:
        terms = self.certificate.source_terms
        wrong_term = replace(
            terms[0], source_row_sha256="0" * 64
        )
        wrong_row = _reseal(
            replace(
                self.certificate,
                source_terms=(wrong_term, *terms[1:]),
            )
        )
        self.assertFalse(self._verify(wrong_row))

        zero = _reseal(
            replace(
                self.certificate,
                contradiction_numerator=0,
                contradiction_denominator=1,
            )
        )
        self.assertFalse(self._verify(zero))

    def test_parent_row_and_literal_mutations_fail(self) -> None:
        mutated = _clone_hz(self.parent)
        mutated.Auc.data[0] = np.nextafter(
            mutated.Auc.data[0], np.inf
        )
        self.assertFalse(
            verify_exact_dual_ray_conflict_certificate(
                mutated,
                self.certificate,
                property_digest=self.property_digest,
            )
        )

        literals = self.certificate.literals
        wrong_literal = replace(
            literals[0], phase=-literals[0].phase
        )
        malformed = _reseal(
            replace(
                self.certificate,
                literals=(wrong_literal, literals[1]),
            )
        )
        self.assertFalse(self._verify(malformed))

    def test_corrupt_csr_structure_and_column_ranges_fail_closed(
        self,
    ) -> None:
        malformed_parents = []

        negative = _clone_hz(self.parent)
        _ = negative.Auc.has_canonical_format
        negative.Auc.indices[0] = -1
        malformed_parents.append(negative)

        positive = _clone_hz(self.parent)
        _ = positive.Auc.has_canonical_format
        positive.Auc.indices[0] = positive.Auc.shape[1]
        malformed_parents.append(positive)

        bad_indptr = _clone_hz(self.parent)
        bad_indptr.Auc.indptr[1] = (
            bad_indptr.Auc.indptr[2] + 1
        )
        malformed_parents.append(bad_indptr)

        bad_lengths = _clone_hz(self.parent)
        bad_lengths.Auc.data = bad_lengths.Auc.data[:-1]
        malformed_parents.append(bad_lengths)

        bad_shape = _clone_hz(self.parent)
        bad_shape.Auc._shape = (
            bad_shape.Auc.shape[0],
            bad_shape.Auc.shape[1] + 1,
        )
        malformed_parents.append(bad_shape)

        for malformed_parent in malformed_parents:
            with self.subTest(
                shape=malformed_parent.Auc.shape,
                indices=malformed_parent.Auc.indices.tolist(),
            ):
                self.assertFalse(
                    verify_exact_dual_ray_conflict_certificate(
                        malformed_parent,
                        self.certificate,
                        property_digest=self.property_digest,
                    )
                )

    def test_source_row_and_frame_deadlines_stop_early(
        self,
    ) -> None:
        wide = _wide_single_source_hz()
        with mock.patch.object(
            pco.time,
            "monotonic",
            side_effect=(0.0, 0.0, 2.0),
        ):
            with self.assertRaises(PersistentConflictOracleError):
                pco._exact_sparse_source_row(
                    wide,
                    "upper",
                    0,
                    deadline=1.0,
                    max_nonzeros=wide.Auc.nnz + wide.Aub.nnz,
                )

        parent_digest = sparse_hz_semantic_digest(self.parent)
        frame_digest = pco._ordered_source_frame_digest(
            self.parent,
            parent_digest=parent_digest,
            deadline=time.monotonic() + 10.0,
        )
        self.assertEqual(len(frame_digest), 64)
        with self.assertRaises(PersistentConflictOracleError):
            pco._ordered_source_frame_digest(
                self.parent,
                parent_digest=parent_digest,
                deadline=time.monotonic() - 1.0,
            )

    def test_bad_raw_rays_never_create_certificate(self) -> None:
        parent_digest = sparse_hz_semantic_digest(self.parent)
        literals = _derive_property_literals(
            self.parent,
            self.rivals,
            parent_digest=parent_digest,
            property_digest=self.property_digest,
        )
        source_frame = pco._ordered_source_frame_digest(
            self.parent,
            parent_digest=parent_digest,
            deadline=time.monotonic() + 10.0,
        )
        for raw_ray in (
            np.ones(self.parent.n_ub, dtype=np.float64),
            np.zeros(self.parent.n_ub, dtype=np.float64),
            np.asarray([np.nan] * self.parent.n_ub),
            np.zeros(self.parent.n_ub - 1, dtype=np.float64),
        ):
            self.assertIsNone(
                exact_certificate_from_highs_dual_ray_candidate(
                    self.parent,
                    (literals[0], literals[1]),
                    raw_ray,
                    parent_digest=parent_digest,
                    property_digest=self.property_digest,
                    source_frame_digest=source_frame,
                    deadline=time.monotonic() + 10.0,
                )
            )

    def test_dual_ray_numerical_dust_is_soundly_sparsified(
        self,
    ) -> None:
        parent = _redundant_dust_hz()
        rivals = _rivals()
        property_digest = ordered_property_digest(rivals)
        parent_digest = sparse_hz_semantic_digest(parent)
        literals = _derive_property_literals(
            parent,
            rivals,
            parent_digest=parent_digest,
            property_digest=property_digest,
        )
        frame = pco._ordered_source_frame_digest(
            parent,
            parent_digest=parent_digest,
            deadline=time.monotonic() + 10.0,
        )
        for dust in (-1.0e-300, 1.0e-300):
            certificate = (
                exact_certificate_from_highs_dual_ray_candidate(
                    parent,
                    (literals[0], literals[1]),
                    np.asarray([-1.0, dust, -1.0]),
                    parent_digest=parent_digest,
                    property_digest=property_digest,
                    source_frame_digest=frame,
                    deadline=time.monotonic() + 10.0,
                )
            )
            with self.subTest(dust=dust):
                self.assertIsNotNone(certificate)
                self.assertEqual(
                    tuple(
                        term.local_row_index
                        for term in certificate.source_terms
                    ),
                    (0, 2),
                )
                self.assertTrue(
                    verify_exact_dual_ray_conflict_certificate(
                        parent,
                        certificate,
                        property_digest=property_digest,
                    )
                )

    def test_result_reorder_fails_but_telemetry_is_diagnostic(self) -> None:
        multi_parent = _complete_c49(4)
        multi_result = run_persistent_conflict_oracle_candidate(
            multi_parent,
            self.rivals,
            deadline=time.monotonic() + 10.0,
        )
        reordered = replace(
            multi_result,
            records=tuple(reversed(multi_result.records)),
        )
        self.assertFalse(
            verify_persistent_conflict_oracle_result(
                multi_parent, self.rivals, reordered
            )
        )
        diagnostic_copy = replace(
            self.result,
            telemetry={
                **self.result.telemetry,
                "wall_seconds": 0.0,
            },
        )
        self.assertTrue(
            verify_persistent_conflict_oracle_result(
                self.parent, self.rivals, diagnostic_copy
            )
        )

    def test_expired_deadline_and_bad_caps_fail_closed(self) -> None:
        with self.assertRaises(PersistentConflictOracleError):
            run_persistent_conflict_oracle_candidate(
                self.parent,
                self.rivals,
                deadline=time.monotonic() - 1.0,
            )
        with self.assertRaises(PersistentConflictOracleError):
            run_persistent_conflict_oracle_candidate(
                self.parent,
                self.rivals,
                deadline=time.monotonic() + 10.0,
                max_source_terms=True,
            )

    def test_full_live_copy_and_mutation_are_single_use_failures(
        self,
    ) -> None:
        parent = _complete_c49(4)
        invocation = _full_invocation(
            parent, self.rivals, gate_id="test_live_copy"
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=invocation,
        )
        copied = replace(result)
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                copied,
                invocation=invocation,
            )
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                result,
                invocation=invocation,
            )
        )

        mutated_invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_live_telemetry_mutation",
        )
        mutated = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=mutated_invocation,
        )
        mutated.oracle_result.telemetry["solve_calls"] += 1
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                mutated,
                invocation=mutated_invocation,
            )
        )

        cut_invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_live_cut_mutation",
        )
        cut_mutated = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=cut_invocation,
        )
        cut_mutated.hz.Aub.data[-1] = np.nextafter(
            cut_mutated.hz.Aub.data[-1], np.inf
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                cut_mutated,
                invocation=cut_invocation,
            )
        )

    def test_wrong_parent_or_equal_content_rivals_revoke_live_result(
        self,
    ) -> None:
        parent = _complete_c49(4)
        parent_invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_wrong_parent",
        )
        wrong_parent_result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=parent_invocation,
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                _clone_hz(parent),
                self.rivals,
                wrong_parent_result,
                invocation=parent_invocation,
            )
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                wrong_parent_result,
                invocation=parent_invocation,
            )
        )

        rivals_invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_wrong_rivals",
        )
        wrong_rivals_result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=rivals_invocation,
        )
        equal_content_rivals = tuple(
            replace(rival) for rival in self.rivals
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                equal_content_rivals,
                wrong_rivals_result,
                invocation=rivals_invocation,
            )
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                wrong_rivals_result,
                invocation=rivals_invocation,
            )
        )

    def test_wrong_invocation_and_expired_capability_fail_closed(
        self,
    ) -> None:
        parent = _complete_c49(4)
        invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_expected_invocation",
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=invocation,
        )
        wrong_invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_other_gate",
            max_source_terms=64,
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                result,
                invocation=wrong_invocation,
            )
        )
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                result,
                invocation=invocation,
            )
        )

        short_invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_expired_live_capability",
            seconds=0.08,
        )
        short_result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=short_invocation,
        )
        time.sleep(0.09)
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                short_result,
                invocation=short_invocation,
            )
        )

    def test_invocation_spec_is_claimed_by_exactly_one_run(
        self,
    ) -> None:
        parent = _complete_c49(4)
        invocation = _full_invocation(
            parent,
            self.rivals,
            gate_id="test_one_run_invocation",
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=invocation,
        )
        with self.assertRaises(PersistentConflictOracleError):
            run_persistent_pc_pcc_candidate(
                parent,
                self.rivals,
                invocation=invocation,
            )
        self.assertTrue(revoke_persistent_pc_pcc_result(result))

    def test_registry_weakref_revoke_and_capacity(self) -> None:
        parent = _complete_c49(4)
        baseline = len(pco._LIVE_PERSISTENT_PC_PCC_RESULTS)
        invocation = _full_invocation(
            parent, self.rivals, gate_id="test_weakref_cleanup"
        )
        result = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=invocation,
        )
        self.assertEqual(
            len(pco._LIVE_PERSISTENT_PC_PCC_RESULTS),
            baseline + 1,
        )
        del result
        gc.collect()
        self.assertEqual(
            len(pco._LIVE_PERSISTENT_PC_PCC_RESULTS), baseline
        )

        revoke_invocation = _full_invocation(
            parent, self.rivals, gate_id="test_explicit_revoke"
        )
        revoked = run_persistent_pc_pcc_candidate(
            parent,
            self.rivals,
            invocation=revoke_invocation,
        )
        self.assertTrue(revoke_persistent_pc_pcc_result(revoked))
        self.assertFalse(revoke_persistent_pc_pcc_result(revoked))
        self.assertFalse(
            verify_persistent_pc_pcc_result(
                parent,
                self.rivals,
                revoked,
                invocation=revoke_invocation,
            )
        )

        with mock.patch.object(pco, "_MAX_LIVE_RESULTS", 1):
            first_invocation = _full_invocation(
                parent,
                self.rivals,
                gate_id="test_capacity_first",
            )
            first = run_persistent_pc_pcc_candidate(
                parent,
                self.rivals,
                invocation=first_invocation,
            )
            second_invocation = _full_invocation(
                parent,
                self.rivals,
                gate_id="test_capacity_second",
            )
            with self.assertRaises(
                PersistentConflictOracleError
            ):
                run_persistent_pc_pcc_candidate(
                    parent,
                    self.rivals,
                    invocation=second_invocation,
                )
            self.assertTrue(
                revoke_persistent_pc_pcc_result(first)
            )


if __name__ == "__main__":
    unittest.main()
