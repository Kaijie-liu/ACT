#!/usr/bin/env python3
"""Focused fail-closed audits for operator-HZ preactivation tightening.

Run without pytest:

    python -m act.back_end.hybridz_tf.test_preactivation_hardening
"""

from __future__ import annotations

from fractions import Fraction
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import scipy.sparse as sp

import act.back_end.hybridz_tf.operator_hz as operator_hz
from act.back_end.hybridz_tf.operator_hz import (
    _AffineExpr,
    _ConstraintBlock,
    _OperatorHZBuilder,
    _PersistentPreactivationHighs,
    _PreactivationLPBase,
    _csr_sha256,
    _independent_preactivation_lagrangian_upper,
    _normalize_preactivation_targets,
)


def _base(
    A: sp.csr_matrix,
    rl: np.ndarray,
    ru: np.ndarray,
    *,
    n_eq: int,
) -> _PreactivationLPBase:
    A = sp.csr_matrix(A, dtype=np.float64)
    A.sum_duplicates()
    A.sort_indices()
    nvar = int(A.shape[1])
    return _PreactivationLPBase(
        A=A,
        rl=np.asarray(rl, dtype=np.float64),
        ru=np.asarray(ru, dtype=np.float64),
        lb=-np.ones(nvar, dtype=np.float64),
        ub=np.ones(nvar, dtype=np.float64),
        n_eq=int(n_eq),
        n_ub=int(A.shape[0] - n_eq),
        csr_sha256=_csr_sha256(A),
    )


def _builder(
    *,
    budget: int = 2,
    limit: float = 2.0,
    targets=None,
    n_cont: int = 1,
    n_bin: int = 0,
) -> _OperatorHZBuilder:
    builder = _OperatorHZBuilder(
        SimpleNamespace(),
        {},
        {},
        exact_budget=0,
        materialize_add=True,
        preactivation_lp_budget=budget,
        preactivation_lp_time_limit=limit,
        preactivation_targets=targets,
        deadline=None,
    )
    builder.n_cont = int(n_cont)
    builder.n_bin = int(n_bin)
    return builder


def _fixed_builder(**kwargs) -> _OperatorHZBuilder:
    builder = _builder(**kwargs)
    builder.eq_blocks = [
        _ConstraintBlock(
            Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
            Ab=sp.csr_matrix((1, builder.n_bin), dtype=np.float64),
            rhs=np.asarray([0.0], dtype=np.float64),
            tag="x_fixed_zero",
        )
    ]
    return builder


class _FakeHighs:
    def __init__(self, *, run_status, model_status, dual, delay=0.0):
        self.run_status = run_status
        self.model_status = model_status
        self.dual = np.asarray(dual, dtype=np.float64)
        self.delay = float(delay)

    def setOptionValue(self, *_args):
        return operator_hz._highspy.HighsStatus.kOk

    def changeColsCost(self, *_args):
        return operator_hz._highspy.HighsStatus.kOk

    def setBasis(self, *_args):
        return operator_hz._highspy.HighsStatus.kOk

    def run(self):
        if self.delay:
            time.sleep(self.delay)
        return self.run_status

    def getModelStatus(self):
        return self.model_status

    def getSolution(self):
        return SimpleNamespace(dual_valid=True, row_dual=self.dual)

    def getBasis(self):
        return SimpleNamespace(valid=True)


class CandidateBoundaryTests(unittest.TestCase):
    def _candidate(self, fake: _FakeHighs):
        builder = _fixed_builder(budget=1, limit=1.0)
        builder._start_preactivation_clock()
        base = builder._preactivation_lp_base()
        model = _PersistentPreactivationHighs(
            highs=fake,
            base=base,
            all_cols=np.asarray([0], dtype=np.int32),
            receipt={},
        )
        return builder, model

    def test_error_and_nonoptimal_duals_are_rejected(self):
        HS = operator_hz._highspy.HighsStatus
        MS = operator_hz._highspy.HighsModelStatus
        cases = (
            (_FakeHighs(
                run_status=HS.kError,
                model_status=MS.kOptimal,
                dual=[-1.0],
            ), "rejected:run_non_ok"),
            (_FakeHighs(
                run_status=HS.kOk,
                model_status=MS.kTimeLimit,
                dual=[-1.0],
            ), "rejected:model_nonoptimal"),
        )
        for fake, expected in cases:
            with self.subTest(expected=expected):
                builder, model = self._candidate(fake)
                dual, receipt = builder._preactivation_candidate_dual(
                    model, np.asarray([1.0]), time_slice=0.2
                )
                self.assertIsNone(dual)
                self.assertEqual(receipt["status"], expected)
                self.assertFalse(receipt["success"])

    def test_solver_slice_overrun_discards_even_optimal_dual(self):
        HS = operator_hz._highspy.HighsStatus
        MS = operator_hz._highspy.HighsModelStatus
        builder, model = self._candidate(_FakeHighs(
            run_status=HS.kOk,
            model_status=MS.kOptimal,
            dual=[-1.0],
            delay=0.003,
        ))
        dual, receipt = builder._preactivation_candidate_dual(
            model, np.asarray([1.0]), time_slice=1.0e-4
        )
        self.assertIsNone(dual)
        self.assertEqual(receipt["status"], "discarded:solver_overrun")
        self.assertEqual(
            builder.preactivation_lp_deadline_stage,
            "candidate_solver_overrun",
        )


class OriginalAuthorityTests(unittest.TestCase):
    def test_tiny_candidate_drop_keeps_original_fraction_and_hash(self):
        tiny = 1.0e-13
        base = _base(
            sp.csr_matrix([[tiny, 1.0]], dtype=np.float64),
            np.asarray([0.0]),
            np.asarray([0.0]),
            n_eq=1,
        )
        before_data = base.A.data.copy()
        before_hash = _csr_sha256(base.A)
        builder = _builder(budget=1, limit=1.0, n_cont=2)
        builder._start_preactivation_clock()
        model, receipt = builder._build_preactivation_candidate_model(base)
        self.assertIsNotNone(model)
        self.assertEqual(receipt["matrix"]["dropped_nnz"], 1)
        self.assertEqual(receipt["matrix"]["loaded_nnz"], 1)
        self.assertEqual(_csr_sha256(base.A), before_hash)
        np.testing.assert_array_equal(base.A.data, before_data)
        self.assertNotEqual(Fraction.from_float(float(base.A.data[0])), 0)

        expr = _AffineExpr(
            c=np.zeros(1),
            G=sp.csr_matrix([[tiny, 1.0]], dtype=np.float64),
            err=np.zeros(1),
        )
        bound, cert = _independent_preactivation_lagrangian_upper(
            expr,
            0,
            sign=1.0,
            base=base,
            row_dual=np.asarray([-1.0]),
        )
        self.assertIsNotNone(bound)
        self.assertTrue(cert["proof_authority"])
        self.assertEqual(_csr_sha256(base.A), before_hash)

    def test_semantic_error_is_added_in_both_directions(self):
        base = _base(
            sp.csr_matrix((0, 1), dtype=np.float64),
            np.zeros(0),
            np.zeros(0),
            n_eq=0,
        )
        expr = _AffineExpr(
            c=np.zeros(1),
            G=sp.csr_matrix((1, 1), dtype=np.float64),
            err=np.asarray([0.125]),
        )
        for sign in (-1.0, 1.0):
            bound, receipt = _independent_preactivation_lagrangian_upper(
                expr, 0, sign=sign, base=base, row_dual=np.zeros(0)
            )
            self.assertTrue(receipt["proof_authority"])
            self.assertEqual(receipt["semantic_error"], 0.125)
            self.assertGreaterEqual(
                Fraction.from_float(float(bound)), Fraction(1, 8)
            )

    def test_binary_columns_are_box_relaxed_and_objective_padded(self):
        base = _base(
            sp.csr_matrix((0, 2), dtype=np.float64),
            np.zeros(0),
            np.zeros(0),
            n_eq=0,
        )
        expr = _AffineExpr(
            c=np.zeros(1),
            G=sp.csr_matrix([[1.0]], dtype=np.float64),
            err=np.zeros(1),
        )
        bound, receipt = _independent_preactivation_lagrangian_upper(
            expr, 0, sign=1.0, base=base, row_dual=np.zeros(0)
        )
        self.assertTrue(receipt["proof_authority"])
        self.assertTrue(receipt["binary_relaxation"])
        self.assertGreaterEqual(float(bound), 1.0)


class SchedulingAndDeadlineTests(unittest.TestCase):
    def test_targets_global_budget_fair_slices_and_one_model_per_layer(self):
        builder = _fixed_builder(
            budget=3,
            limit=1.0,
            targets=[(10, 2), (10, 0), (11, 1)],
        )
        expr = _AffineExpr(
            c=np.zeros(3),
            G=sp.csr_matrix(np.ones((3, 1)), dtype=np.float64),
            err=np.zeros(3),
        )
        cube_l = -np.ones(3)
        cube_u = np.ones(3)
        _, _, first = builder._tighten_relu_bounds(
            10, expr, cube_l, cube_u
        )
        self.assertEqual([item["row"] for item in first["receipts"]], [2, 0])
        self.assertEqual(first["rows_attempted"], 2)
        self.assertEqual(first["persistent_model_constructions"], 1)
        self.assertEqual(builder.preactivation_lp_model_builds, 1)
        slices = [
            direction["fair_slice"]
            for row in first["receipts"]
            for direction in row["directions"].values()
        ]
        self.assertEqual(len(slices), 4)
        self.assertTrue(all(value > 0.0 for value in slices))
        self.assertLess(slices[0], slices[-1])
        self.assertTrue(
            first["receipts"][0]["directions"]["negated_lower"][
                "candidate"
            ]["basis_reused"]
        )

        _, _, repeated = builder._tighten_relu_bounds(
            10, expr, cube_l, cube_u
        )
        self.assertEqual(repeated["status"], "layer_already_attempted")
        self.assertEqual(builder.preactivation_lp_model_builds, 1)
        _, _, second = builder._tighten_relu_bounds(
            11, expr, cube_l, cube_u
        )
        self.assertEqual(second["rows_attempted"], 1)
        self.assertEqual(builder.preactivation_lp_model_builds, 2)
        _, _, exhausted = builder._tighten_relu_bounds(
            12, expr, cube_l, cube_u
        )
        self.assertEqual(exhausted["status"], "budget_exhausted")

    def test_snapshot_and_certificate_are_inside_absolute_deadline(self):
        builder = _fixed_builder(budget=1, limit=5.0e-4)
        expr = _AffineExpr(
            c=np.zeros(1),
            G=sp.csr_matrix([[1.0]], dtype=np.float64),
            err=np.zeros(1),
        )
        original_stack = operator_hz._stack_padded

        def slow_stack(*args, **kwargs):
            time.sleep(0.001)
            return original_stack(*args, **kwargs)

        with patch.object(operator_hz, "_stack_padded", side_effect=slow_stack):
            lower, upper, receipt = builder._tighten_relu_bounds(
                5, expr, np.asarray([-1.0]), np.asarray([1.0])
            )
        np.testing.assert_array_equal(lower, [-1.0])
        np.testing.assert_array_equal(upper, [1.0])
        self.assertEqual(receipt["status"], "deadline")
        self.assertEqual(receipt["deadline_stage"], "snapshot_after")
        self.assertGreater(receipt["snapshot_seconds"], 0.0)

    def test_certificate_overrun_is_measured_and_discarded(self):
        builder = _fixed_builder(budget=1, limit=0.03)
        expr = _AffineExpr(
            c=np.zeros(1),
            G=sp.csr_matrix([[1.0]], dtype=np.float64),
            err=np.zeros(1),
        )

        def slow_certificate(*_args, **_kwargs):
            time.sleep(0.04)
            return -0.5, {
                "proof_authority": True,
                "status": "verified_upper",
            }

        with patch.object(
            operator_hz,
            "_independent_preactivation_lagrangian_upper",
            side_effect=slow_certificate,
        ):
            lower, upper, receipt = builder._tighten_relu_bounds(
                6, expr, np.asarray([-1.0]), np.asarray([1.0])
            )
        np.testing.assert_array_equal(lower, [-1.0])
        np.testing.assert_array_equal(upper, [1.0])
        self.assertEqual(receipt["status"], "deadline")
        self.assertEqual(receipt["deadline_stage"], "certificate_after")
        self.assertGreaterEqual(receipt["certificate_seconds"], 0.04)
        self.assertFalse(receipt["proof_authority"])

    def test_crossing_certificates_restore_cube(self):
        builder = _fixed_builder(budget=1, limit=1.0)
        expr = _AffineExpr(
            c=np.zeros(1),
            G=sp.csr_matrix([[1.0]], dtype=np.float64),
            err=np.zeros(1),
        )
        fake_receipt = {"proof_authority": True, "status": "verified_upper"}
        with patch.object(
            operator_hz,
            "_independent_preactivation_lagrangian_upper",
            side_effect=[(-0.5, dict(fake_receipt)), (-0.5, dict(fake_receipt))],
        ):
            lower, upper, receipt = builder._tighten_relu_bounds(
                7, expr, np.asarray([-1.0]), np.asarray([1.0])
            )
        np.testing.assert_array_equal(lower, [-1.0])
        np.testing.assert_array_equal(upper, [1.0])
        self.assertEqual(
            receipt["receipts"][0]["conflict"],
            "certified_bounds_crossed_cube_restored",
        )
        self.assertFalse(receipt["proof_authority"])

    def test_target_normalization_is_explicit_and_deterministic(self):
        self.assertEqual(
            _normalize_preactivation_targets(
                {4: [2, 1, 2], 9: np.asarray([0])}
            ),
            {4: (2, 1), 9: (0,)},
        )
        self.assertEqual(
            _normalize_preactivation_targets([(9, 3), (4, 1), (9, 3)]),
            {9: (3,), 4: (1,)},
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
