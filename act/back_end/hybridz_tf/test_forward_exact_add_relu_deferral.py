#!/usr/bin/env python3
"""Strict synthetic audits for disconnected exact ADD -> ReLU deferral."""

from __future__ import annotations

from fractions import Fraction
import random
import time
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.forward_exact_add_relu_deferral import (
    benchmark_exact_add_handoff,
    benchmark_exact_add_relu_routes,
    build_exact_add_relu_deferral_candidate,
    materialize_exact_add_frame,
)
from act.back_end.hybridz_tf.tf_mlp import (
    sparse_hz_add_same_frame,
    sparse_hz_from_bounds,
    sparse_hz_linear,
)


def _residual_add(width: int = 1) -> SparseHZono:
    bounds = Bounds(
        lb=-torch.ones((1, width), dtype=torch.float64),
        ub=torch.ones((1, width), dtype=torch.float64),
    )
    x = sparse_hz_from_bounds(bounds, drop_zero_radius=False)
    identity = sp.eye(width, format="csr", dtype=np.float64)
    residual = sparse_hz_linear(
        x,
        0.5 * identity,
        np.full(width, -0.25, dtype=np.float64),
    )
    return sparse_hz_add_same_frame(x, residual)


def _counts(hz: SparseHZono):
    return (hz.n_cont, hz.n_bin, hz.n_eq, hz.n_ub, hz.constraint_nnz)


def _fingerprint(hz: SparseHZono):
    arrays = []
    for matrix in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        arrays.append(
            None
            if matrix is None
            else (
                matrix.shape,
                matrix.data.tobytes(),
                matrix.indices.tobytes(),
                matrix.indptr.tobytes(),
            )
        )
    return (
        hz.c.tobytes(),
        hz.b.tobytes(),
        None if hz.ub is None else hz.ub.tobytes(),
        None if hz.col_ids is None else hz.col_ids.tobytes(),
        None if hz.bcol_ids is None else hz.bcol_ids.tobytes(),
        tuple(arrays),
    )


def _fraction_dot(row: sp.csr_matrix, values):
    total = Fraction(0)
    for column, coefficient in zip(row.indices, row.data):
        total += Fraction.from_float(float(coefficient)) * values[int(column)]
    return total


def _assert_feasible_point(
    case: unittest.TestCase,
    hz: SparseHZono,
    cont,
    binary,
    expected_output: Fraction,
):
    cvals = tuple(Fraction(value) for value in cont)
    bvals = tuple(Fraction(value) for value in binary)
    case.assertEqual(len(cvals), hz.n_cont)
    case.assertEqual(len(bvals), hz.n_bin)
    for row in range(hz.n_eq):
        lhs = _fraction_dot(hz.Ac.getrow(row), cvals)
        lhs += _fraction_dot(hz.Ab.getrow(row), bvals)
        case.assertEqual(lhs, Fraction.from_float(float(hz.b[row])))
    for row in range(hz.n_ub):
        lhs = _fraction_dot(hz.Auc.getrow(row), cvals)
        lhs += _fraction_dot(hz.Aub.getrow(row), bvals)
        case.assertLessEqual(lhs, Fraction.from_float(float(hz.ub[row])))
    output = Fraction.from_float(float(hz.c[0]))
    output += _fraction_dot(hz.Gc.getrow(0), cvals)
    output += _fraction_dot(hz.Gb.getrow(0), bvals)
    case.assertEqual(output, expected_output)


def _assert_exact_tangent(
    case: unittest.TestCase,
    hz: SparseHZono,
    cont_derivative,
    binary_derivative,
    expected_output_derivative: Fraction,
):
    dc = tuple(Fraction(value) for value in cont_derivative)
    db = tuple(Fraction(value) for value in binary_derivative)
    for row in range(hz.n_eq):
        lhs = _fraction_dot(hz.Ac.getrow(row), dc)
        lhs += _fraction_dot(hz.Ab.getrow(row), db)
        case.assertEqual(lhs, 0)
    derivative = _fraction_dot(hz.Gc.getrow(0), dc)
    derivative += _fraction_dot(hz.Gb.getrow(0), db)
    case.assertEqual(derivative, expected_output_derivative)


class ExactAddReLUDeferralTests(unittest.TestCase):
    def test_residual_single_relu_reduces_exact_structure(self):
        add = _residual_add()
        deferred = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("RELU",)
        )
        fallback = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("RELU", "IDENTITY")
        )
        self.assertTrue(deferred.eligible)
        self.assertTrue(deferred.used_deferral)
        self.assertIsNone(deferred.add)
        self.assertFalse(fallback.eligible)
        self.assertIsNotNone(fallback.add)
        self.assertEqual(_counts(deferred.output), (3, 1, 1, 2, 8))
        self.assertEqual(_counts(fallback.output), (4, 1, 1, 4, 12))
        self.assertLess(
            deferred.receipt["after"]["payload_bytes"],
            fallback.receipt["after"]["payload_bytes"],
        )
        self.assertGreaterEqual(deferred.receipt["wall_ns"], 0)
        self.assertGreaterEqual(fallback.receipt["wall_ns"], 0)

    def test_fraction_points_phases_and_jacobian(self):
        add = _residual_add()
        deferred = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("RELU",)
        ).output
        materialized = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("RELU", "IDENTITY")
        ).output
        # add(x) = 3*x/2 - 1/4, alpha=-7/4, beta=5/4.
        alpha = Fraction(-7, 4)
        beta = Fraction(5, 4)
        for x in (Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 6), Fraction(1)):
            pre = Fraction(3, 2) * x - Fraction(1, 4)
            expected = max(Fraction(0), pre)
            eta = x  # materialized center=-1/4 and radius=3/2.
            if pre > 0:
                z = Fraction(-1)
                xi1 = Fraction(1)
                xi2 = Fraction(1) - Fraction(2) * pre / beta
                jacobian = Fraction(3, 2)
            else:
                z = Fraction(1)
                xi1 = Fraction(2) * pre / alpha - Fraction(1)
                xi2 = Fraction(1)
                jacobian = Fraction(0)
            _assert_feasible_point(
                self, deferred, (x, xi1, xi2), (z,), expected
            )
            _assert_feasible_point(
                self, materialized, (x, eta, xi1, xi2), (z,), expected
            )
            self.assertEqual(jacobian, Fraction(3, 2) if pre > 0 else 0)
            if pre != 0:
                if pre > 0:
                    dxi1 = Fraction(0)
                    dxi2 = -Fraction(3) / beta
                else:
                    dxi1 = Fraction(3) / alpha
                    dxi2 = Fraction(0)
                _assert_exact_tangent(
                    self,
                    deferred,
                    (Fraction(1), dxi1, dxi2),
                    (Fraction(0),),
                    jacobian,
                )
                _assert_exact_tangent(
                    self,
                    materialized,
                    (Fraction(1), Fraction(1), dxi1, dxi2),
                    (Fraction(0),),
                    jacobian,
                )

    def test_seeded_dyadic_point_phase_and_jacobian_census(self):
        rng = random.Random(20260809)
        base_bounds = Bounds(
            lb=torch.tensor([[-1.0]], dtype=torch.float64),
            ub=torch.tensor([[1.0]], dtype=torch.float64),
        )
        for _case in range(24):
            numerator = rng.choice((-7, -5, -3, -1, 1, 3, 5, 7))
            denominator = rng.choice((2, 4, 8))
            slope = Fraction(numerator, denominator)
            bias = slope / rng.choice((-4, -2, 2, 4))
            x_hz = sparse_hz_from_bounds(base_bounds, drop_zero_radius=False)
            residual = sparse_hz_linear(
                x_hz,
                [[float(slope - 1)]],
                [float(bias)],
            )
            add = sparse_hz_add_same_frame(x_hz, residual)
            deferred = build_exact_add_relu_deferral_candidate(
                add, consumer_kinds=("RELU",)
            ).output
            materialized = build_exact_add_relu_deferral_candidate(
                add, consumer_kinds=("RELU", "IDENTITY")
            ).output
            alpha = bias - abs(slope)
            beta = bias + abs(slope)
            self.assertLess(alpha, 0)
            self.assertGreater(beta, 0)
            for x in (Fraction(-3, 4), Fraction(-1, 4), Fraction(1, 4), Fraction(3, 4)):
                pre = slope * x + bias
                if pre == 0:
                    continue
                expected = max(Fraction(0), pre)
                eta = x if slope > 0 else -x
                if pre > 0:
                    z, xi1 = Fraction(-1), Fraction(1)
                    xi2 = Fraction(1) - Fraction(2) * pre / beta
                    dxi1, dxi2 = Fraction(0), -Fraction(2) * slope / beta
                    jacobian = slope
                else:
                    z, xi2 = Fraction(1), Fraction(1)
                    xi1 = Fraction(2) * pre / alpha - Fraction(1)
                    dxi1, dxi2 = Fraction(2) * slope / alpha, Fraction(0)
                    jacobian = Fraction(0)
                _assert_feasible_point(
                    self, deferred, (x, xi1, xi2), (z,), expected
                )
                _assert_feasible_point(
                    self, materialized, (x, eta, xi1, xi2), (z,), expected
                )
                _assert_exact_tangent(
                    self,
                    deferred,
                    (Fraction(1), dxi1, dxi2),
                    (Fraction(0),),
                    jacobian,
                )
                _assert_exact_tangent(
                    self,
                    materialized,
                    (
                        Fraction(1),
                        Fraction(1) if slope > 0 else Fraction(-1),
                        dxi1,
                        dxi2,
                    ),
                    (Fraction(0),),
                    jacobian,
                )
    def test_non_relu_consumer_falls_back_without_relu(self):
        add = _residual_add()
        result = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("DENSE",)
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "sole_consumer_not_relu")
        self.assertIsNotNone(result.add)
        self.assertIsNone(result.relu)
        self.assertEqual(result.output.n_cont, add.n_cont + add.n_out)
        self.assertEqual(result.output.n_ub, add.n_ub + 2 * add.n_out)

    def test_multi_consumer_falls_back_and_retains_relu_branch(self):
        result = build_exact_add_relu_deferral_candidate(
            _residual_add(), consumer_kinds=("RELU", "DENSE")
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "consumer_count_not_one")
        self.assertIsNotNone(result.add)
        self.assertIsNotNone(result.relu)

    def test_active_relu_is_exact_noop_on_structure(self):
        bounds = Bounds(
            lb=torch.tensor([[1.0]], dtype=torch.float64),
            ub=torch.tensor([[2.0]], dtype=torch.float64),
        )
        x = sparse_hz_from_bounds(bounds, drop_zero_radius=False)
        add = sparse_hz_add_same_frame(x, sparse_hz_linear(x, [[0.0]], [1.0]))
        result = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("RELU",)
        )
        self.assertEqual(result.receipt["phase_counts"], (1, 0, 0))
        self.assertEqual(_counts(result.output), _counts(add))
        np.testing.assert_array_equal(result.output.c, add.c)
        self.assertEqual((result.output.Gc - add.Gc).nnz, 0)

    def test_input_and_exact_buffers_are_not_mutated(self):
        add = _residual_add(3)
        before = _fingerprint(add)
        result = build_exact_add_relu_deferral_candidate(
            add, consumer_kinds=("RELU",)
        )
        self.assertEqual(before, _fingerprint(add))
        with self.assertRaises(TypeError):
            result.receipt["used_deferral"] = False
        with self.assertRaises(TypeError):
            result.receipt["after"]["C"] = -1

    def test_strict_type_and_topology_fail_closed(self):
        add = _residual_add()
        with self.assertRaises(TypeError):
            build_exact_add_relu_deferral_candidate(
                add, consumer_kinds=["RELU"]
            )
        with self.assertRaises(TypeError):
            build_exact_add_relu_deferral_candidate(
                add, consumer_kinds=(str("RELU"),), deadline=1
            )
        class SparseSubclass(SparseHZono):
            pass
        sub = SparseSubclass(*add.solver_tuple(), col_ids=add.col_ids, bcol_ids=add.bcol_ids)
        with self.assertRaises(TypeError):
            build_exact_add_relu_deferral_candidate(
                sub, consumer_kinds=("RELU",)
            )

    def test_expired_deadline_forces_exact_fallback(self):
        result = build_exact_add_relu_deferral_candidate(
            _residual_add(),
            consumer_kinds=("RELU",),
            deadline=float(time.monotonic() - 1.0),
        )
        self.assertFalse(result.eligible)
        self.assertEqual(result.reason, "deadline_expired")
        self.assertIsNotNone(result.add)
        self.assertIsNotNone(result.relu)

    def test_nonfinite_fast_bounds_fail_closed(self):
        add = _residual_add()
        add.c[0] = np.finfo(np.float64).max
        add.Gc.data[0] = np.finfo(np.float64).max
        add.Gc.data[1:] = np.finfo(np.float64).max
        with np.errstate(over="ignore"):
            with self.assertRaises(ValueError):
                build_exact_add_relu_deferral_candidate(
                    add, consumer_kinds=("RELU",)
                )

    def test_uncertified_tighter_bounds_fail_closed(self):
        add = _residual_add()
        claimed = Bounds(
            lb=torch.tensor([[-0.5]], dtype=torch.float64),
            ub=torch.tensor([[0.5]], dtype=torch.float64),
        )
        with self.assertRaises(ValueError):
            build_exact_add_relu_deferral_candidate(
                add, consumer_kinds=("RELU",), pre_bounds=claimed
            )

    def test_materialization_signed_rows_are_exact(self):
        add = _residual_add()
        materialized = materialize_exact_add_frame(add)
        self.assertEqual(materialized.n_ub, add.n_ub + 2)
        np.testing.assert_array_equal(
            materialized.Auc[-1].toarray(), -materialized.Auc[-2].toarray()
        )
        np.testing.assert_array_equal(
            materialized.Aub[-1].toarray(), -materialized.Aub[-2].toarray()
        )
        self.assertEqual(materialized.ub[-1], -materialized.ub[-2])

    def test_handoff_only_performance_gate(self):
        receipt = benchmark_exact_add_handoff(_residual_add(128), repeats=7)
        self.assertEqual(receipt["measured_stage"], "post_add_pre_relu_handoff_only")
        self.assertFalse(receipt["includes_relu"])
        self.assertFalse(receipt["includes_add_arithmetic"])
        self.assertGreaterEqual(receipt["speedup"], 1.5)

    def test_completed_add_through_exact_relu_performance_gate(self):
        receipt = benchmark_exact_add_relu_routes(
            _residual_add(128), repeats=7
        )
        self.assertEqual(
            receipt["measured_stage"],
            "completed_add_through_compact_exact_relu",
        )
        self.assertTrue(receipt["includes_relu"])
        self.assertFalse(receipt["includes_add_arithmetic"])
        self.assertTrue(receipt["includes_bound_preparation"])
        self.assertTrue(receipt["stored_real_numeric_exact"])
        self.assertFalse(receipt["python_fraction_used_by_candidate"])
        self.assertEqual(receipt["materialized"]["C"], 512)
        self.assertEqual(receipt["deferred"]["C"], 384)
        self.assertGreaterEqual(receipt["speedup"], 2.0)

    def test_receipt_has_no_authority(self):
        receipt = build_exact_add_relu_deferral_candidate(
            _residual_add(), consumer_kinds=("RELU",)
        ).receipt
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["production_authority"])
        self.assertFalse(receipt["authenticity_verified"])
        self.assertTrue(receipt["forward_only"])
        self.assertTrue(receipt["exact_binary_relu"])
        self.assertTrue(receipt["compressed"])
        self.assertFalse(receipt["valid_cuts"])


if __name__ == "__main__":
    unittest.main()
