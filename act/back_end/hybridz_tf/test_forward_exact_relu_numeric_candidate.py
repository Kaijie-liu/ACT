#!/usr/bin/env python3
"""Fraction and independent-MILP gates for the exact ReLU numeric candidate."""

from __future__ import annotations

from fractions import Fraction
import random
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import Bounds as MilpBounds
from scipy.optimize import LinearConstraint, milp
import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import SparseHZono
from act.back_end.hybridz_tf.forward_exact_relu_numeric_candidate import (
    ACTIVE,
    COMPACT,
    HALF_FREE,
    INACTIVE,
    build_forward_exact_relu_numeric_candidate,
    rigorous_sparse_hz_box_arrays,
)


def _freeze(hz: SparseHZono) -> SparseHZono:
    def seal(value):
        array = np.asarray(value)
        return np.frombuffer(
            array.tobytes(order="C"), dtype=array.dtype
        ).reshape(array.shape)

    for name in ("c", "b", "ub", "col_ids", "bcol_ids"):
        value = getattr(hz, name)
        if value is not None:
            value = seal(value)
            setattr(hz, name, value)
    for matrix in (hz.Gc, hz.Gb, hz.Ac, hz.Ab, hz.Auc, hz.Aub):
        if matrix is not None:
            matrix.data = seal(matrix.data)
            matrix.indices = seal(matrix.indices)
            matrix.indptr = seal(matrix.indptr)
    return hz


def _source(c, G, *, first_id: int = 10) -> SparseHZono:
    center = np.asarray(c, dtype=np.float64).reshape(-1)
    generators = sp.csr_matrix(np.asarray(G, dtype=np.float64))
    generators.eliminate_zeros()
    generators.sort_indices()
    n_out, n_cont = generators.shape
    return _freeze(
        SparseHZono(
            c=center,
            Gc=generators,
            Gb=sp.csr_matrix((n_out, 0), dtype=np.float64),
            Ac=sp.csr_matrix((0, n_cont), dtype=np.float64),
            Ab=sp.csr_matrix((0, 0), dtype=np.float64),
            b=np.zeros(0, dtype=np.float64),
            Auc=sp.csr_matrix((0, n_cont), dtype=np.float64),
            Aub=sp.csr_matrix((0, 0), dtype=np.float64),
            ub=np.zeros(0, dtype=np.float64),
            col_ids=np.arange(first_id, first_id + n_cont, dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
    )


def _fraction_dot(row: sp.csr_matrix, values) -> Fraction:
    total = Fraction(0)
    for column, coefficient in zip(row.indices, row.data):
        total += Fraction.from_float(float(coefficient)) * values[int(column)]
    return total


def _is_feasible(hz: SparseHZono, cont, binary) -> bool:
    cvals = tuple(Fraction(v) for v in cont)
    bvals = tuple(Fraction(v) for v in binary)
    if len(cvals) != hz.n_cont or len(bvals) != hz.n_bin:
        return False
    if any(value < -1 or value > 1 for value in cvals):
        return False
    if any(value not in (-1, 1) for value in bvals):
        return False
    for row in range(hz.n_eq):
        lhs = _fraction_dot(hz.Ac.getrow(row), cvals)
        lhs += _fraction_dot(hz.Ab.getrow(row), bvals)
        if lhs != Fraction.from_float(float(hz.b[row])):
            return False
    for row in range(hz.n_ub):
        lhs = _fraction_dot(hz.Auc.getrow(row), cvals)
        lhs += _fraction_dot(hz.Aub.getrow(row), bvals)
        if lhs > Fraction.from_float(float(hz.ub[row])):
            return False
    return True


def _output(hz: SparseHZono, row: int, cont, binary) -> Fraction:
    cvals = tuple(Fraction(v) for v in cont)
    bvals = tuple(Fraction(v) for v in binary)
    value = Fraction.from_float(float(hz.c[row]))
    value += _fraction_dot(hz.Gc.getrow(row), cvals)
    value += _fraction_dot(hz.Gb.getrow(row), bvals)
    return value


def _assert_tangent(
    case: unittest.TestCase,
    hz: SparseHZono,
    row: int,
    dcont,
    dbinary,
    expected: Fraction,
) -> None:
    dc = tuple(Fraction(v) for v in dcont)
    db = tuple(Fraction(v) for v in dbinary)
    for eq_row in range(hz.n_eq):
        lhs = _fraction_dot(hz.Ac.getrow(eq_row), dc)
        lhs += _fraction_dot(hz.Ab.getrow(eq_row), db)
        case.assertEqual(lhs, 0)
    derivative = _fraction_dot(hz.Gc.getrow(row), dc)
    derivative += _fraction_dot(hz.Gb.getrow(row), db)
    case.assertEqual(derivative, expected)


def _phase_point_and_tangent(result, source: SparseHZono, xi: Fraction):
    """Return exact candidate factors and their derivative with respect to xi."""

    out = result.hz
    assert source.n_out == source.n_cont == 1 and source.n_bin == 0
    c = Fraction.from_float(float(source.c[0]))
    g = Fraction.from_float(float(source.Gc.data[0]))
    x = c + g * xi
    alpha = Fraction.from_float(float(result.lower[0]))
    beta = Fraction.from_float(float(result.upper[0]))
    encoding = result.encoding_by_output[0]
    cont = [xi, Fraction(0), Fraction(0)]
    dcont = [Fraction(1), Fraction(0), Fraction(0)]
    if result.receipt["shared_fixed_one_factor"]:
        cont.append(Fraction(1))
        dcont.append(Fraction(0))
    if encoding == COMPACT:
        if x > 0:
            z = Fraction(-1)
            cont[1] = Fraction(1)
            cont[2] = Fraction(1) - Fraction(2) * x / beta
            dcont[2] = -Fraction(2) * g / beta
        else:
            z = Fraction(1)
            cont[1] = Fraction(2) * x / alpha - Fraction(1)
            cont[2] = Fraction(1)
            dcont[1] = Fraction(2) * g / alpha
    elif encoding == HALF_FREE:
        if x > 0:
            z = Fraction(-1)
            cont[1] = Fraction(0)
            cont[2] = x / beta
            dcont[2] = g / beta
        else:
            z = Fraction(1)
            cont[1] = x / alpha
            cont[2] = Fraction(0)
            dcont[1] = g / alpha
    else:  # pragma: no cover - helper is only for unstable one-row cases
        raise AssertionError(f"unexpected encoding {encoding}")
    return x, tuple(cont), (z,), tuple(dcont), (Fraction(0),)


def _milp_extreme(
    hz: SparseHZono,
    *,
    constant: float,
    cont_coeff,
    bin_coeff,
    maximize: bool,
    fixed_binary: dict[int, int],
) -> float:
    """Optimize one affine expression with an independent scipy MILP model."""

    gc = np.asarray(cont_coeff, dtype=np.float64).reshape(-1)
    gb = np.asarray(bin_coeff, dtype=np.float64).reshape(-1)
    # Transform each signed binary z=2*w-1, w in {0,1}.
    coeff = np.concatenate((gc, 2.0 * gb))
    shifted_constant = float(constant - gb.sum())
    objective = -coeff if maximize else coeff
    Aeq = sp.hstack((hz.Ac, 2.0 * hz.Ab), format="csr")
    beq = hz.b + np.asarray(hz.Ab.sum(axis=1)).reshape(-1)
    Aub = sp.hstack((hz.Auc, 2.0 * hz.Aub), format="csr")
    bub = hz.ub + np.asarray(hz.Aub.sum(axis=1)).reshape(-1)
    constraints = []
    if hz.n_eq:
        constraints.append(LinearConstraint(Aeq, beq, beq))
    if hz.n_ub:
        constraints.append(LinearConstraint(Aub, -np.inf, bub))
    lower = np.concatenate((-np.ones(hz.n_cont), np.zeros(hz.n_bin)))
    upper = np.ones(hz.n_cont + hz.n_bin)
    for index, signed_value in fixed_binary.items():
        bit = 1.0 if int(signed_value) == 1 else 0.0
        lower[hz.n_cont + int(index)] = bit
        upper[hz.n_cont + int(index)] = bit
    result = milp(
        objective,
        integrality=np.concatenate(
            (np.zeros(hz.n_cont, dtype=np.int8), np.ones(hz.n_bin, dtype=np.int8))
        ),
        bounds=MilpBounds(lower, upper),
        constraints=constraints,
        options={"disp": False},
    )
    if not result.success:
        raise AssertionError(result.message)
    value = shifted_constant + float(coeff @ result.x)
    return value


class ForwardExactReLUNumericCandidateTests(unittest.TestCase):
    def test_rounded_rhs_counterexample_is_removed_exactly(self):
        source = _source([0.1], [[0.7]])
        result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.encoding_by_output, (COMPACT,))
        self.assertTrue(result.receipt["shared_fixed_one_factor"])
        self.assertEqual(result.receipt["compact_residual_rows"], 1)

        c = Fraction.from_float(0.1)
        g = Fraction.from_float(0.7)
        alpha = Fraction.from_float(float(result.lower[0]))
        beta = Fraction.from_float(float(result.upper[0]))
        beta_half = beta / 2
        exact_rhs = c - beta_half
        rounded_rhs = Fraction.from_float(float(exact_rhs))
        residual = exact_rhs - rounded_rhs
        self.assertEqual(residual, Fraction(1, 1 << 55))

        # The rounded legacy row admits positive x=residual on its inactive
        # phase with xi1=-1, xi2=1, z=1 and output zero.
        xi = (residual - c) / g
        xi1 = Fraction(-1)
        xi2 = Fraction(1)
        z = Fraction(1)
        legacy_lhs = (
            (alpha / 2) * xi1
            - beta_half * xi2
            + (alpha / 2) * z
            - g * xi
        )
        self.assertEqual(legacy_lhs, rounded_rhs)
        self.assertGreater(c + g * xi, 0)
        self.assertEqual(beta_half * (1 - xi2), 0)

        # The candidate's fixed-one residual makes that same assignment
        # infeasible, while the active representation remains exactly feasible.
        inactive_cont = (xi, xi1, xi2, Fraction(1))
        self.assertFalse(_is_feasible(result.hz, inactive_cont, (z,)))
        x, active_cont, active_bin, _dc, _db = _phase_point_and_tangent(
            result, source, xi
        )
        self.assertEqual(x, residual)
        self.assertTrue(_is_feasible(result.hz, active_cont, active_bin))
        self.assertEqual(_output(result.hz, 0, active_cont, active_bin), residual)

    def test_minimum_subnormal_uses_division_free_complete_graph(self):
        tiny = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
        source = _source([0.0], [[tiny]], first_id=100)
        result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.lower[0], -tiny)
        self.assertEqual(result.upper[0], tiny)
        self.assertEqual(result.encoding_by_output, (HALF_FREE,))
        self.assertFalse(result.receipt["shared_fixed_one_factor"])
        self.assertEqual(result.receipt["compact_residual_rows"], 0)
        self.assertEqual(result.receipt["added_cont"], 2)
        self.assertEqual(result.receipt["added_bin"], 1)
        self.assertEqual(result.receipt["added_eq"], 1)
        self.assertEqual(result.receipt["added_upper"], 4)
        self.assertIn(tiny, result.hz.Gc.data)
        self.assertIn(-tiny, result.hz.Ac.data)

        for xi in (Fraction(-1), Fraction(0), Fraction(1)):
            x, cont, binary, dcont, dbinary = _phase_point_and_tangent(
                result, source, xi
            )
            self.assertTrue(_is_feasible(result.hz, cont, binary))
            self.assertEqual(_output(result.hz, 0, cont, binary), max(0, x))
            _assert_tangent(
                self,
                result.hz,
                0,
                dcont,
                dbinary,
                Fraction.from_float(tiny) if x > 0 else Fraction(0),
            )

    def test_seeded_fraction_phase_point_and_jacobian_census(self):
        rng = random.Random(20260809)
        cases = [(0.1, 0.7), (-0.3, 1.1), (0.2, 0.9)]
        for _ in range(29):
            g = float(rng.randint(2, 31) / rng.choice((7, 10, 13, 16)))
            c = float(g * rng.choice((-3, -2, -1, 0, 1, 2, 3)) / 8.0)
            cases.append((c, g))
        for case_no, (center, generator) in enumerate(cases):
            with self.subTest(case=case_no, center=center, generator=generator):
                source = _source([center], [[generator]], first_id=1000 + case_no * 8)
                result = build_forward_exact_relu_numeric_candidate(source)
                self.assertIn(result.encoding_by_output[0], (COMPACT, HALF_FREE))
                for xi in (
                    Fraction(-1), Fraction(-3, 4), Fraction(-1, 4),
                    Fraction(0), Fraction(1, 4), Fraction(3, 4), Fraction(1),
                ):
                    x, cont, binary, dcont, dbinary = _phase_point_and_tangent(
                        result, source, xi
                    )
                    self.assertTrue(_is_feasible(result.hz, cont, binary))
                    self.assertEqual(
                        _output(result.hz, 0, cont, binary), max(Fraction(0), x)
                    )
                    expected_jacobian = (
                        Fraction.from_float(generator) if x > 0 else Fraction(0)
                    )
                    _assert_tangent(
                        self, result.hz, 0, dcont, dbinary, expected_jacobian
                    )

    def test_preexisting_binary_factor_is_linked_in_both_phases(self):
        source = _freeze(
            SparseHZono(
                c=np.asarray([0.1], dtype=np.float64),
                Gc=sp.csr_matrix([[0.4]], dtype=np.float64),
                Gb=sp.csr_matrix([[0.2]], dtype=np.float64),
                Ac=sp.csr_matrix((0, 1), dtype=np.float64),
                Ab=sp.csr_matrix((0, 1), dtype=np.float64),
                b=np.zeros(0, dtype=np.float64),
                Auc=sp.csr_matrix((0, 1), dtype=np.float64),
                Aub=sp.csr_matrix((0, 1), dtype=np.float64),
                ub=np.zeros(0, dtype=np.float64),
                col_ids=np.asarray([3_000_000], dtype=np.int64),
                bcol_ids=np.asarray([3_000_001], dtype=np.int64),
            )
        )
        result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.encoding_by_output, (COMPACT,))
        c = Fraction.from_float(0.1)
        gc = Fraction.from_float(0.4)
        gb = Fraction.from_float(0.2)
        alpha = Fraction.from_float(float(result.lower[0]))
        beta = Fraction.from_float(float(result.upper[0]))
        for old_binary in (Fraction(-1), Fraction(1)):
            for xi in (Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2), Fraction(1)):
                x = c + gc * xi + gb * old_binary
                cont = [xi, Fraction(0), Fraction(0)]
                dcont = [Fraction(1), Fraction(0), Fraction(0)]
                if result.receipt["shared_fixed_one_factor"]:
                    cont.append(Fraction(1))
                    dcont.append(Fraction(0))
                if x > 0:
                    new_binary = Fraction(-1)
                    cont[1] = Fraction(1)
                    cont[2] = Fraction(1) - Fraction(2) * x / beta
                    dcont[2] = -Fraction(2) * gc / beta
                else:
                    new_binary = Fraction(1)
                    cont[1] = Fraction(2) * x / alpha - Fraction(1)
                    cont[2] = Fraction(1)
                    dcont[1] = Fraction(2) * gc / alpha
                binary = (old_binary, new_binary)
                self.assertTrue(_is_feasible(result.hz, cont, binary))
                self.assertEqual(
                    _output(result.hz, 0, cont, binary), max(Fraction(0), x)
                )
                _assert_tangent(
                    self,
                    result.hz,
                    0,
                    dcont,
                    (Fraction(0), Fraction(0)),
                    gc if x > 0 else Fraction(0),
                )

    def test_outward_bounds_cover_exact_sum_when_sparse_sum_is_inward(self):
        small = float(2.0 ** -54)
        source = _source([0.0], [[1.0, small, small]], first_id=4000)
        lower, upper = rigorous_sparse_hz_box_arrays(source)
        exact_radius = Fraction(1) + Fraction(1, 1 << 53)
        self.assertLessEqual(Fraction.from_float(float(lower[0])), -exact_radius)
        self.assertGreaterEqual(Fraction.from_float(float(upper[0])), exact_radius)
        naive = float(np.asarray(np.abs(source.Gc).sum(axis=1)).reshape(-1)[0])
        self.assertLess(Fraction.from_float(naive), exact_radius)
        self.assertGreater(upper[0], naive)

    def test_mixed_layer_keeps_compact_rows_and_localizes_half_free_cost(self):
        tiny = float(np.nextafter(np.float64(0.0), np.float64(1.0)))
        source = _source([0.1, 0.0], [[0.7, 0.0], [0.0, tiny]], first_id=5000)
        result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.encoding_by_output, (COMPACT, HALF_FREE))
        self.assertEqual(result.receipt["compact_rows"], 1)
        self.assertEqual(result.receipt["half_free_rows"], 1)
        self.assertTrue(result.receipt["shared_fixed_one_factor"])
        self.assertEqual(result.receipt["added_cont"], 5)
        self.assertEqual(result.receipt["added_bin"], 2)
        self.assertEqual(result.receipt["added_eq"], 3)
        self.assertEqual(result.receipt["added_upper"], 6)

    def test_exact_rhs_needs_no_shared_factor_when_residual_is_zero(self):
        source = _source([0.0], [[1.0]], first_id=6000)
        result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.encoding_by_output, (COMPACT,))
        self.assertFalse(result.receipt["shared_fixed_one_factor"])
        self.assertEqual(result.receipt["added_cont"], 2)
        self.assertEqual(result.receipt["added_eq"], 1)
        self.assertEqual(result.receipt["added_upper"], 2)

    def test_stable_rows_are_disjoint_exact_copy_or_zero(self):
        source = _source(
            [-2.0, 0.0, 2.0],
            [[0.5, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]],
            first_id=7000,
        )
        result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.phase_counts, (2, 1, 0))
        self.assertEqual(result.encoding_by_output, (INACTIVE, ACTIVE, ACTIVE))
        self.assertEqual(result.hz.n_cont, source.n_cont)
        self.assertEqual(result.hz.n_bin, source.n_bin)
        self.assertEqual(result.hz.n_eq, source.n_eq)
        self.assertEqual(result.hz.n_ub, source.n_ub)
        np.testing.assert_array_equal(result.hz.c, np.asarray([0.0, 0.0, 2.0]))
        np.testing.assert_array_equal(
            result.hz.Gc.toarray(),
            np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
        )

    def test_parent_constraint_prefix_is_copied_exactly_and_canonical(self):
        source = _freeze(
            SparseHZono(
                c=np.asarray([-0.25], dtype=np.float64),
                Gc=sp.csr_matrix([[1.0]], dtype=np.float64),
                Gb=sp.csr_matrix((1, 0), dtype=np.float64),
                Ac=sp.csr_matrix([[1.0]], dtype=np.float64),
                Ab=sp.csr_matrix((1, 0), dtype=np.float64),
                b=np.asarray([0.5], dtype=np.float64),
                Auc=sp.csr_matrix([[-1.0]], dtype=np.float64),
                Aub=sp.csr_matrix((1, 0), dtype=np.float64),
                ub=np.asarray([0.75], dtype=np.float64),
                col_ids=np.asarray([7_500_000], dtype=np.int64),
                bcol_ids=np.zeros(0, dtype=np.int64),
            )
        )
        result = build_forward_exact_relu_numeric_candidate(source)
        out = result.hz
        np.testing.assert_array_equal(out.Ac[: source.n_eq, :1].toarray(), source.Ac.toarray())
        np.testing.assert_array_equal(out.Ab[: source.n_eq, :0].toarray(), source.Ab.toarray())
        np.testing.assert_array_equal(out.b[: source.n_eq], source.b)
        np.testing.assert_array_equal(out.Auc[: source.n_ub, :1].toarray(), source.Auc.toarray())
        np.testing.assert_array_equal(out.Aub[: source.n_ub, :0].toarray(), source.Aub.toarray())
        np.testing.assert_array_equal(out.ub[: source.n_ub], source.ub)
        for matrix in (out.Gc, out.Gb, out.Ac, out.Ab, out.Auc, out.Aub):
            self.assertTrue(matrix.has_canonical_format)
            self.assertTrue(matrix.has_sorted_indices)
            self.assertFalse(np.any(matrix.data == 0.0))

    def test_independent_milp_checks_both_exact_phases(self):
        source = _source([-0.125], [[1.25]], first_id=8000)
        result = build_forward_exact_relu_numeric_candidate(source)
        out = result.hz
        x_gc = np.zeros(out.n_cont, dtype=np.float64)
        x_gc[: source.n_cont] = source.Gc.getrow(0).toarray().reshape(-1)
        x_gb = np.zeros(out.n_bin, dtype=np.float64)
        y_gc = out.Gc.getrow(0).toarray().reshape(-1)
        y_gb = out.Gb.getrow(0).toarray().reshape(-1)
        z_index = source.n_bin

        max_inactive_x = _milp_extreme(
            out,
            constant=float(source.c[0]),
            cont_coeff=x_gc,
            bin_coeff=x_gb,
            maximize=True,
            fixed_binary={z_index: 1},
        )
        max_inactive_y = _milp_extreme(
            out,
            constant=float(out.c[0]),
            cont_coeff=y_gc,
            bin_coeff=y_gb,
            maximize=True,
            fixed_binary={z_index: 1},
        )
        min_inactive_y = _milp_extreme(
            out,
            constant=float(out.c[0]),
            cont_coeff=y_gc,
            bin_coeff=y_gb,
            maximize=False,
            fixed_binary={z_index: 1},
        )
        min_active_x = _milp_extreme(
            out,
            constant=float(source.c[0]),
            cont_coeff=x_gc,
            bin_coeff=x_gb,
            maximize=False,
            fixed_binary={z_index: -1},
        )
        diff_gc = y_gc - x_gc
        diff_gb = y_gb - x_gb
        diff_const = float(out.c[0] - source.c[0])
        max_diff = _milp_extreme(
            out,
            constant=diff_const,
            cont_coeff=diff_gc,
            bin_coeff=diff_gb,
            maximize=True,
            fixed_binary={z_index: -1},
        )
        min_diff = _milp_extreme(
            out,
            constant=diff_const,
            cont_coeff=diff_gc,
            bin_coeff=diff_gb,
            maximize=False,
            fixed_binary={z_index: -1},
        )
        self.assertLessEqual(max_inactive_x, 1e-8)
        self.assertAlmostEqual(max_inactive_y, 0.0, places=8)
        self.assertAlmostEqual(min_inactive_y, 0.0, places=8)
        self.assertGreaterEqual(min_active_x, -1e-8)
        self.assertAlmostEqual(max_diff, 0.0, places=8)
        self.assertAlmostEqual(min_diff, 0.0, places=8)

    def test_supplied_bounds_have_no_tightening_authority(self):
        source = _source([0.1], [[0.7]], first_id=9000)
        lower, upper = rigorous_sparse_hz_box_arrays(source)
        exact = Bounds(
            lb=torch.from_numpy(lower.copy()).reshape(1, -1),
            ub=torch.from_numpy(upper.copy()).reshape(1, -1),
        )
        accepted = build_forward_exact_relu_numeric_candidate(
            source, pre_bounds=exact
        )
        self.assertTrue(accepted.receipt["bounds_internally_recomputed"])
        inward = Bounds(
            lb=exact.lb.clone(),
            ub=torch.nextafter(exact.ub, torch.full_like(exact.ub, -torch.inf)),
        )
        with self.assertRaises(ValueError):
            build_forward_exact_relu_numeric_candidate(source, pre_bounds=inward)

    def test_stable_ids_are_globally_disjoint_and_above_parent(self):
        source = _source([0.1], [[0.7]], first_id=10_000_000)
        result = build_forward_exact_relu_numeric_candidate(source)
        out = result.hz
        self.assertTrue(np.array_equal(out.col_ids[: source.n_cont], source.col_ids))
        self.assertGreater(int(out.col_ids[source.n_cont :].min()), 10_000_000)
        self.assertGreater(int(out.bcol_ids.min()), int(out.col_ids.max()))
        self.assertEqual(
            np.intersect1d(out.col_ids, out.bcol_ids, assume_unique=True).size, 0
        )

    def test_sealed_candidate_can_feed_the_next_forward_exact_relu(self):
        first = build_forward_exact_relu_numeric_candidate(
            _source([0.1], [[0.7]], first_id=10_500_000)
        )
        second = build_forward_exact_relu_numeric_candidate(first.hz)
        self.assertEqual(second.phase_counts, (1, 0, 0))
        self.assertEqual(second.encoding_by_output, (ACTIVE,))
        self.assertEqual(second.hz.n_cont, first.hz.n_cont)
        self.assertEqual(second.hz.n_bin, first.hz.n_bin)
        self.assertEqual(second.hz.n_eq, first.hz.n_eq)
        self.assertEqual(second.hz.n_ub, first.hz.n_ub)
        np.testing.assert_array_equal(second.hz.c, first.hz.c)
        np.testing.assert_array_equal(second.hz.Gc.toarray(), first.hz.Gc.toarray())

    def test_raw_property_spoof_is_ignored_but_subclass_and_mutability_fail_closed(self):
        source = _source([0.1], [[0.7]], first_id=11_000_000)
        with mock.patch.object(
            SparseHZono, "n_out", new=property(lambda _self: 999_999)
        ):
            result = build_forward_exact_relu_numeric_candidate(source)
        self.assertEqual(result.hz.c.size, 1)

        class SparseSubclass(SparseHZono):
            pass

        writable = SparseSubclass(
            c=np.asarray([0.1]),
            Gc=sp.csr_matrix([[0.7]]),
            Gb=sp.csr_matrix((1, 0)),
            Ac=sp.csr_matrix((0, 1)),
            Ab=sp.csr_matrix((0, 0)),
            b=np.zeros(0),
            col_ids=np.asarray([12_000_000], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        with self.assertRaises(TypeError):
            build_forward_exact_relu_numeric_candidate(writable)

        ordinary = SparseHZono(
            c=np.asarray([0.1]),
            Gc=sp.csr_matrix([[0.7]]),
            Gb=sp.csr_matrix((1, 0)),
            Ac=sp.csr_matrix((0, 1)),
            Ab=sp.csr_matrix((0, 0)),
            b=np.zeros(0),
            col_ids=np.asarray([13_000_000], dtype=np.int64),
            bcol_ids=np.zeros(0, dtype=np.int64),
        )
        with self.assertRaisesRegex(ValueError, "read-only"):
            build_forward_exact_relu_numeric_candidate(ordinary)

    def test_malformed_csr_fails_closed_even_if_cached_flags_claim_canonical(self):
        def base():
            return SparseHZono(
                c=np.asarray([0.0]),
                Gc=sp.csr_matrix([[1.0, 2.0]], dtype=np.float64),
                Gb=sp.csr_matrix((1, 0), dtype=np.float64),
                Ac=sp.csr_matrix((0, 2), dtype=np.float64),
                Ab=sp.csr_matrix((0, 0), dtype=np.float64),
                b=np.zeros(0),
                col_ids=np.asarray([13_500_000, 13_500_001], dtype=np.int64),
                bcol_ids=np.zeros(0, dtype=np.int64),
            )

        negative = base()
        negative.Gc.indices[0] = -1
        negative.Gc.has_sorted_indices = True
        negative.Gc.has_canonical_format = True
        with self.assertRaisesRegex(ValueError, "outside"):
            build_forward_exact_relu_numeric_candidate(_freeze(negative))

        at_ncols = base()
        at_ncols.Gc.indices[-1] = at_ncols.Gc.shape[1]
        at_ncols.Gc.has_sorted_indices = True
        at_ncols.Gc.has_canonical_format = True
        with self.assertRaisesRegex(ValueError, "outside"):
            build_forward_exact_relu_numeric_candidate(_freeze(at_ncols))

        descending = base()
        descending.Gc.indices[:] = np.asarray([1, 0], dtype=descending.Gc.indices.dtype)
        descending.Gc.has_sorted_indices = True
        descending.Gc.has_canonical_format = True
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            build_forward_exact_relu_numeric_candidate(_freeze(descending))

        bad_endpoint = base()
        bad_endpoint.Gc.indptr[-1] = 1
        bad_endpoint.Gc.has_sorted_indices = True
        bad_endpoint.Gc.has_canonical_format = True
        with self.assertRaisesRegex(ValueError, "row pointers"):
            build_forward_exact_relu_numeric_candidate(_freeze(bad_endpoint))

    def test_snapshot_is_independent_and_candidate_buffers_are_read_only(self):
        source = _source([0.1], [[0.7]], first_id=14_000_000)
        result = build_forward_exact_relu_numeric_candidate(source)
        before = result.hz.Ac.data.copy()
        source.Gc = sp.csr_matrix([[0.5]], dtype=np.float64)
        np.testing.assert_array_equal(result.hz.Ac.data, before)
        with self.assertRaises(ValueError):
            result.hz.Ac.data[0] = 0.0
        with self.assertRaises(ValueError):
            result.hz.Ac.data.flags.writeable = True
        with self.assertRaises(ValueError):
            result.hz.c.flags.writeable = True
        with self.assertRaises(ValueError):
            result.lower.flags.writeable = True
        with self.assertRaises(TypeError):
            result.receipt["proof_authority"] = True

    def test_receipt_is_explicitly_disconnected_and_has_no_band(self):
        receipt = build_forward_exact_relu_numeric_candidate(
            _source([0.0], [[1.0]], first_id=15_000_000)
        ).receipt
        self.assertFalse(receipt["proof_authority"])
        self.assertFalse(receipt["production_authority"])
        self.assertFalse(receipt["authenticity_verified"])
        self.assertTrue(receipt["forward_only"])
        self.assertTrue(receipt["complete_graph_exact_over_stored_reals"])
        self.assertFalse(receipt["equality_band"])
        self.assertFalse(receipt["solver_called"])
        self.assertFalse(receipt["caller_tightening_authority"])
        self.assertTrue(receipt["integer_dyadic_bounds"])
        self.assertFalse(receipt["python_fraction_used"])
        self.assertTrue(receipt["numeric_buffers_bytes_sealed"])
        self.assertFalse(receipt["public_hz_rebind_protected"])
        expected_keys = {
            "schema", "proof_authority", "production_authority",
            "authenticity_verified", "forward_only",
            "complete_graph_exact_over_stored_reals", "equality_band",
            "solver_called", "bounds_internally_recomputed",
            "integer_dyadic_bounds", "python_fraction_used",
            "caller_tightening_authority", "raw_readonly_snapshot",
            "numeric_buffers_bytes_sealed", "public_hz_rebind_protected",
            "active", "inactive", "unstable", "compact_rows",
            "compact_residual_rows", "half_free_rows",
            "shared_fixed_one_factor", "added_cont", "added_bin",
            "added_eq", "added_upper", "added_constraint_nnz",
        }
        self.assertEqual(set(receipt), expected_keys)
        for key in (
            "proof_authority", "production_authority", "authenticity_verified",
            "forward_only", "complete_graph_exact_over_stored_reals",
            "equality_band", "solver_called", "bounds_internally_recomputed",
            "integer_dyadic_bounds", "python_fraction_used",
            "caller_tightening_authority", "raw_readonly_snapshot",
            "numeric_buffers_bytes_sealed", "public_hz_rebind_protected",
            "shared_fixed_one_factor",
        ):
            self.assertIs(type(receipt[key]), bool)
        for key in (
            "active", "inactive", "unstable", "compact_rows",
            "compact_residual_rows", "half_free_rows", "added_cont",
            "added_bin", "added_eq", "added_upper", "added_constraint_nnz",
        ):
            self.assertIs(type(receipt[key]), int)
        self.assertIs(type(receipt["schema"]), str)


if __name__ == "__main__":
    unittest.main()
