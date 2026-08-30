#!/usr/bin/env python3
"""Focused gates for the disconnected live-row exact-ReLU stream.

The tests use analytic toy graphs only.  They do not execute ONNX models,
sample network inputs, run PGD, or ask the production verifier for a verdict.
For the scalar residual graph, every binary assignment and every rational LP
vertex of the stored HybridZ constraints is enumerated with ``Fraction``.
"""

from __future__ import annotations

from fractions import Fraction
import itertools
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import (
    forward_exact_relu_live_row_stream_candidate as _candidate_module,
)
from act.back_end.hybridz_tf.exact_sparse_conv_csr_candidate import (
    exact_sparse_conv2d_matrix_from_layer_candidate,
)
from act.back_end.hybridz_tf.forward_exact_relu_live_row_stream_candidate import (
    ExactReLULiveRowStreamResult,
    build_forward_exact_relu_live_row_stream_candidate,
)
from act.back_end.hybridz_tf import test_operator_add_fusion as _toy_audit
from act.back_end.hybridz_tf.test_operator_residual_normal_form import (
    _scalar_relu_toy,
)


def _fraction(value: object) -> Fraction:
    return Fraction.from_float(float(value))


def _fact(lower: list[float], upper: list[float]) -> Fact:
    return Fact(
        Bounds(
            torch.tensor([lower], dtype=torch.float64),
            torch.tensor([upper], dtype=torch.float64),
        ),
        ConSet(),
    )


def _exact_two_variable_projection(
    result: ExactReLULiveRowStreamResult,
) -> tuple[Fraction, Fraction]:
    """Enumerate the exact stored binary64 polytope for a two-factor toy."""

    hz = result.hz
    if hz.n_out != 1 or hz.n_cont < 2:
        raise AssertionError("the exact vertex oracle requires one output")

    expressions: list[tuple[Fraction, tuple[Fraction, Fraction]] | None] = [
        (Fraction(0), (Fraction(1), Fraction(0))),
        (Fraction(0), (Fraction(0), Fraction(1))),
    ] + [None] * (hz.n_cont - 2)
    for row in range(hz.n_eq):
        if hz.Ab.getrow(row).nnz:
            raise AssertionError("the local-link oracle does not accept binary equalities")
        coefficients = hz.Ac.getrow(row)
        unknown = [
            int(column)
            for column in coefficients.indices
            if expressions[int(column)] is None
        ]
        if len(unknown) != 1:
            raise AssertionError("an equality does not define one local factor")
        target = unknown[0]
        target_coefficient = _fraction(hz.Ac[row, target])
        constant = Fraction(0)
        base = [Fraction(0), Fraction(0)]
        for column, value in zip(coefficients.indices, coefficients.data):
            if int(column) == target:
                continue
            expression = expressions[int(column)]
            if expression is None:
                raise AssertionError("local factors are not topologically ordered")
            coefficient = _fraction(value)
            constant += coefficient * expression[0]
            base[0] += coefficient * expression[1][0]
            base[1] += coefficient * expression[1][1]
        rhs = _fraction(hz.b[row])
        expressions[target] = (
            (rhs - constant) / target_coefficient,
            (-base[0] / target_coefficient, -base[1] / target_coefficient),
        )
    if any(expression is None for expression in expressions):
        raise AssertionError("the exact vertex oracle found an unconstrained extra factor")

    def substitute(row: sp.csr_matrix) -> tuple[Fraction, Fraction, Fraction]:
        constant = Fraction(0)
        first = Fraction(0)
        second = Fraction(0)
        for column, value in zip(row.indices, row.data):
            expression = expressions[int(column)]
            assert expression is not None
            coefficient = _fraction(value)
            constant += coefficient * expression[0]
            first += coefficient * expression[1][0]
            second += coefficient * expression[1][1]
        return first, second, constant

    objective_first, objective_second, objective_constant = substitute(
        hz.Gc.getrow(0)
    )
    center = _fraction(hz.c[0]) + objective_constant
    binary_objective = tuple(
        _fraction(value) for value in hz.Gb.getrow(0).toarray()[0]
    )
    lower: Fraction | None = None
    upper: Fraction | None = None
    for assignment in itertools.product((-1, 1), repeat=hz.n_bin):
        inequalities: list[tuple[Fraction, Fraction, Fraction]] = []
        for row in range(hz.n_ub):
            first, second, constant = substitute(hz.Auc.getrow(row))
            binary = hz.Aub.getrow(row).toarray()[0]
            rhs = _fraction(hz.ub[row]) - constant - sum(
                (_fraction(value) * bit for value, bit in zip(binary, assignment)),
                Fraction(0),
            )
            inequalities.append((first, second, rhs))
        for expression in expressions:
            assert expression is not None
            constant, (first, second) = expression
            inequalities.append((first, second, Fraction(1) - constant))
            inequalities.append((-first, -second, Fraction(1) + constant))
        vertices: set[tuple[Fraction, Fraction]] = set()
        for left, right in itertools.combinations(inequalities, 2):
            a, b, c = left
            d, e, f = right
            determinant = a * e - b * d
            if determinant == 0:
                continue
            point = ((c * e - b * f) / determinant, (a * f - c * d) / determinant)
            if all(p * point[0] + q * point[1] <= r for p, q, r in inequalities):
                vertices.add(point)
        binary_shift = sum(
            (coefficient * bit for coefficient, bit in zip(binary_objective, assignment)),
            Fraction(0),
        )
        for first, second in vertices:
            value = (
                center
                + binary_shift
                + objective_first * first
                + objective_second * second
            )
            lower = value if lower is None else min(lower, value)
            upper = value if upper is None else max(upper, value)
    if lower is None or upper is None:
        raise AssertionError("the exact vertex oracle found no feasible phase")
    return lower, upper


def _extend_constraint_local_factors(
    hz: object, prefix: tuple[Fraction, ...]
) -> tuple[Fraction, ...]:
    values: list[Fraction | None] = list(prefix) + [None] * (hz.n_cont - len(prefix))
    for row in range(hz.n_eq):
        coefficients = hz.Ac.getrow(row)
        unknown = [
            int(column)
            for column in coefficients.indices
            if values[int(column)] is None
        ]
        if len(unknown) != 1:
            raise AssertionError("an equality does not define one local factor")
        target = unknown[0]
        known = sum(
            (
                _fraction(value) * values[int(column)]
                for column, value in zip(coefficients.indices, coefficients.data)
                if int(column) != target
            ),
            Fraction(0),
        )
        values[target] = (
            _fraction(hz.b[row]) - known
        ) / _fraction(hz.Ac[row, target])
        if abs(values[target]) > 1:
            raise AssertionError("a constraint-local factor escaped its box")
    completed = tuple(Fraction(0) if value is None else value for value in values)
    for row in range(hz.n_eq):
        lhs = sum(
            (
                _fraction(value) * completed[int(column)]
                for column, value in zip(
                    hz.Ac.getrow(row).indices, hz.Ac.getrow(row).data
                )
            ),
            Fraction(0),
        )
        if lhs != _fraction(hz.b[row]):
            raise AssertionError("a constraint-local equality did not replay")
    return completed


@unittest.skipUnless(torch.cuda.is_available(), "live-row candidate requires CUDA")
class ExactReLULiveRowStreamCandidateTests(unittest.TestCase):
    def test_local_link_scale_encloses_the_exact_stored_row_l1(self) -> None:
        matrix = sp.csr_matrix(
            np.asarray(
                [
                    [0.1, -0.7, 2.0**-1022, 0.0],
                    [2.0**53, 1.0, -(2.0**53), 0.25],
                    [0.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        )
        scales = _candidate_module._oh._row_l1_upper(
            matrix, name="local_link_scale_test"
        )
        scales[scales == 0.0] = 1.0
        for row in range(matrix.shape[0]):
            exact_l1 = sum(
                (_fraction(abs(value)) for value in matrix.getrow(row).data),
                Fraction(0),
            )
            self.assertGreaterEqual(_fraction(scales[row]), exact_l1)
            for signs in itertools.product((-1, 1), repeat=matrix.getrow(row).nnz):
                exact_value = sum(
                    (
                        _fraction(value) * sign
                        for value, sign in zip(matrix.getrow(row).data, signs)
                    ),
                    Fraction(0),
                )
                self.assertLessEqual(abs(exact_value / _fraction(scales[row])), 1)

    def test_tight_big_m_keeps_original_output_factor_scale(self) -> None:
        empty = np.zeros(0, dtype=np.int64)
        frame = _candidate_module._PhaseFrame(
            np.asarray([-1.0], dtype=np.float64),
            np.asarray([1.0], dtype=np.float64),
            empty,
            empty,
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            # The preliminary stream used U=2 and therefore y=1+eta_y.
            np.asarray([1.0], dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            empty,
            np.zeros(0, dtype=bool),
            empty,
            np.asarray([0], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        )
        equal_continuous, equal_binary, equal_rhs, continuous, binary, upper = (
            _candidate_module._build_constraints(
            (SimpleNamespace(id=11),),
            {11: frame},
            {11: sp.csr_matrix([[1.0, 0.0, 0.0]], dtype=np.float64)},
            {
                11: _candidate_module._Shadow(
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray([0.0], dtype=np.float64),
                    np.asarray([1.0], dtype=np.float64),
                )
            },
            n_cont=3,
            n_bin=1,
            )
        )
        self.assertEqual(equal_continuous.shape, (1, 3))
        self.assertEqual(equal_binary.shape, (1, 1))
        np.testing.assert_array_equal(equal_rhs, np.zeros(1, dtype=np.float64))

        # The tight upper U=1 changes y<=U*phase to
        # eta_y - z/2 <= -1/2 while the output factor scale remains one.
        self.assertEqual(_fraction(binary[2, 0]), Fraction(-1, 2))
        self.assertEqual(_fraction(upper[2]), Fraction(-1, 2))
        for xi in (Fraction(-1), Fraction(0), Fraction(1)):
            y = max(Fraction(0), xi)
            eta_y = y - 1
            phase = 1 if xi >= 0 else -1
            link_scale = -_fraction(equal_continuous[0, 2])
            local_preactivation = xi / link_scale
            values = (xi, eta_y, local_preactivation)
            self.assertEqual(
                sum(
                    (
                        _fraction(value) * values[column]
                        for column, value in zip(
                            equal_continuous.getrow(0).indices,
                            equal_continuous.getrow(0).data,
                        )
                    ),
                    Fraction(0),
                ),
                Fraction(0),
            )
            for row in range(3):
                lhs = sum(
                    (
                        _fraction(value) * values[column]
                        for column, value in zip(
                            continuous.getrow(row).indices,
                            continuous.getrow(row).data,
                        )
                    ),
                    Fraction(0),
                )
                lhs += _fraction(binary[row, 0]) * phase
                self.assertLessEqual(lhs, _fraction(upper[row]))

    def test_deferred_active_rows_equal_exact_graph_with_phase_fixed(self) -> None:
        lower = np.asarray([-1.0], dtype=np.float64)
        upper = np.asarray([2.0], dtype=np.float64)
        empty = np.zeros(0, dtype=np.int64)
        original = _candidate_module._PhaseFrame(
            lower,
            upper,
            empty,
            empty,
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1.0], dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            empty,
            np.zeros(0, dtype=bool),
            empty,
            np.asarray([0], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        )
        deferred = _candidate_module._PhaseFrame(
            lower,
            upper,
            np.asarray([0], dtype=np.int64),
            empty,
            empty,
            empty,
            empty,
            np.asarray([1.0], dtype=np.float64),
            np.asarray([0], dtype=np.int64),
            np.asarray([1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([True], dtype=bool),
            np.asarray([1], dtype=np.int64),
            np.asarray([0], dtype=np.int64),
            np.asarray([2], dtype=np.int64),
        )
        preactivation = {7: sp.csr_matrix([[0.5, 0.0, 0.0]], dtype=np.float64)}
        shadow = {
            7: _candidate_module._Shadow(
                np.asarray([0.5], dtype=np.float64),
                np.asarray([0.0], dtype=np.float64),
                np.asarray([1.0], dtype=np.float64),
            )
        }
        order = (SimpleNamespace(id=7),)
        exact_E, exact_Eb, exact_e, exact_A, exact_B, exact_u = _candidate_module._build_constraints(
            order,
            {7: original},
            preactivation,
            shadow,
            n_cont=3,
            n_bin=1,
        )
        fixed_E, fixed_Eb, fixed_e, fixed_A, fixed_B, fixed_u = _candidate_module._build_constraints(
            order,
            {7: deferred},
            preactivation,
            shadow,
            n_cont=3,
            n_bin=0,
        )

        # z=+1 leaves the first two exact-ReLU facets.  The third facet is
        # eta_y<=1, already present in the HybridZ factor box.
        exact_fixed_u = exact_u - exact_B.toarray()[:, 0]
        np.testing.assert_array_equal(exact_A[:2].indptr, fixed_A.indptr)
        np.testing.assert_array_equal(exact_A[:2].indices, fixed_A.indices)
        np.testing.assert_array_equal(
            exact_A[:2].data.view(np.uint64), fixed_A.data.view(np.uint64)
        )
        np.testing.assert_array_equal(
            exact_fixed_u[:2].view(np.uint64), fixed_u.view(np.uint64)
        )
        self.assertEqual(fixed_B.shape, (2, 0))
        np.testing.assert_array_equal(exact_E.indptr, fixed_E.indptr)
        np.testing.assert_array_equal(exact_E.indices, fixed_E.indices)
        np.testing.assert_array_equal(
            exact_E.data.view(np.uint64), fixed_E.data.view(np.uint64)
        )
        self.assertEqual(exact_Eb.shape, (1, 1))
        self.assertEqual(fixed_Eb.shape, (1, 0))
        np.testing.assert_array_equal(exact_e, fixed_e)
        self.assertEqual(exact_A.getrow(2).indices.tolist(), [1])
        self.assertEqual(_fraction(exact_fixed_u[2]), Fraction(1))

    def test_ordered_csr_kernel_is_concurrent_bitwise_and_enveloped(self) -> None:
        matrix = sp.csr_matrix(
            np.asarray(
                [
                    [2.0**53, 1.0, -(2.0**53)],
                    [2.0**-1022, -2.0**-1022, 1.0],
                ],
                dtype=np.float64,
            )
        )
        dense = torch.ones((3, 1), dtype=torch.float64, device="cuda")
        device = _candidate_module._device_csr(matrix)

        def invoke(_index: int) -> np.ndarray:
            return (
                _candidate_module._ordered_csr_dense(device, dense)
                .detach()
                .cpu()
                .numpy()
            )

        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(invoke, range(4)))
        for output in outputs[1:]:
            np.testing.assert_array_equal(
                output.view(np.uint64), outputs[0].view(np.uint64)
            )
        for row in range(matrix.shape[0]):
            exact = sum(
                (
                    _fraction(value)
                    for value in matrix.getrow(row).data
                ),
                Fraction(0),
            )
            rounded = _fraction(outputs[0][row, 0])
            exact_mass = sum(
                (_fraction(abs(value)) for value in matrix.getrow(row).data),
                Fraction(0),
            )
            gamma = _candidate_module._oh._gamma_ops(
                2 * matrix.getrow(row).nnz + 2,
                name="ordered_csr_test_gamma",
            )
            envelope = _fraction(gamma) * exact_mass
            self.assertLessEqual(abs(rounded - exact), envelope)

    def test_residual_exact_graph_fraction_projection(self) -> None:
        toy = _toy_audit.OperatorAddFusionAuditTests._residual_add_relu_toy()
        before = dict(toy.facts)
        before[4] = _fact([-1.75], [1.25])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )

        exact = _toy_audit._exact_graph_range(toy)
        self.assertEqual((exact.lower, exact.upper), (Fraction(0), Fraction(5, 4)))
        self.assertEqual(_exact_two_variable_projection(result), (exact.lower, exact.upper))
        self.assertEqual(result.hz.n_bin, 1)
        self.assertEqual(result.hz.n_eq, 1)
        self.assertEqual(result.hz.n_ub, 3)
        self.assertEqual(result.receipt.exact_rows, 1)
        self.assertTrue(result.receipt.all_unstable_exact)
        self.assertEqual(result.receipt.triangle_rows, 0)
        self.assertEqual(result.receipt.runtime_fallbacks, 0)
        self.assertFalse(result.receipt.input_sampling_used)
        self.assertFalse(result.receipt.pgd_used)
        self.assertFalse(result.receipt.concrete_onnx_execution)
        self.assertFalse(result.receipt.proof_authority)
        self.assertFalse(result.receipt.verdict_authority)

    def test_active_exact_inactive_rows_and_live_affine_pruning(self) -> None:
        toy = _toy_audit.OperatorAddFusionAuditTests._mixed_phase_affine_toy()
        before = dict(toy.facts)
        before[5] = _fact([0.5, -2.0, -5.0], [4.5, 2.0, -1.0])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )

        self.assertEqual(result.hz.n_bin, 1)
        self.assertEqual(result.receipt.exact_rows, 1)
        self.assertEqual(result.hz.n_eq, 1)
        self.assertEqual(result.hz.n_ub, 3)
        self.assertLess(
            result.receipt.streamed_affine_nnz,
            result.receipt.full_affine_nnz,
        )
        self.assertEqual(result.receipt.full_affine_nnz, 7)
        self.assertEqual(result.receipt.streamed_affine_nnz, 5)

    def test_stable_affine_relu_preserves_stored_jacobian(self) -> None:
        toy = _scalar_relu_toy(
            lower=1,
            upper=2,
            pre_weight=Fraction(3, 4),
            pre_bias=Fraction(1, 4),
            out_weight=-2,
        )
        before = dict(toy.facts)
        before[3] = _fact([1.0], [1.75])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )

        # x = 3/2 + (1/2) xi, hence d(-2*(3x/4+1/4))/d xi = -3/4.
        self.assertEqual(result.hz.n_bin, 0)
        self.assertEqual(result.hz.n_ub, 0)
        radius = float(result.hz.operator_input_radius[0])
        expected = np.float64(-2.0 * np.float64(0.75 * radius))
        self.assertEqual(
            np.float64(result.hz.Gc[0, 0]).view(np.uint64),
            expected.view(np.uint64),
        )
        self.assertEqual(_fraction(result.hz.c[0]), Fraction(-11, 4))

    def test_one_pass_cube_refines_interval_unstable_row_to_active(self) -> None:
        toy = _scalar_relu_toy(
            lower=1,
            upper=2,
            pre_weight=Fraction(3, 4),
            pre_bias=Fraction(1, 4),
            out_weight=-2,
        )
        before = dict(toy.facts)
        # Deliberately retain a sound but loose interval fact.  The fixed
        # preliminary HybridZ stream proves the actual preactivation positive.
        before[3] = _fact([-1.0], [2.0])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )

        self.assertEqual(result.receipt.phase_refinement_passes, 1)
        self.assertEqual(result.receipt.interval_exact_rows, 1)
        self.assertEqual(result.receipt.refined_stable_rows, 1)
        self.assertEqual(result.receipt.exact_rows, 0)
        self.assertEqual(result.hz.n_bin, 0)
        self.assertEqual(result.hz.n_eq, 1)
        self.assertEqual(result.hz.n_ub, 2)

        # The sole generator stream keeps the provisional ReLU y factor and
        # the two rows constrain it to the active affine image.  Check the
        # stored binary64 polytope itself at three rational input factors;
        # this is not a sampled ONNX execution.
        for input_factor in (Fraction(-1), Fraction(0), Fraction(1)):
            local_factor = Fraction(3, 8) * input_factor + Fraction(3, 8)
            continuous = _extend_constraint_local_factors(
                result.hz, (input_factor, local_factor)
            )
            for row in range(result.hz.n_ub):
                lhs = sum(
                    (
                        _fraction(value) * continuous[column]
                        for column, value in zip(
                            result.hz.Auc.getrow(row).indices,
                            result.hz.Auc.getrow(row).data,
                        )
                    ),
                    Fraction(0),
                )
                self.assertLessEqual(lhs, _fraction(result.hz.ub[row]))
            output = _fraction(result.hz.c[0]) + sum(
                (
                    _fraction(value) * continuous[column]
                    for column, value in zip(
                        result.hz.Gc.getrow(0).indices,
                        result.hz.Gc.getrow(0).data,
                    )
                ),
                Fraction(0),
            )
            expected = -2 * (
                Fraction(11, 8) + Fraction(3, 8) * input_factor
            )
            self.assertEqual(output, expected)

    def test_one_pass_cube_refines_interval_unstable_row_to_inactive(self) -> None:
        toy = _scalar_relu_toy(
            lower=1,
            upper=2,
            pre_weight=Fraction(3, 4),
            pre_bias=-2,
            out_weight=-2,
        )
        before = dict(toy.facts)
        before[3] = _fact([-2.0], [1.0])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )

        self.assertEqual(result.receipt.phase_refinement_passes, 1)
        self.assertEqual(result.receipt.interval_exact_rows, 1)
        self.assertEqual(result.receipt.refined_stable_rows, 1)
        self.assertEqual(result.receipt.exact_rows, 0)
        self.assertEqual(result.hz.n_bin, 0)
        self.assertEqual(result.hz.n_ub, 1)
        self.assertEqual(result.hz.Auc.getrow(0).nnz, 1)
        local_column = int(result.hz.Auc.getrow(0).indices[0])
        coefficient = _fraction(result.hz.Auc.getrow(0).data[0])
        self.assertEqual(coefficient, Fraction(1, 2))
        self.assertEqual(_fraction(result.hz.ub[0]), Fraction(-1, 2))
        self.assertEqual(_fraction(result.hz.c[0]), Fraction(-1))
        self.assertEqual(_fraction(result.hz.Gc[0, local_column]), Fraction(-1))
        # The factor box gives eta >= -1; the single row gives eta <= -1.
        # Therefore y = 1/2 + (1/2) eta = 0 exactly.

    def test_point_box_has_no_spurious_factor_or_phase(self) -> None:
        toy = _scalar_relu_toy(lower=0, upper=0)
        before = dict(toy.facts)
        before[3] = _fact([0.0], [0.0])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )

        self.assertEqual(result.hz.n_cont, 0)
        self.assertEqual(result.hz.n_bin, 0)
        self.assertEqual(result.hz.n_ub, 0)
        np.testing.assert_array_equal(result.hz.c, np.zeros(1, dtype=np.float64))

    def test_selected_conv_emission_matches_full_operator_submatrix(self) -> None:
        weight = torch.tensor(
            [
                [[[1.0, 0.0], [-0.5, 2.0]], [[0.25, 1.0], [0.0, -1.0]]],
                [[[0.0, 0.75], [1.5, -0.25]], [[-2.0, 0.0], [0.5, 1.0]]],
            ],
            dtype=torch.float64,
        )
        layer = SimpleNamespace(
            id=90,
            kind="CONV2D",
            out_vars=list(range(8)),
            params={
                "weight": weight,
                "bias": torch.tensor([0.25, -0.5], dtype=torch.float64),
                "input_shape": (1, 2, 3, 3),
                "output_shape": (1, 2, 2, 2),
                "stride": (1, 1),
                "padding": (0, 0),
                "dilation": (1, 1),
                "groups": 1,
            },
        )
        snapshot = _candidate_module._affine_snapshot(layer, input_size=18)
        rows = np.asarray([0, 3, 4, 7], dtype=np.int64)
        possible = np.asarray(
            [True, False, True, True, False, True, True, True, False] * 2,
            dtype=bool,
        )
        actual = _candidate_module._selected_affine_matrix(
            snapshot, rows, possible, name="selected_conv_test"
        )
        full, _bias = exact_sparse_conv2d_matrix_from_layer_candidate(layer)
        expected = full[rows].multiply(possible[None, :]).tocsr()
        expected.eliminate_zeros()
        expected.sort_indices()
        self.assertEqual(actual.shape, expected.shape)
        np.testing.assert_array_equal(actual.indptr, expected.indptr)
        np.testing.assert_array_equal(actual.indices, expected.indices)
        np.testing.assert_array_equal(
            actual.data.view(np.uint64), expected.data.view(np.uint64)
        )

    def test_conv_residual_graph_points_satisfy_every_exact_phase_row(self) -> None:
        toy = _toy_audit.OperatorAddFusionAuditTests._projection_skip_chain_toy()
        before = dict(toy.facts)
        before[3] = _fact([-1.0], [1.0])
        before[7] = _fact([-1.0], [2.0])
        before[12] = _fact([-11.0 / 16.0], [37.0 / 16.0])
        result = build_forward_exact_relu_live_row_stream_candidate(
            toy.net, before, toy.facts
        )
        hz = result.hz
        exact = _toy_audit._exact_graph_range(toy)
        self.assertEqual((exact.lower, exact.upper), (Fraction(0), Fraction(37, 16)))
        self.assertEqual(result.receipt.exact_rows, 3)
        self.assertEqual(hz.n_bin, 3)
        self.assertEqual(hz.n_eq, 3)
        self.assertEqual(hz.n_ub, 9)

        for x in (Fraction(-1), Fraction(0), Fraction(1)):
            y3 = max(Fraction(0), x)
            pre7 = y3 + x
            y7 = max(Fraction(0), pre7)
            pre12 = (
                Fraction(3, 4) * y7
                + Fraction(1, 2) * (y3 + x + Fraction(1, 8))
                - Fraction(1, 4)
            )
            y12 = max(Fraction(0), pre12)
            continuous = _extend_constraint_local_factors(hz, (
                x,
                2 * y3 - 1,
                y7 - 1,
                2 * y12 / Fraction(37, 16) - 1,
            ))
            binary = (
                1 if x >= 0 else -1,
                1 if pre7 >= 0 else -1,
                1 if pre12 >= 0 else -1,
            )
            for row in range(hz.n_ub):
                lhs = sum(
                    (
                        _fraction(value) * continuous[column]
                        for column, value in zip(
                            hz.Auc.getrow(row).indices,
                            hz.Auc.getrow(row).data,
                        )
                    ),
                    Fraction(0),
                )
                lhs += sum(
                    (
                        _fraction(value) * binary[column]
                        for column, value in zip(
                            hz.Aub.getrow(row).indices,
                            hz.Aub.getrow(row).data,
                        )
                    ),
                    Fraction(0),
                )
                self.assertLessEqual(lhs, _fraction(hz.ub[row]))
            output = _fraction(hz.c[0]) + sum(
                (
                    _fraction(value) * continuous[column]
                    for column, value in zip(
                        hz.Gc.getrow(0).indices,
                        hz.Gc.getrow(0).data,
                    )
                ),
                Fraction(0),
            )
            self.assertEqual(output, y12)


if __name__ == "__main__":
    unittest.main()
