#!/usr/bin/env python3
"""Controlled audits for property-targeted ReLU residual normal form.

No interval-analysis fact is used as an oracle.  Scalar graph extrema are
enumerated exactly with :class:`fractions.Fraction`; SciPy/HiGHS is used only
to measure relaxation tightness and explicit witness feasibility.

Run from the repository root with::

    python -m act.back_end.hybridz_tf.test_operator_residual_normal_form
"""

from __future__ import annotations

from fractions import Fraction
import itertools
import random
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuildError,
    _OperatorHZBuilder,
    _relu_triangle_parameters,
    build_operator_hz,
)
from act.back_end.hybridz_tf.test_operator_add_fusion import (
    _assemble_scalar_toy,
    _assemble_width_toy,
    _dense,
    _dense_matrix,
    _exact_graph_range,
    _input_layers,
    _layer,
    _lp_output_range,
    _output_cube_fraction,
    _wide_layer,
)


def _scalar_relu_toy(
    *,
    lower: Fraction | float | int = -1,
    upper: Fraction | float | int = 1,
    pre_weight: Fraction | float | int = 1,
    pre_bias: Fraction | float | int = 0,
    out_weight: Fraction | float | int = 1,
):
    input_layer, spec = _input_layers(lower, upper)
    layers = [
        input_layer,
        spec,
        _dense(2, pre_weight, pre_bias),
        _layer(3, "RELU"),
        _dense(4, out_weight, 0),
        _layer(5, "ASSERT"),
    ]
    return _assemble_scalar_toy(
        layers,
        {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
        input_lb=lower,
        input_ub=upper,
    )


def _build(toy, targets=None, *, exact_budget: int = 0):
    return build_operator_hz(
        toy.net,
        toy.facts,
        toy.facts,
        exact_budget=exact_budget,
        materialize_add=True,
        residual_targets=targets,
    )


def _csr_exact(left: sp.csr_matrix, right: sp.csr_matrix) -> bool:
    left = left.tocsr()
    right = right.tocsr()
    return (
        left.shape == right.shape
        and np.array_equal(left.indptr, right.indptr)
        and np.array_equal(left.indices, right.indices)
        and np.array_equal(left.data, right.data)
    )


def _enumerated_binary_output_upper(build) -> float:
    """Measure one-output HZ support by enumerating toy binary factors."""

    hz = build.hz
    if hz.n_out != 1:
        raise AssertionError("the binary toy support helper requires one output")
    gc = np.asarray(hz.Gc.toarray(), dtype=np.float64)[0]
    gb = np.asarray(hz.Gb.toarray(), dtype=np.float64)[0]
    best = -np.inf
    for assignment in itertools.product(
        (-1.0, 1.0), repeat=int(hz.Gb.shape[1])
    ):
        binary = np.asarray(assignment, dtype=np.float64)
        result = linprog(
            -gc,
            A_ub=hz.Auc,
            b_ub=hz.ub - np.asarray(hz.Aub @ binary).reshape(-1),
            A_eq=hz.Ac,
            b_eq=hz.b - np.asarray(hz.Ab @ binary).reshape(-1),
            bounds=[(-1.0, 1.0)] * int(gc.size),
            method="highs",
        )
        if result.success:
            best = max(
                best,
                float(hz.c[0] + gb @ binary + gc @ result.x),
            )
    if not np.isfinite(best):
        raise AssertionError("the binary toy HZ was unexpectedly infeasible")
    return best


class OperatorResidualNormalFormTests(unittest.TestCase):
    def test_four_guards_are_sound_and_have_expected_rows(self) -> None:
        toy = _scalar_relu_toy()
        exact = _exact_graph_range(toy)
        expected_rows = {"none": 0, "zero": 1, "identity": 1, "both": 2}
        measured = {}
        for guard, rows in expected_rows.items():
            build = _build(toy, [(3, 0, guard)])
            cube_lower, cube_upper = _output_cube_fraction(build)
            self.assertLessEqual(cube_lower, exact.lower)
            self.assertGreaterEqual(cube_upper, exact.upper)
            lp_lower, lp_upper = _lp_output_range(build)
            self.assertLessEqual(lp_lower, float(exact.lower) + 1e-12)
            self.assertGreaterEqual(lp_upper, float(exact.upper) - 1e-12)
            tags = {
                item["tag"]: item["rows"]
                for item in build.metadata["constraint_tags_ub"]
                if item["tag"].startswith("relu_residual_")
            }
            self.assertEqual(sum(tags.values()), rows)
            layer_meta = next(
                item for item in build.metadata["layers"]
                if item["layer_id"] == 3
            )
            self.assertEqual(layer_meta["relu_residual_rows"], 1)
            self.assertEqual(
                layer_meta["relu_residual_expected_rows_saved"],
                2 - rows,
            )
            measured[guard] = (lp_lower, lp_upper)

        # l=-1,u=1,a=1/2,b=1/2.  Without the zero guard the strip admits
        # (x=-1,rho=0,y=-1/2); the identity guard does not remove it.
        self.assertLess(measured["none"][0], -0.49)
        self.assertLess(measured["identity"][0], -0.49)
        self.assertGreaterEqual(measured["zero"][0], -1e-10)
        self.assertGreaterEqual(measured["both"][0], -1e-10)
        self.assertLessEqual(measured["both"][0], measured["zero"][0] + 1e-9)

    def test_fraction_endpoint_envelope_and_subnormal_rho_box(self) -> None:
        for lower, upper in ((-10.0, 0.01), (-10.0, 0.2)):
            slope, intercept, _ = _relu_triangle_parameters(
                np.asarray([lower]), np.asarray([upper])
            )
            lf = Fraction.from_float(lower)
            uf = Fraction.from_float(upper)
            sf = Fraction.from_float(float(slope[0]))
            required = max(
                Fraction(0),
                -sf * lf,
                (Fraction(1) - sf) * uf,
            )
            self.assertGreaterEqual(
                Fraction.from_float(float(intercept[0])),
                required,
            )

        tiny = float(np.nextafter(0.0, np.inf))
        slope, intercept, _ = _relu_triangle_parameters(
            np.asarray([-tiny]), np.asarray([tiny])
        )
        self.assertEqual(float(slope[0]), 0.5)
        self.assertGreaterEqual(float(intercept[0]), tiny)

        builder = _OperatorHZBuilder(
            SimpleNamespace(),
            {},
            {},
            exact_budget=0,
            materialize_add=True,
            preactivation_lp_budget=0,
            preactivation_lp_time_limit=0.0,
            residual_targets=None,
            deadline=None,
        )
        rho, count = builder._box_expr(
            np.asarray([0.0]),
            np.asarray([tiny]),
        )
        self.assertEqual(count, 1)
        self.assertEqual(rho.G.nnz, 1)
        cube_lower, cube_upper = builder._cube_bounds(rho)
        self.assertLessEqual(float(cube_lower[0]), 0.0)
        self.assertGreaterEqual(float(cube_upper[0]), tiny)

    def test_mixed_phase_exact_triangle_and_residual_partition(self) -> None:
        input_layer, spec = _input_layers(-1, 1)
        pre = _dense_matrix(
            2,
            [[1], [1], [1], [2], [-1]],
            [2, -2, 0, 0, 0.1],
        )
        relu = _wide_layer(3, "RELU", 5)
        out = _dense_matrix(4, [[1, 1, 1, 1, 1]], [0])
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, out, _layer(5, "ASSERT")],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            input_lb=-1,
            input_ub=1,
        )
        build = _build(
            toy,
            [
                (3, 0, "none"),
                (3, 1, "zero"),
                (3, 2, "both"),
                (3, 3, "identity"),
            ],
            exact_budget=1,
        )
        meta = next(
            item for item in build.metadata["layers"]
            if item["layer_id"] == 3
        )
        self.assertEqual(meta["relu_active"], 1)
        self.assertEqual(meta["relu_inactive"], 1)
        self.assertEqual(meta["relu_exact"], 1)
        self.assertEqual(meta["relu_residual_rows"], 1)
        self.assertEqual(meta["relu_triangle_rows"], 1)
        statuses = {
            (item["row"], item["guard"]): item["status"]
            for item in build.metadata["residual_target_receipts"]
        }
        self.assertEqual(statuses[(0, "none")], "skipped_active")
        self.assertEqual(statuses[(1, "zero")], "skipped_inactive")
        self.assertEqual(statuses[(2, "both")], "skipped_exact")
        self.assertEqual(statuses[(3, "identity")], "applied")

    def test_positive_exact_budget_uses_property_target_not_prefix(self) -> None:
        input_layer, spec = _input_layers(-1, 1)
        pre = _dense_matrix(
            2,
            [[1], [1]],
            [0, 0],
        )
        relu = _wide_layer(3, "RELU", 2)
        out = _dense_matrix(4, [[0, 1]], [0])
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, out, _layer(5, "ASSERT")],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            input_lb=-1,
            input_ub=1,
        )
        build = _build(
            toy,
            [(3, 1, "both")],
            exact_budget=1,
        )
        meta = next(
            item for item in build.metadata["layers"]
            if item["layer_id"] == 3
        )
        self.assertEqual(meta["exact_index_preview"], [1])
        self.assertEqual(
            meta["relu_exact_selection"],
            "property_gap_adjoint_facility_targets_v1",
        )
        self.assertEqual(
            build.metadata["exact_selection"],
            "property_gap_adjoint_facility_targets_v1",
        )
        self.assertEqual(
            build.metadata["residual_target_receipts"][0]["status"],
            "skipped_exact",
        )
        lower, upper = _output_cube_fraction(build)
        self.assertLessEqual(lower, Fraction(0))
        self.assertGreaterEqual(upper, Fraction(1))

    def test_property_gap_target_closes_prefix_exact_relaxation_gap(self) -> None:
        # The three preactivations share one scalar input.  Their exact
        # weighted ReLU sum has maximum zero, but spending the sole binary on
        # topological row 0 leaves a 2.5 relaxation gap.  The property score
        # |adjoint| * (-l*u/(u-l)) is largest at row 2; exactifying that row
        # closes the gap while row 0 remains relaxed.
        input_layer, spec = _input_layers(-1, 1)
        pre = _dense_matrix(
            2,
            [[-3], [3], [-3]],
            [1, -1, -0.5],
        )
        relu = _wide_layer(3, "RELU", 3)
        out = _dense_matrix(4, [[-2, -3, 3]], [-0.25])
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, out, _layer(5, "ASSERT")],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3], 5: [4]},
            input_lb=-1,
            input_ub=1,
        )
        prefix = _build(toy, None, exact_budget=1)
        guided = _build(
            toy,
            [(3, 2, "both")],
            exact_budget=1,
        )
        prefix_meta = next(
            item for item in prefix.metadata["layers"]
            if item["layer_id"] == 3
        )
        guided_meta = next(
            item for item in guided.metadata["layers"]
            if item["layer_id"] == 3
        )
        self.assertEqual(prefix_meta["exact_index_preview"], [0])
        self.assertEqual(guided_meta["exact_index_preview"], [2])

        breakpoints = [
            Fraction(-1),
            Fraction(1),
            Fraction(1, 3),
            Fraction(-1, 3),
            Fraction(-1, 6),
        ]
        exact_values = []
        for x in breakpoints:
            values = (
                max(Fraction(0), -3 * x + 1),
                max(Fraction(0), 3 * x - 1),
                max(Fraction(0), -3 * x - Fraction(1, 2)),
            )
            exact_values.append(
                -2 * values[0]
                - 3 * values[1]
                + 3 * values[2]
                - Fraction(1, 4)
            )
        exact_upper = max(exact_values)
        self.assertEqual(exact_upper, Fraction(-1, 4))

        prefix_upper = _enumerated_binary_output_upper(prefix)
        guided_upper = _enumerated_binary_output_upper(guided)
        self.assertGreater(prefix_upper, 2.0)
        self.assertLess(guided_upper, -0.24)
        self.assertGreaterEqual(
            Fraction.from_float(guided_upper), exact_upper
        )
        self.assertGreater(prefix_upper - guided_upper, 2.49)

    def test_one_node_fanout_shares_rho_and_cancels(self) -> None:
        input_layer, spec = _input_layers(-1, 1)
        layers = [
            input_layer,
            spec,
            _dense(2, 1, 0),
            _layer(3, "RELU"),
            _dense(4, 1, 0),
            _dense(5, -1, 0),
            _layer(6, "ADD"),
            _layer(7, "ASSERT"),
        ]
        toy = _assemble_scalar_toy(
            layers,
            {
                0: [],
                1: [0],
                2: [1],
                3: [2],
                4: [3],
                5: [3],
                6: [4, 5],
                7: [6],
            },
            input_lb=-1,
            input_ub=1,
        )
        build = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=False,
            residual_targets=[(3, 0, "none")],
        )
        receipts = build.metadata["residual_target_receipts"]
        self.assertEqual(len(receipts), 1)
        residual_column = receipts[0]["factor_column"]
        self.assertEqual(build.hz.Gc[:, residual_column].nnz, 0)
        lower, upper = _output_cube_fraction(build)
        exact = _exact_graph_range(toy)
        self.assertLessEqual(lower, exact.lower)
        self.assertGreaterEqual(upper, exact.upper)
        self.assertLess(float(upper - lower), 1e-12)

    def test_distinct_rows_get_fresh_factors_and_keep_joint_witness(self) -> None:
        input_layer = _wide_layer(0, "INPUT", 2)
        input_layer.params = {"shape": (1, 2)}
        spec = _wide_layer(1, "INPUT_SPEC", 2)
        spec.params = {
            "kind": "BOX",
            "lb": torch.tensor([[-1.0, -1.0]], dtype=torch.float64),
            "ub": torch.tensor([[1.0, 1.0]], dtype=torch.float64),
        }
        pre = _dense_matrix(2, [[1, 0], [0, 1]], [0, 0])
        relu = _wide_layer(3, "RELU", 2)
        assertion = _wide_layer(4, "ASSERT", 2)
        toy = _assemble_width_toy(
            [input_layer, spec, pre, relu, assertion],
            {0: [], 1: [0], 2: [1], 3: [2], 4: [3]},
            input_lb=-1,
            input_ub=1,
        )
        # Restore the exact two-dimensional input facts overwritten by the
        # scalar convenience assembler.
        toy.facts[0].bounds.lb[:] = spec.params["lb"]
        toy.facts[0].bounds.ub[:] = spec.params["ub"]
        toy.facts[1].bounds.lb[:] = spec.params["lb"]
        toy.facts[1].bounds.ub[:] = spec.params["ub"]
        build = _build(
            toy,
            [(3, 0, "both"), (3, 1, "both")],
        )
        receipts = build.metadata["residual_target_receipts"]
        columns = [item["factor_column"] for item in receipts]
        self.assertEqual(len(set(columns)), 2)

        # The true graph point ReLU([0,1])=[0,1] must remain feasible.  If the
        # two residual nodes reused one factor, their required rho values
        # (0 and 1/2) would conflict.
        hz = build.hz
        Aeq = sp.vstack(
            [hz.Ac, hz.Gc],
            format="csr",
        )
        beq = np.concatenate([hz.b, np.asarray([0.0, 1.0]) - hz.c])
        result = linprog(
            np.zeros(hz.n_cont),
            A_ub=hz.Auc if hz.n_ub else None,
            b_ub=hz.ub if hz.n_ub else None,
            A_eq=Aeq,
            b_eq=beq,
            bounds=[(-1.0, 1.0)] * hz.n_cont,
            method="highs",
        )
        self.assertTrue(result.success, result.message)

    def test_invalid_targets_fail_before_build_or_are_phase_skipped(self) -> None:
        toy = _scalar_relu_toy()
        invalid = (
            [(3, 0, "bad")],
            [(3, 0, "none"), (3, 0, "both")],
            [(99, 0, "none")],
            [(2, 0, "none")],
            [(3, 1, "none")],
            [(-1, 0, "none")],
        )
        for targets in invalid:
            with self.subTest(targets=targets):
                with self.assertRaises(OperatorHZBuildError):
                    _build(toy, targets)

        stable = _scalar_relu_toy(lower=1, upper=2)
        build = _build(stable, [(3, 0, "none")])
        self.assertEqual(
            build.metadata["residual_target_receipts"][0]["status"],
            "skipped_active",
        )

    def test_default_disabled_is_exact_csr_fingerprint(self) -> None:
        toy = _scalar_relu_toy(
            lower=Fraction(-7, 5),
            upper=Fraction(9, 7),
            pre_weight=Fraction(5, 3),
            pre_bias=Fraction(-1, 11),
            out_weight=Fraction(-4, 7),
        )
        legacy = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
        )
        explicit_none = build_operator_hz(
            toy.net,
            toy.facts,
            toy.facts,
            exact_budget=0,
            materialize_add=True,
            residual_targets=None,
        )
        self.assertTrue(np.array_equal(legacy.hz.c, explicit_none.hz.c))
        for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
            self.assertTrue(
                _csr_exact(getattr(legacy.hz, name), getattr(explicit_none.hz, name)),
                name,
            )
        for name in ("b", "ub"):
            self.assertTrue(
                np.array_equal(
                    getattr(legacy.hz, name),
                    getattr(explicit_none.hz, name),
                ),
                name,
            )

    def test_32_rational_operator_dags_all_guards(self) -> None:
        rng = random.Random(90473)
        for case in range(32):
            weight = Fraction(
                rng.choice((-5, -3, -2, 2, 3, 5)),
                rng.choice((1, 2, 3)),
            )
            radius = abs(weight)
            bias = Fraction(
                rng.randint(-3, 3),
                rng.choice((4, 5, 7)),
            )
            if not (-radius < bias < radius):
                bias = Fraction(0)
            out_weight = Fraction(
                rng.choice((-4, -2, -1, 1, 2, 4)),
                rng.choice((1, 3, 5)),
            )
            toy = _scalar_relu_toy(
                pre_weight=weight,
                pre_bias=bias,
                out_weight=out_weight,
            )
            exact = _exact_graph_range(toy)
            measured = {}
            for guard in ("none", "zero", "identity", "both"):
                build = _build(toy, [(3, 0, guard)])
                lower, upper = _output_cube_fraction(build)
                self.assertLessEqual(lower, exact.lower, (case, guard))
                self.assertGreaterEqual(upper, exact.upper, (case, guard))
                lp_lower, lp_upper = _lp_output_range(build)
                self.assertLessEqual(
                    lp_lower, float(exact.lower) + 1e-9, (case, guard)
                )
                self.assertGreaterEqual(
                    lp_upper, float(exact.upper) - 1e-9, (case, guard)
                )
                measured[guard] = (lp_lower, lp_upper)
            self.assertGreaterEqual(
                measured["both"][0],
                measured["none"][0] - 1e-8,
            )
            self.assertLessEqual(
                measured["both"][1],
                measured["none"][1] + 1e-8,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
