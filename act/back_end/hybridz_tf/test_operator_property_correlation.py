#!/usr/bin/env python3
"""Toy-first audits for property-conditioned ADD correlation shadows.

The controlled graph materializes the two correlated coordinates ``(x,-x)``
at an ADD.  The following affine row sums them.  Treating the two normalized
ADD coordinates as an independent cube yields ``[-2,2]``; recomposing only
the selected row over the pre-ADD generators recovers the cancellation.

No propagated interval fact is used as proof authority.  Exact graph values
are checked with :class:`fractions.Fraction`, and the HZ projection is queried
only to measure the relaxation gap.
"""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
import unittest

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuildError,
    build_operator_hz,
)


def _layer(layer_id: int, kind: str, width: int, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        out_vars=list(range(int(width))),
        in_vars=[],
    )


def _correlated_add_toy(
    lower=-1.0,
    upper=1.0,
    *,
    branch=(1.0, -1.0),
    readout=(1.0, 1.0),
    bias=0.25,
):
    dtype = torch.float64
    layers = [
        _layer(0, "INPUT", 1, {"shape": (1, 1)}),
        _layer(
            1,
            "INPUT_SPEC",
            1,
            {
                "kind": "BOX",
                "lb": torch.tensor([[lower]], dtype=dtype),
                "ub": torch.tensor([[upper]], dtype=dtype),
            },
        ),
        _layer(
            2,
            "DENSE",
            2,
            {
                "weight": torch.tensor(
                    [[branch[0]], [branch[1]]], dtype=dtype
                ),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
        ),
        _layer(
            3,
            "DENSE",
            2,
            {
                "weight": torch.zeros((2, 1), dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
        ),
        _layer(4, "ADD", 2),
        _layer(
            5,
            "DENSE",
            1,
            {
                "weight": torch.tensor([list(readout)], dtype=dtype),
                "bias": torch.tensor([bias], dtype=dtype),
                "in_features": 2,
                "out_features": 1,
            },
        ),
        _layer(6, "RELU", 1),
        _layer(
            7,
            "DENSE",
            1,
            {
                "weight": torch.zeros((1, 2), dtype=dtype),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 2,
                "out_features": 1,
            },
        ),
        _layer(8, "ADD", 1),
        _layer(9, "ASSERT", 1, {"kind": "UNSAFE_LINEAR"}),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [2, 3],
        5: [4],
        6: [5],
        7: [4],
        8: [6, 7],
        9: [8],
    }
    succs = {layer.id: [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={layer.id: layer for layer in layers},
    )
    facts = {}
    widths = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 1,
        6: 1,
        7: 1,
        8: 1,
        9: 1,
    }
    for layer in layers:
        width = widths[layer.id]
        if layer.id in {0, 1}:
            lb = torch.tensor([[lower]], dtype=dtype)
            ub = torch.tensor([[upper]], dtype=dtype)
        else:
            lb = torch.full((1, width), -100.0, dtype=dtype)
            ub = torch.full((1, width), 100.0, dtype=dtype)
        facts[layer.id] = Fact(Bounds(lb, ub), ConSet())
    return net, facts


def _two_block_skip_toy():
    """Two residual depths; only recursive skip provenance cancels ``x,-x``."""

    dtype = torch.float64
    layers = [
        _layer(0, "INPUT", 1, {"shape": (1, 1)}),
        _layer(
            1,
            "INPUT_SPEC",
            1,
            {
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=dtype),
                "ub": torch.tensor([[1.0]], dtype=dtype),
            },
        ),
        _layer(
            2,
            "DENSE",
            2,
            {
                "weight": torch.tensor([[1.0], [-1.0]], dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
        ),
        _layer(
            3,
            "DENSE",
            2,
            {
                "weight": torch.zeros((2, 1), dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 1,
                "out_features": 2,
            },
        ),
        _layer(4, "ADD", 2),
        _layer(
            5,
            "DENSE",
            2,
            {
                "weight": torch.zeros((2, 2), dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 2,
                "out_features": 2,
            },
        ),
        _layer(6, "RELU", 2),
        _layer(
            7,
            "DENSE",
            2,
            {
                "weight": torch.eye(2, dtype=dtype),
                "bias": torch.zeros(2, dtype=dtype),
                "in_features": 2,
                "out_features": 2,
            },
        ),
        _layer(8, "ADD", 2),
        _layer(
            9,
            "DENSE",
            1,
            {
                "weight": torch.ones((1, 2), dtype=dtype),
                "bias": torch.tensor([0.25], dtype=dtype),
                "in_features": 2,
                "out_features": 1,
            },
        ),
        _layer(10, "RELU", 1),
        _layer(11, "ASSERT", 1, {"kind": "UNSAFE_LINEAR"}),
    ]
    preds = {
        0: [],
        1: [0],
        2: [1],
        3: [1],
        4: [2, 3],
        5: [4],
        6: [5],
        7: [6],
        8: [7, 4],
        9: [8],
        10: [9],
        11: [10],
    }
    succs = {layer.id: [] for layer in layers}
    for child, parents in preds.items():
        for parent in parents:
            succs[parent].append(child)
    net = SimpleNamespace(
        layers=layers,
        preds=preds,
        succs=succs,
        by_id={layer.id: layer for layer in layers},
    )
    widths = {
        0: 1,
        1: 1,
        2: 2,
        3: 2,
        4: 2,
        5: 2,
        6: 2,
        7: 2,
        8: 2,
        9: 1,
        10: 1,
        11: 1,
    }
    facts = {}
    for layer in layers:
        width = widths[layer.id]
        if layer.id in {0, 1}:
            lb = torch.tensor([[-1.0]], dtype=dtype)
            ub = torch.tensor([[1.0]], dtype=dtype)
        else:
            lb = torch.full((1, width), -100.0, dtype=dtype)
            ub = torch.full((1, width), 100.0, dtype=dtype)
        facts[layer.id] = Fact(Bounds(lb, ub), ConSet())
    return net, facts


def _continuous_output_upper(build) -> float:
    hz = build.hz
    objective = np.concatenate(
        (
            hz.Gc.getrow(0).toarray().reshape(-1),
            hz.Gb.getrow(0).toarray().reshape(-1),
        )
    )
    A_ub = sp.hstack((hz.Auc, hz.Aub), format="csr")
    A_eq = sp.hstack((hz.Ac, hz.Ab), format="csr")
    result = linprog(
        -objective,
        A_ub=A_ub if A_ub.shape[0] else None,
        b_ub=hz.ub if A_ub.shape[0] else None,
        A_eq=A_eq if A_eq.shape[0] else None,
        b_eq=hz.b if A_eq.shape[0] else None,
        bounds=(-1.0, 1.0),
        method="highs",
    )
    if not result.success:
        raise AssertionError(result.message)
    return float(hz.c[0] - result.fun)


def _stored_output_cube_upper(build) -> Fraction:
    """Exact support of the stored output row over its factor cube."""

    hz = build.hz
    result = Fraction.from_float(float(hz.c[0]))
    for value in hz.Gc.getrow(0).data:
        result += abs(Fraction.from_float(float(value)))
    for value in hz.Gb.getrow(0).data:
        result += abs(Fraction.from_float(float(value)))
    return result


def _relu_metadata(build):
    return next(
        item
        for item in build.metadata["layers"]
        if item["layer_id"] == 6
    )


class PropertyCorrelationShadowTests(unittest.TestCase):
    def test_residual_bound_screen_tightens_ambiguous_triangle(self):
        net, facts = _correlated_add_toy(
            -1.0,
            1.0,
            branch=(1.0, -1.0),
            readout=(1.0, 0.5),
            bias=0.0,
        )
        phase_only = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            residual_phase_screen=True,
        )
        bounded = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            residual_bound_screen=True,
        )
        phase_meta = _relu_metadata(phase_only)
        bound_meta = _relu_metadata(bounded)
        self.assertEqual(
            phase_meta["residual_phase_screen"]["status"],
            "no_stable_shadow_rows",
        )
        receipt = bound_meta["residual_phase_screen"]
        self.assertEqual(receipt["status"], "applied")
        self.assertEqual(receipt["rows_tightened"], 1)
        self.assertEqual(bound_meta["relu_relaxed"], 1)
        self.assertLess(
            bound_meta["preactivation_phase_screen_ub_max"],
            0.500000001,
        )
        self.assertGreater(
            bound_meta["preactivation_phase_screen_lb_min"],
            -0.500000001,
        )
        self.assertGreater(
            phase_meta["preactivation_phase_screen_ub_max"], 1.49
        )
        self.assertGreaterEqual(
            _stored_output_cube_upper(bounded), Fraction(1, 2)
        )

    def test_phase_screen_carries_shadow_across_skip_fanout(self):
        net, facts = _two_block_skip_toy()
        baseline = build_operator_hz(
            net, facts, facts, materialize_add=True
        )
        screened = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            residual_phase_screen=True,
        )
        baseline_relu = next(
            item
            for item in baseline.metadata["layers"]
            if item["layer_id"] == 10
        )
        screened_relu = next(
            item
            for item in screened.metadata["layers"]
            if item["layer_id"] == 10
        )
        second_add = next(
            item
            for item in screened.metadata["layers"]
            if item["layer_id"] == 8
        )
        self.assertEqual(baseline_relu["relu_relaxed"], 1)
        self.assertEqual(screened_relu["relu_active"], 1)
        self.assertEqual(
            screened_relu["residual_phase_screen"]["stabilized_active"], 1
        )
        self.assertTrue(second_add["residual_skip_shadow_recursive"])
        self.assertGreaterEqual(
            _stored_output_cube_upper(screened), Fraction(1, 4)
        )
        self.assertLess(_continuous_output_upper(screened), 0.250000001)

    def test_residual_phase_screen_commits_only_proven_phase(self):
        net, facts = _correlated_add_toy()
        stable = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            residual_phase_screen=True,
        )
        stable_meta = _relu_metadata(stable)
        receipt = stable_meta["residual_phase_screen"]
        self.assertEqual(receipt["status"], "applied")
        self.assertTrue(receipt["proof_authority"])
        self.assertEqual(receipt["stabilized_active"], 1)
        self.assertEqual(stable_meta["relu_active"], 1)
        self.assertEqual(
            stable.metadata["residual_phase_screen_stabilized_active"], 1
        )
        self.assertGreaterEqual(
            _stored_output_cube_upper(stable), Fraction(1, 4)
        )

        crossing_net, crossing_facts = _correlated_add_toy(
            -1.0,
            1.0,
            branch=(1.0, 0.0),
            readout=(1.0, 0.0),
            bias=0.0,
        )
        crossing = build_operator_hz(
            crossing_net,
            crossing_facts,
            crossing_facts,
            materialize_add=True,
            residual_phase_screen=True,
        )
        crossing_meta = _relu_metadata(crossing)
        self.assertEqual(
            crossing_meta["residual_phase_screen"]["status"],
            "no_stable_shadow_rows",
        )
        self.assertEqual(crossing_meta["relu_relaxed"], 1)
        self.assertEqual(
            crossing.metadata["residual_phase_screen_stabilized_active"], 0
        )
        self.assertEqual(
            crossing.metadata["residual_phase_screen_stabilized_inactive"], 0
        )

    def test_selected_row_recovers_cross_coordinate_cancellation(self):
        net, facts = _correlated_add_toy()
        baseline = build_operator_hz(
            net, facts, facts, materialize_add=True
        )
        tightened = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            correlation_targets={6: [0]},
        )
        ordinary = _relu_metadata(baseline)
        selected = _relu_metadata(tightened)

        self.assertLess(ordinary["preactivation_cube_lb_min"], -1.7)
        self.assertGreater(ordinary["preactivation_cube_ub_max"], 2.2)
        receipt = selected["property_correlation_shadow"]
        self.assertEqual(receipt["status"], "applied")
        self.assertTrue(receipt["proof_authority"])
        self.assertEqual(receipt["rows_tightened"], 1)
        self.assertEqual(receipt["stabilized_active"], 1)
        self.assertEqual(receipt["shadow_generator_nnz"], 0)
        self.assertGreater(
            selected["preactivation_correlation_lb_min"], 0.249999999
        )
        self.assertLess(
            selected["preactivation_correlation_ub_max"], 0.250000001
        )
        self.assertEqual(selected["relu_active"], 1)
        self.assertEqual(selected["relu_relaxed"], 0)

        # The exact stored-float graph is ReLU(x + (-x) + 1/4) = 1/4.
        exact = Fraction(1, 4)
        self.assertGreaterEqual(_stored_output_cube_upper(tightened), exact)
        self.assertLess(_continuous_output_upper(tightened), 0.250000001)
        self.assertGreater(_continuous_output_upper(baseline), 0.25)

    def test_target_budget_is_row_local_and_monotone(self):
        net, facts = _correlated_add_toy()
        off = build_operator_hz(
            net, facts, facts, materialize_add=True, correlation_targets={}
        )
        on = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            correlation_targets=[(6, 0), (6, 0)],
        )
        self.assertLessEqual(
            _continuous_output_upper(on), _continuous_output_upper(off)
        )
        self.assertEqual(on.metadata["correlation_target_count"], 1)
        self.assertEqual(on.metadata["correlation_shadow_rows_prepared"], 1)
        self.assertEqual(on.metadata["correlation_shadow_rows_tightened"], 1)

    def test_point_box_consistency(self):
        point = Fraction(3, 8)
        net, facts = _correlated_add_toy(float(point), float(point))
        build = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            correlation_targets={6: 0},
        )
        meta = _relu_metadata(build)
        self.assertEqual(meta["relu_active"], 1)
        self.assertLess(
            meta["preactivation_correlation_ub_max"]
            - meta["preactivation_correlation_lb_min"],
            1.0e-12,
        )
        self.assertGreaterEqual(
            _stored_output_cube_upper(build), Fraction(1, 4)
        )

    def test_fraction_oracle_random_dyadic_rows(self):
        values = (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5)
        rng = np.random.default_rng(29072026)
        for _case in range(96):
            a, b, p, q = (
                values[int(index)]
                for index in rng.integers(0, len(values), size=4)
            )
            bias = values[int(rng.integers(0, len(values)))] / 4.0
            lower = values[int(rng.integers(0, len(values)))]
            upper = values[int(rng.integers(0, len(values)))]
            lower, upper = min(lower, upper), max(lower, upper)
            net, facts = _correlated_add_toy(
                lower,
                upper,
                branch=(a, b),
                readout=(p, q),
                bias=bias,
            )
            baseline = build_operator_hz(
                net, facts, facts, materialize_add=True
            )
            tightened = build_operator_hz(
                net,
                facts,
                facts,
                materialize_add=True,
                correlation_targets={6: [0]},
            )
            ordinary = _relu_metadata(baseline)
            selected = _relu_metadata(tightened)

            aq, bq = Fraction.from_float(a), Fraction.from_float(b)
            pq, qq = Fraction.from_float(p), Fraction.from_float(q)
            slope = pq * aq + qq * bq
            intercept = Fraction.from_float(bias)
            endpoints = (
                slope * Fraction.from_float(lower) + intercept,
                slope * Fraction.from_float(upper) + intercept,
            )
            exact_lower, exact_upper = min(endpoints), max(endpoints)
            stored_lower = Fraction.from_float(
                selected["preactivation_correlation_lb_min"]
            )
            stored_upper = Fraction.from_float(
                selected["preactivation_correlation_ub_max"]
            )
            self.assertLessEqual(stored_lower, exact_lower)
            self.assertGreaterEqual(stored_upper, exact_upper)
            self.assertLessEqual(
                selected["preactivation_correlation_ub_max"]
                - selected["preactivation_correlation_lb_min"],
                ordinary["preactivation_cube_ub_max"]
                - ordinary["preactivation_cube_lb_min"]
                + 1.0e-12,
            )

    def test_affine_jacobian_survives_shared_fanout(self):
        # d/dx [3*(2x) + 4*(-x) + 3] = 2 exactly.
        net, facts = _correlated_add_toy(
            -1.0,
            1.0,
            branch=(2.0, -1.0),
            readout=(3.0, 4.0),
            bias=3.0,
        )
        build = build_operator_hz(
            net,
            facts,
            facts,
            materialize_add=True,
            correlation_targets={6: [0]},
        )
        meta = _relu_metadata(build)
        correlation_width = (
            meta["preactivation_correlation_ub_max"]
            - meta["preactivation_correlation_lb_min"]
        )
        self.assertAlmostEqual(correlation_width, 4.0, places=11)
        self.assertGreater(
            meta["preactivation_cube_ub_max"]
            - meta["preactivation_cube_lb_min"],
            19.9,
        )
        self.assertEqual(meta["relu_active"], 1)

    def test_bad_target_fails_before_build(self):
        net, facts = _correlated_add_toy()
        with self.assertRaisesRegex(
            OperatorHZBuildError, "exceed width"
        ):
            build_operator_hz(
                net,
                facts,
                facts,
                correlation_targets={6: [1]},
            )
        with self.assertRaisesRegex(
            OperatorHZBuildError, "not RELU"
        ):
            build_operator_hz(
                net,
                facts,
                facts,
                correlation_targets={5: [0]},
            )


if __name__ == "__main__":
    unittest.main()
