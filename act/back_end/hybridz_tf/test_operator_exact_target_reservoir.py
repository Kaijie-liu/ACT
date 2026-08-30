#!/usr/bin/env python3
"""Toy-first soundness audits for the RBS same-layer exact-bit reservoir."""

from __future__ import annotations

from fractions import Fraction
import itertools
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.sparse as sp
from scipy.optimize import linprog
import torch

from act.back_end.core import Bounds, ConSet, Fact
from act.back_end.hybridz_tf import operator_hz as operator_module
from act.back_end.hybridz_tf.operator_hz import (
    OperatorHZBuildError,
    build_operator_hz,
)
from act.back_end.solver.solver_hz import SparseHZono


def _layer(layer_id: int, kind: str, width: int, params=None):
    return SimpleNamespace(
        id=int(layer_id),
        kind=str(kind),
        params=dict(params or {}),
        out_vars=list(range(int(width))),
        in_vars=[],
    )


def _reservoir_toy():
    """Return one residual route with two post-RBS-stable and three crossing rows.

    The materialized ADD stores ``(x,-x)`` as an independent cube.  Its
    pre-materialization shadow proves row 0 to be the constant ``+1/4`` and
    tightens rows 1--3 while they remain crossing-zero.  Row 4 is already the
    independent-cube constant ``-1/4`` and therefore is not an RBS vacancy.
    """

    dtype = torch.float64
    pre_weight = torch.tensor(
        [
            [1.0, 1.0],
            [1.0, 0.5],
            [0.5, 1.0],
            [1.0, -0.5],
            [0.0, 0.0],
        ],
        dtype=dtype,
    )
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
            5,
            {
                "weight": pre_weight,
                "bias": torch.tensor(
                    [0.25, 0.0, 0.0, 0.0, -0.25], dtype=dtype
                ),
                "in_features": 2,
                "out_features": 5,
            },
        ),
        _layer(6, "RELU", 5),
        _layer(
            7,
            "DENSE",
            1,
            {
                "weight": torch.tensor(
                    [[0.0, 1.0, 1.0, 0.0, 0.0]], dtype=dtype
                ),
                "bias": torch.zeros(1, dtype=dtype),
                "in_features": 5,
                "out_features": 1,
            },
        ),
        _layer(8, "ASSERT", 1, {"kind": "UNSAFE_LINEAR"}),
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
        8: [7],
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
    widths = {0: 1, 1: 1, 2: 2, 3: 2, 4: 2, 5: 5, 6: 5, 7: 1, 8: 1}
    facts = {}
    for layer in layers:
        width = widths[layer.id]
        if layer.id in {0, 1}:
            lower = torch.tensor([[-1.0]], dtype=dtype)
            upper = torch.tensor([[1.0]], dtype=dtype)
        else:
            lower = torch.full((1, width), -100.0, dtype=dtype)
            upper = torch.full((1, width), 100.0, dtype=dtype)
        facts[layer.id] = Fact(Bounds(lower, upper), ConSet())
    return net, facts


def _relu_metadata(build):
    return next(
        item for item in build.metadata["layers"]
        if item["layer_id"] == 6
    )


def _assert_csr_equal(test: unittest.TestCase, left, right) -> None:
    left = left.tocsr()
    right = right.tocsr()
    test.assertEqual(left.shape, right.shape)
    test.assertTrue(np.array_equal(left.indptr, right.indptr))
    test.assertTrue(np.array_equal(left.indices, right.indices))
    test.assertTrue(np.array_equal(left.data, right.data))


def _assert_hz_core_equal(test: unittest.TestCase, left, right) -> None:
    # Stable IDs are intentionally minted afresh for every independent build;
    # compare the represented numeric core and only the ID cardinalities.
    for name in ("c", "b", "ub"):
        test.assertTrue(np.array_equal(getattr(left, name), getattr(right, name)))
    test.assertEqual(np.asarray(left.col_ids).size, np.asarray(right.col_ids).size)
    test.assertEqual(np.asarray(left.bcol_ids).size, np.asarray(right.bcol_ids).size)
    for name in ("Gc", "Gb", "Ac", "Ab", "Auc", "Aub"):
        _assert_csr_equal(test, getattr(left, name), getattr(right, name))


def _integer_output_max(build) -> float:
    hz = build.hz
    objective = hz.Gc.getrow(0).toarray().reshape(-1)
    binary_objective = hz.Gb.getrow(0).toarray().reshape(-1)
    best = -np.inf
    for assignment in itertools.product((-1.0, 1.0), repeat=hz.n_bin):
        binary = np.asarray(assignment, dtype=np.float64)
        result = linprog(
            -objective,
            A_ub=hz.Auc if hz.n_ub else None,
            b_ub=(
                hz.ub - np.asarray(hz.Aub @ binary).reshape(-1)
                if hz.n_ub else None
            ),
            A_eq=hz.Ac if hz.n_eq else None,
            b_eq=(
                hz.b - np.asarray(hz.Ab @ binary).reshape(-1)
                if hz.n_eq else None
            ),
            bounds=[(-1.0, 1.0)] * hz.n_cont,
            method="highs",
        )
        if result.success:
            best = max(
                best,
                float(
                    hz.c[0]
                    + binary_objective @ binary
                    - result.fun
                ),
            )
    if not np.isfinite(best):
        raise AssertionError("no exact-binary HZ leaf was feasible")
    return best


def _exact_point_is_feasible(build, x_value: float) -> bool:
    """Fix both the normalized input and exact network output, then enumerate bits."""

    hz = build.hz
    stable_input_id = int(np.asarray(build.input_col_ids).reshape(-1)[0])
    positions = np.flatnonzero(np.asarray(hz.col_ids) == stable_input_id)
    if positions.size != 1:
        raise AssertionError("input stable id did not map uniquely")
    input_position = int(positions[0])
    output_value = 0.5 * abs(float(x_value))
    input_row = sp.csr_matrix(
        ([1.0], ([0], [input_position])), shape=(1, hz.n_cont)
    )
    output_row = hz.Gc.getrow(0)
    extra = sp.vstack([input_row, output_row], format="csr")
    A_eq = sp.vstack([hz.Ac, extra], format="csr")
    output_binary = hz.Gb.getrow(0).toarray().reshape(-1)
    for assignment in itertools.product((-1.0, 1.0), repeat=hz.n_bin):
        binary = np.asarray(assignment, dtype=np.float64)
        base_rhs = (
            hz.b - np.asarray(hz.Ab @ binary).reshape(-1)
            if hz.n_eq else np.zeros(0, dtype=np.float64)
        )
        extra_rhs = np.asarray(
            [
                float(x_value),
                output_value - float(hz.c[0]) - float(output_binary @ binary),
            ],
            dtype=np.float64,
        )
        result = linprog(
            np.zeros(hz.n_cont, dtype=np.float64),
            A_ub=hz.Auc if hz.n_ub else None,
            b_ub=(
                hz.ub - np.asarray(hz.Aub @ binary).reshape(-1)
                if hz.n_ub else None
            ),
            A_eq=A_eq,
            b_eq=np.concatenate([base_rhs, extra_rhs]),
            bounds=[(-1.0, 1.0)] * hz.n_cont,
            method="highs",
        )
        if result.success:
            return True
    return False


class OperatorExactTargetReservoirTests(unittest.TestCase):
    def test_final_stack_uses_zero_column_views_and_releases_traversal(self):
        left = sp.csr_matrix(
            np.array([[1.0, 0.0], [0.0, 2.0]], dtype=np.float64)
        )
        right = sp.csr_matrix(
            np.array([[0.0, 3.0, 0.0]], dtype=np.float64)
        )
        left_before = left.copy()
        right_before = right.copy()
        stacked = operator_module._stack_padded(
            (left, right), width=5
        )
        expected = sp.csr_matrix(
            np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 2.0, 0.0, 0.0, 0.0],
                    [0.0, 3.0, 0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            )
        )
        _assert_csr_equal(self, stacked, expected)
        _assert_csr_equal(self, left, left_before)
        _assert_csr_equal(self, right, right_before)
        self.assertFalse(np.shares_memory(stacked.data, left.data))
        self.assertFalse(np.shares_memory(stacked.data, right.data))

        net, facts = _reservoir_toy()
        with mock.patch.object(
            SparseHZono,
            "__post_init__",
            side_effect=AssertionError(
                "owned operator core was redundantly recopied"
            ),
        ):
            build = build_operator_hz(
                net,
                facts,
                facts,
                exact_budget=2,
                materialize_add=True,
                residual_bound_screen=True,
                residual_targets=[(6, 0, "both"), (6, 1, "both")],
                exact_target_reservoir=[(6, 2), (6, 3)],
                export_verified_preactivation_frame=False,
            )
        receipt = build.metadata["traversal_cache_release"]
        self.assertEqual(
            receipt["status"],
            "released_before_final_sparse_assembly",
        )
        self.assertFalse(receipt["numeric_semantics_changed"])
        self.assertGreater(receipt["expr_count"], 0)
        self.assertGreater(receipt["expr_value_nnz_reference_sum"], 0)
        self.assertGreater(receipt["upper_block_count_released"], 0)
        self.assertEqual(build.hz.n_bin, 2)
        self.assertEqual(
            build.metadata["sparse_hz_core_assembly"],
            "owned_canonical_no_recopy_v1",
        )
        self.assertNotIn("performance_diagnostic", build.metadata)
        performance = build.performance_diagnostic
        self.assertIsNotNone(performance)
        self.assertEqual(
            performance["schema"],
            "operator_hz_build_performance_diagnostic_v1",
        )
        self.assertFalse(performance["proof_authority"])
        self.assertFalse(performance["verdict_authority"])
        self.assertEqual(
            [item["layer_id"] for item in performance["layers"]],
            [layer.id for layer in net.layers],
        )
        for item in performance["layers"]:
            self.assertGreaterEqual(item["wall_seconds"], 0.0)
            self.assertGreaterEqual(item["process_cpu_seconds"], 0.0)
            self.assertGreaterEqual(item["minor_faults_delta"], 0)
            self.assertGreaterEqual(item["major_faults_delta"], 0)
        for value in performance["stages"].values():
            self.assertGreaterEqual(value, 0.0)
        self.assertEqual(
            build.metadata["build_seconds"],
            performance["total_wall_seconds"],
        )

    def test_post_rbs_stable_primary_is_replaced_only_from_same_layer(self):
        net, facts = _reservoir_toy()
        primary = [(6, 0, "both"), (6, 1, "both")]
        no_reserve = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=2,
            materialize_add=True,
            residual_bound_screen=True,
            residual_targets=primary,
            export_verified_preactivation_frame=False,
        )
        candidate = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=2,
            materialize_add=True,
            residual_bound_screen=True,
            residual_targets=primary,
            exact_target_reservoir=[(6, 2), (6, 3)],
            export_verified_preactivation_frame=False,
        )
        self.assertEqual(no_reserve.hz.n_bin, 1)
        self.assertEqual(candidate.hz.n_bin, 2)
        meta = _relu_metadata(candidate)
        self.assertEqual(meta["exact_index_preview"], [1, 2])
        self.assertEqual(meta["relu_triangle_rows"], 1)
        self.assertEqual(meta["relu_residual_rows"], 0)
        receipt = meta["exact_target_reservoir"]
        self.assertEqual(receipt["status"], "filled")
        self.assertEqual(receipt["replacement_slots"], [
            {"stabilized_primary_row": 0, "selected_reserve_row": 2}
        ])
        self.assertEqual(receipt["selected_rows"], [1, 2])
        self.assertTrue(receipt["all_selected_rows_rbs_tightened"])
        self.assertTrue(receipt["unselected_reserves_use_ordinary_triangle"])
        self.assertEqual(candidate.metadata["exact_budget_used"], 2)
        self.assertEqual(
            candidate.metadata["exact_target_reservoir_replacements_used"], 1
        )

        # Exact oracle: the selected rows are +/-x/2, so their summed ReLU
        # maximum is exactly 1/2.  Integer leaf enumeration must reproduce it.
        measured_upper = _integer_output_max(candidate)
        self.assertAlmostEqual(measured_upper, 0.5, places=10)
        self.assertEqual(
            Fraction.from_float(measured_upper).limit_denominator(1 << 20),
            Fraction(1, 2),
        )
        for point in (-1.0, -0.5, 0.0, 0.25, 1.0):
            self.assertTrue(
                _exact_point_is_feasible(candidate, point),
                f"exact graph point x={point} was excluded",
            )

    def test_unused_reservoir_is_core_identical_to_old_selection(self):
        net, facts = _reservoir_toy()
        primary = [(6, 1, "both"), (6, 2, "both")]
        baseline = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=2,
            materialize_add=True,
            residual_bound_screen=True,
            residual_targets=primary,
            export_verified_preactivation_frame=False,
        )
        candidate = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=2,
            materialize_add=True,
            residual_bound_screen=True,
            residual_targets=primary,
            exact_target_reservoir=[(6, 3)],
            export_verified_preactivation_frame=False,
        )
        _assert_hz_core_equal(self, baseline.hz, candidate.hz)
        receipt = _relu_metadata(candidate)["exact_target_reservoir"]
        self.assertFalse(receipt["reservoir_consulted"])
        self.assertEqual(receipt["selected_reserve_rows"], [])

    def test_exhausted_reservoir_is_sound_shortfall_not_cross_layer_fill(self):
        net, facts = _reservoir_toy()
        candidate = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=1,
            materialize_add=True,
            residual_bound_screen=True,
            residual_targets=[(6, 0, "both")],
            exact_target_reservoir=[(6, 4)],
            export_verified_preactivation_frame=False,
        )
        self.assertEqual(candidate.hz.n_bin, 0)
        receipt = _relu_metadata(candidate)["exact_target_reservoir"]
        self.assertEqual(receipt["status"], "post_screen_reservoir_exhausted")
        self.assertEqual(receipt["shortfall"], 1)
        self.assertEqual(candidate.metadata["exact_target_reservoir_shortfall"], 1)

    def test_preexisting_stable_primary_does_not_open_rbs_vacancy(self):
        net, facts = _reservoir_toy()
        candidate = build_operator_hz(
            net,
            facts,
            facts,
            exact_budget=1,
            materialize_add=True,
            residual_bound_screen=True,
            residual_targets=[(6, 4, "both")],
            exact_target_reservoir=[(6, 3)],
            export_verified_preactivation_frame=False,
        )
        self.assertEqual(candidate.hz.n_bin, 0)
        receipt = _relu_metadata(candidate)["exact_target_reservoir"]
        self.assertFalse(receipt["reservoir_consulted"])
        self.assertEqual(receipt["rbs_newly_stabilized_primary"], [])
        self.assertEqual(receipt["non_rbs_stable_primary_not_replaced"], [4])
        self.assertEqual(receipt["selected_reserve_rows"], [])
        self.assertEqual(receipt["shortfall"], 1)

    def test_frame_export_switch_changes_no_hz_semantics(self):
        net, facts = _reservoir_toy()
        exported = build_operator_hz(
            net,
            facts,
            facts,
            residual_bound_screen=True,
            export_verified_preactivation_frame=True,
        )
        closed = build_operator_hz(
            net,
            facts,
            facts,
            residual_bound_screen=True,
            export_verified_preactivation_frame=False,
        )
        self.assertIsNotNone(exported.verified_preactivation_frame)
        self.assertIsNone(closed.verified_preactivation_frame)
        self.assertTrue(exported.metadata["verified_preactivation_frame_exported"])
        self.assertFalse(closed.metadata["verified_preactivation_frame_exported"])
        _assert_hz_core_equal(self, exported.hz, closed.hz)

    def test_malformed_or_cross_mode_reservoirs_fail_closed(self):
        net, facts = _reservoir_toy()
        primary = [(6, 0, "both")]
        bad_kwargs = (
            {"residual_bound_screen": False, "exact_target_reservoir": [(6, 1)]},
            {"residual_phase_screen": True, "exact_target_reservoir": [(6, 1)]},
            {"exact_budget": 2, "exact_target_reservoir": [(6, 1)]},
            {"exact_target_reservoir": [(6, 0)]},
            {"exact_target_reservoir": [(7, 0)]},
            {"exact_target_reservoir": [(6, 1), (6, 1)]},
            {
                "exact_target_reservoir": [
                    (6, 1), (6, 2), (6, 3), (6, 4)
                ]
            },
            {"exact_target_reservoir": [(6, True)]},
            {"exact_target_reservoir": [(6, 99)]},
        )
        for overrides in bad_kwargs:
            kwargs = {
                "exact_budget": 1,
                "materialize_add": True,
                "residual_bound_screen": True,
                "residual_targets": primary,
                **overrides,
            }
            with self.subTest(overrides=overrides):
                with self.assertRaises(OperatorHZBuildError):
                    build_operator_hz(net, facts, facts, **kwargs)
        with self.assertRaises(OperatorHZBuildError):
            build_operator_hz(
                net,
                facts,
                facts,
                residual_bound_screen=True,
                export_verified_preactivation_frame=1,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
