#!/usr/bin/env python3
"""Deterministic toy gates for multi-rival residual target scheduling."""

from __future__ import annotations

import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from act.back_end.config import BackendConfig, HybridZConfig
from act.back_end.core import Bounds, ConSet, Fact, Layer, Net
from act.back_end.hybridz_tf.property_residual_targets import (
    plan_from_property_adjoints,
    plan_sparse_query_rows_from_property_adjoints,
    property_correlation_layer_quotas,
    select_property_residual_targets,
)
from act.back_end.transfer_functions import (
    set_solver_mode,
    set_transfer_function_mode,
)
from act.back_end.verifier import verify_once


def _fact(lower, upper) -> Fact:
    return Fact(
        Bounds(
            torch.as_tensor(lower, dtype=torch.float64),
            torch.as_tensor(upper, dtype=torch.float64),
        ),
        ConSet(),
    )


def _one_relu_net() -> tuple[Net, dict[int, Fact], dict[int, Fact]]:
    layers = [
        Layer(
            id=0,
            kind="INPUT",
            params={"shape": (1, 1), "dtype": "torch.float64"},
            in_vars=[],
            out_vars=[0],
        ),
        Layer(
            id=1,
            kind="INPUT_SPEC",
            params={
                "kind": "BOX",
                "lb": torch.tensor([[-1.0]], dtype=torch.float64),
                "ub": torch.tensor([[1.0]], dtype=torch.float64),
            },
            in_vars=[0],
            out_vars=[0],
        ),
        Layer(
            id=2,
            kind="RELU",
            params={},
            in_vars=[0],
            out_vars=[1],
        ),
        Layer(
            id=3,
            kind="ASSERT",
            params={
                "kind": "LINEAR_LE",
                "C": torch.tensor([[1.0]], dtype=torch.float64),
                "thresholds": torch.tensor([[0.0]], dtype=torch.float64),
                "M": 1,
            },
            in_vars=[1],
            out_vars=[1],
        ),
    ]
    net = Net(
        layers=layers,
        preds={0: [], 1: [0], 2: [1], 3: [2]},
        succs={0: [1], 1: [2], 2: [3], 3: []},
    )
    before = {
        0: _fact([[-1.0]], [[1.0]]),
        1: _fact([[-1.0]], [[1.0]]),
        2: _fact([[-1.0]], [[1.0]]),
        3: _fact([[0.0]], [[1.0]]),
    }
    after = {
        0: _fact([[-1.0]], [[1.0]]),
        1: _fact([[-1.0]], [[1.0]]),
        2: _fact([[0.0]], [[1.0]]),
        3: _fact([[0.0]], [[1.0]]),
    }
    return net, before, after


class PropertyResidualTargetTests(unittest.TestCase):
    def test_correlation_quotas_cover_each_residual_depth_first(self) -> None:
        def layer(layer_id: int, kind: str, width: int):
            return SimpleNamespace(
                id=layer_id,
                kind=kind,
                params={},
                in_vars=[],
                out_vars=list(range(width)),
            )

        layers = [
            layer(0, "INPUT", 1),
            layer(1, "DENSE", 4),
            layer(2, "DENSE", 4),
            layer(3, "ADD", 4),
            layer(4, "DENSE", 4),
            layer(5, "RELU", 4),
            layer(6, "ADD", 3),
            layer(7, "RELU", 3),
            layer(8, "ASSERT", 3),
        ]
        preds = {
            0: [],
            1: [0],
            2: [0],
            3: [1, 2],
            4: [3],
            5: [4],
            6: [5, 3],
            7: [6],
            8: [7],
        }
        succs = {item.id: [] for item in layers}
        for child, parents in preds.items():
            for parent in parents:
                succs[parent].append(child)
        net = SimpleNamespace(
            layers=layers, preds=preds, succs=succs
        )
        self.assertEqual(
            property_correlation_layer_quotas(
                net, budget=5, per_layer_cap=3
            ),
            {5: 3, 7: 2},
        )
        self.assertEqual(
            property_correlation_layer_quotas(
                net, budget=1, per_layer_cap=3
            ),
            {5: 1, 7: 0},
        )

    def test_sparse_query_facility_prefix_is_nested_per_layer(self) -> None:
        bounds = {
            7: _fact([[-1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0]]),
            9: _fact(
                [[-2.0, -2.0, 0.1, -2.0]],
                [[2.0, 2.0, 1.0, 2.0]],
            ),
        }
        adjoints = {
            7: torch.tensor(
                [
                    [1.0, 0.0, 0.65],
                    [0.0, 1.0, 0.65],
                    [0.1, 0.1, 0.65],
                ],
                dtype=torch.float64,
            ),
            9: torch.tensor(
                [
                    [3.0, 0.2, 1000.0, 1.0],
                    [0.1, 2.0, 1000.0, 1.0],
                    [0.2, 0.1, 1000.0, 1.5],
                ],
                dtype=torch.float64,
            ),
        }
        common = dict(
            property_adjoints=adjoints,
            before=bounds,
            rival_ids=[10, 20, 30],
            rival_hardness=[3.0, 2.0, 1.0],
            all_rivals_processed=True,
            property_sha256="sparse-toy",
            pool_per_rival=3,
        )
        small = plan_sparse_query_rows_from_property_adjoints(
            layer_quotas={7: 1, 9: 1},
            **common,
        )
        large = plan_sparse_query_rows_from_property_adjoints(
            layer_quotas={7: 2, 9: 3},
            **common,
        )
        self.assertEqual(small.rows_for_layer(7), large.rows_for_layer(7)[:1])
        self.assertEqual(small.rows_for_layer(9), large.rows_for_layer(9)[:1])
        self.assertEqual(small.rows_for_layer(7), (2,))
        # Stable row 2 is a deliberately enormous distractor.
        self.assertNotIn(2, large.rows_for_layer(9))
        self.assertEqual(len(large.rows_for_layer(9)), 3)
        for layer in large.receipt["layers"]:
            self.assertTrue(layer["quota_filled"])
            self.assertTrue(layer["partition_complete"])
            self.assertTrue(layer["partition_disjoint"])

    def test_sparse_query_multi_rival_schedule_survives_permutation(self) -> None:
        bounds = {
            5: _fact([[-1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0]])
        }
        adjoint = torch.tensor(
            [
                [1.0, 0.2, 0.8],
                [0.2, 1.0, 0.8],
                [0.2, 0.2, 0.8],
            ],
            dtype=torch.float64,
        )
        first = plan_sparse_query_rows_from_property_adjoints(
            {5: adjoint},
            bounds,
            layer_quotas={5: 3},
            rival_ids=[10, 20, 30],
            rival_hardness=[1.0, 1.0, 1.0],
            all_rivals_processed=True,
            property_sha256="permuted",
            pool_per_rival=3,
        )
        permutation = [2, 0, 1]
        second = plan_sparse_query_rows_from_property_adjoints(
            {5: adjoint[permutation]},
            bounds,
            layer_quotas={5: 3},
            rival_ids=[30, 10, 20],
            rival_hardness=[1.0, 1.0, 1.0],
            all_rivals_processed=True,
            property_sha256="permuted",
            pool_per_rival=3,
        )
        self.assertEqual(first.rows_for_layer(5), second.rows_for_layer(5))
        self.assertEqual(first.selection_sha256, second.selection_sha256)
        self.assertEqual(
            [target.dominant_rival for target in first.targets],
            [target.dominant_rival for target in second.targets],
        )
        self.assertEqual(first.targets[0].row, 2)
        self.assertEqual(first.targets[0].dominant_rival, 10)

    def test_facility_greedy_covers_multiple_rivals_and_is_nested(self) -> None:
        bounds = {7: _fact([[-2.0, -2.0, -2.0]], [[2.0, 2.0, 2.0]])}
        # Candidate columns:
        #   j0=[1,0,.1], j1=[0,1,.1], j2=[.65,.65,.65].
        # The first facility choice must cover all three rivals.
        adjoint = torch.tensor(
            [
                [1.0, 0.0, 0.65],
                [0.0, 1.0, 0.65],
                [0.1, 0.1, 0.65],
            ],
            dtype=torch.float64,
        )
        common = dict(
            property_adjoints={7: adjoint},
            before=bounds,
            rival_ids=[0, 1, 2],
            rival_hardness=[1.0, 1.0, 1.0],
            all_rivals_processed=True,
            property_sha256="toy",
            pool_per_rival=3,
        )
        one = plan_from_property_adjoints(budget=1, **common)
        two = plan_from_property_adjoints(budget=2, **common)
        self.assertEqual((one.targets[0].layer_id, one.targets[0].row), (7, 2))
        self.assertEqual(two.targets[:1], one.targets)
        self.assertEqual(len(two.targets), 2)

    def test_cutoff_ties_make_budget_four_a_hashed_prefix_of_sixteen(
        self,
    ) -> None:
        width = 24
        bounds = {
            layer_id: _fact([[-1.0] * width], [[1.0] * width])
            for layer_id in (3, 7)
        }
        early_adjoint = torch.full((2, width), 0.1, dtype=torch.float64)
        early_adjoint[0, 0:4] = 4.0
        early_adjoint[0, 4:10] = 2.0
        early_adjoint[1, 12:16] = 4.0
        early_adjoint[1, 16:22] = 2.0
        late_adjoint = torch.full((2, width), 0.1, dtype=torch.float64)
        late_adjoint[0, 0:4] = 2.0
        late_adjoint[1, 12:16] = 2.0
        common = dict(
            property_adjoints={3: early_adjoint, 7: late_adjoint},
            before=bounds,
            rival_ids=[10, 20],
            rival_hardness=[2.0, 1.0],
            all_rivals_processed=True,
            property_sha256="cutoff-prefix-toy",
            pool_per_rival=8,
            phase_joint_focus_after_first=True,
        )

        budget_four = plan_from_property_adjoints(budget=4, **common)
        budget_sixteen = plan_from_property_adjoints(budget=16, **common)

        # Each rival has six rows tied at the eighth-place cutoff.  Row-id
        # breaks that boundary tie, and layer-id breaks the equal-score tie
        # between the two ReLUs before the per-rival pool is capped.
        self.assertEqual(
            {
                (target.layer_id, target.row)
                for target in budget_sixteen.targets
            },
            {(3, row) for row in range(8)}
            | {(3, row) for row in range(12, 20)},
        )
        self.assertEqual(
            budget_four.targets,
            budget_sixteen.targets[:4],
        )
        self.assertEqual(
            budget_four.receipt["schedule"],
            budget_sixteen.receipt["schedule"][:4],
        )

        repeated_four = plan_from_property_adjoints(budget=4, **common)
        self.assertEqual(
            budget_four.targets_sha256,
            repeated_four.targets_sha256,
        )
        self.assertNotEqual(
            budget_four.targets_sha256,
            budget_sixteen.targets_sha256,
        )

    def test_joint_phase_second_target_stays_on_first_bottleneck(self) -> None:
        bounds = {
            7: _fact([[-2.0, -2.0, -2.0]], [[2.0, 2.0, 2.0]])
        }
        adjoint = torch.tensor(
            [
                [0.9, 0.1, 0.8],
                [0.1, 2.0, 0.8],
                [0.1, 0.1, 0.8],
            ],
            dtype=torch.float64,
        )
        common = dict(
            property_adjoints={7: adjoint},
            before=bounds,
            budget=2,
            rival_ids=[0, 1, 2],
            rival_hardness=[3.0, 2.0, 1.0],
            all_rivals_processed=True,
            property_sha256="joint-focus",
            pool_per_rival=3,
        )
        facility = plan_from_property_adjoints(**common)
        focused = plan_from_property_adjoints(
            **common,
            phase_joint_focus_after_first=True,
        )
        self.assertEqual([target.row for target in facility.targets], [2, 1])
        self.assertEqual([target.row for target in focused.targets], [2, 0])
        self.assertEqual(
            [target.dominant_rival for target in focused.targets],
            [0, 0],
        )
        self.assertEqual(
            focused.receipt["selection_policy"],
            "facility_first_then_same_rival_joint",
        )
        self.assertEqual(focused.receipt["joint_focus_rival_id"], 0)

    def test_triangle_gap_adjoint_score_selects_tightness_critical_row(self) -> None:
        # Matches the controlled exact-budget tightness toy: row 2 has the
        # largest |adjoint| * triangle-gap score and is therefore selected
        # ahead of the topological row 0.
        bounds = {
            3: _fact(
                [[-2.0, -4.0, -3.5]],
                [[4.0, 2.0, 2.5]],
            )
        }
        plan = plan_from_property_adjoints(
            {
                3: torch.tensor(
                    [[-2.0, -3.0, 3.0]],
                    dtype=torch.float64,
                )
            },
            bounds,
            budget=1,
            rival_ids=[0],
            rival_hardness=[1.0],
            all_rivals_processed=True,
            property_sha256="guided-exact-tightness-toy",
            pool_per_rival=3,
        )
        self.assertEqual(len(plan.targets), 1)
        self.assertEqual(
            (plan.targets[0].layer_id, plan.targets[0].row),
            (3, 2),
        )
        self.assertGreater(plan.targets[0].score, 4.37)

    def test_signed_all_rival_guard_and_incomplete_memory_fallback(self) -> None:
        bounds = {4: _fact([[-1.0, -1.0]], [[1.0, 1.0]])}
        adjoint = torch.tensor(
            [[2.0, 3.0], [1.0, -4.0]],
            dtype=torch.float64,
        )
        complete = plan_from_property_adjoints(
            {4: adjoint},
            bounds,
            budget=2,
            rival_ids=[10, 11],
            rival_hardness=[2.0, 1.0],
            all_rivals_processed=True,
            property_sha256="sign",
            pool_per_rival=2,
        )
        guards = {
            target.row: target.guard for target in complete.targets
        }
        self.assertEqual(guards[0], "none")
        self.assertEqual(guards[1], "both")

        incomplete = plan_from_property_adjoints(
            {4: adjoint[:1]},
            bounds,
            budget=2,
            rival_ids=[10],
            rival_hardness=[2.0],
            all_rivals_processed=False,
            property_sha256="incomplete",
            pool_per_rival=2,
        )
        self.assertTrue(
            all(target.guard == "both" for target in incomplete.targets)
        )

    def test_stable_neurons_are_never_scheduled(self) -> None:
        bounds = {
            9: _fact(
                [[-1.0, 0.1, -2.0]],
                [[1.0, 2.0, -0.1]],
            )
        }
        plan = plan_from_property_adjoints(
            {9: torch.tensor([[1.0, 1.0e9, 1.0e9]], dtype=torch.float64)},
            bounds,
            budget=3,
            rival_ids=[0],
            rival_hardness=[1.0],
            all_rivals_processed=True,
            property_sha256="stable",
            pool_per_rival=3,
        )
        self.assertEqual(
            [
                (target.layer_id, target.row)
                for target in plan.targets
            ],
            [(9, 0)],
        )

    def test_rival_permutation_preserves_coordinate_schedule(self) -> None:
        bounds = {5: _fact([[-1.0, -1.0, -1.0]], [[1.0, 1.0, 1.0]])}
        adjoint = torch.tensor(
            [[4.0, 0.2, 1.0], [0.1, 3.0, 1.0], [0.2, 0.1, 2.0]],
            dtype=torch.float64,
        )
        first = plan_from_property_adjoints(
            {5: adjoint},
            bounds,
            budget=3,
            rival_ids=[0, 1, 2],
            rival_hardness=[3.0, 2.0, 1.0],
            all_rivals_processed=True,
            property_sha256="perm",
            pool_per_rival=3,
        )
        permutation = [2, 0, 1]
        second = plan_from_property_adjoints(
            {5: adjoint[permutation]},
            bounds,
            budget=3,
            rival_ids=permutation,
            rival_hardness=[1.0, 3.0, 2.0],
            all_rivals_processed=True,
            property_sha256="perm",
            pool_per_rival=3,
        )
        self.assertEqual(
            [(target.layer_id, target.row) for target in first.targets],
            [(target.layer_id, target.row) for target in second.targets],
        )

    def test_dualsolver_wrapper_uses_final_property_signs(self) -> None:
        net, before, after = _one_relu_net()
        positive = select_property_residual_targets(
            net=net,
            before=before,
            after=after,
            C=torch.tensor([[1.0]], dtype=torch.float64),
            thresholds=torch.tensor([-2.0], dtype=torch.float64),
            kind="TOP1_ROBUST",
            output_layer_id=2,
            budget=1,
            time_limit=2.0,
            deadline=time.monotonic() + 2.0,
            max_adjoint_cells=10,
            pool_per_rival=1,
        )
        self.assertEqual(len(positive.targets), 1)
        self.assertEqual(positive.targets[0].guard, "none")
        self.assertFalse(positive.receipt["proof_authority"])

        mixed = select_property_residual_targets(
            net=net,
            before=before,
            after=after,
            C=torch.tensor([[1.0], [-1.0]], dtype=torch.float64),
            thresholds=torch.tensor([-2.0, -2.0], dtype=torch.float64),
            kind="TOP1_ROBUST",
            output_layer_id=2,
            budget=1,
            time_limit=2.0,
            max_adjoint_cells=10,
            pool_per_rival=1,
        )
        self.assertEqual(len(mixed.targets), 1)
        self.assertEqual(mixed.targets[0].guard, "both")

    def test_cell_cap_forces_both_and_unsafe_linear_is_disabled(self) -> None:
        net, before, after = _one_relu_net()
        capped = select_property_residual_targets(
            net=net,
            before=before,
            after=after,
            C=torch.tensor([[1.0], [2.0]], dtype=torch.float64),
            thresholds=torch.tensor([-2.0, -2.0], dtype=torch.float64),
            kind="TOP1_ROBUST",
            output_layer_id=2,
            budget=1,
            time_limit=2.0,
            max_adjoint_cells=1,
            pool_per_rival=1,
        )
        self.assertEqual(len(capped.targets), 1)
        self.assertEqual(capped.targets[0].guard, "both")
        self.assertFalse(capped.receipt["all_interval_survivors_processed"])

        unsupported = select_property_residual_targets(
            net=net,
            before=before,
            after=after,
            C=torch.tensor([[1.0]], dtype=torch.float64),
            thresholds=torch.tensor([0.0], dtype=torch.float64),
            kind="UNSAFE_LINEAR",
            output_layer_id=2,
            budget=1,
            time_limit=2.0,
        )
        self.assertEqual(unsupported.targets, ())
        self.assertEqual(
            unsupported.receipt["status"],
            "unsupported_joint_unsafe",
        )

    def test_zero_budget_never_invokes_dual(self) -> None:
        net, before, after = _one_relu_net()
        plan = select_property_residual_targets(
            net=net,
            before=before,
            after=after,
            C=torch.tensor([[1.0]], dtype=torch.float64),
            thresholds=torch.tensor([0.0], dtype=torch.float64),
            kind="TOP1_ROBUST",
            output_layer_id=2,
            budget=0,
            time_limit=0.0,
        )
        self.assertEqual(plan.targets, ())
        self.assertEqual(plan.receipt["status"], "disabled")

    def test_verifier_default_off_does_not_call_selector(self) -> None:
        net, _before, _after = _one_relu_net()
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            config = BackendConfig(
                solver="hybridz",
                device="cpu",
                dtype="float64",
                hybridz=HybridZConfig(
                    timeout=2.0,
                    engine="operator_hz_objbound",
                    property_residual_budget=0,
                    property_residual_time_limit=0.0,
                ),
            )
            with patch(
                "act.back_end.hybridz_tf.property_residual_targets."
                "select_property_residual_targets",
                side_effect=AssertionError("default-off selector was called"),
            ):
                result = verify_once(net, backend_cfg=config)[0]
            selector = result.metadata["property_residual_selector"]
            self.assertEqual(selector["status"], "disabled")
            self.assertFalse(
                result.metadata["operator_hz"]["residual_targets_explicit"]
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verifier_routes_same_final_property_to_explicit_targets(self) -> None:
        net, _before, _after = _one_relu_net()
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            config = BackendConfig(
                solver="hybridz",
                device="cpu",
                dtype="float64",
                hybridz=HybridZConfig(
                    timeout=3.0,
                    engine="operator_hz_objbound",
                    property_residual_budget=1,
                    property_residual_time_limit=1.0,
                    property_residual_max_adjoint_cells=10,
                    property_residual_pool_per_rival=1,
                ),
            )
            result = verify_once(net, backend_cfg=config)[0]
            selector = result.metadata["property_residual_selector"]
            operator = result.metadata["operator_hz"]
            self.assertEqual(selector["targets_selected"], 1)
            self.assertEqual(selector["guard_none"], 1)
            self.assertFalse(selector["proof_authority"])
            self.assertTrue(operator["residual_targets_explicit"])
            self.assertEqual(operator["residual_targets_applied"], 1)
            self.assertEqual(
                result.metadata["property_residual_property_sha256"],
                selector["property_sha256"],
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")

    def test_verifier_routes_property_target_into_positive_exact_budget(self) -> None:
        net, _before, _after = _one_relu_net()
        set_solver_mode("hybridz")
        set_transfer_function_mode("hybridz")
        try:
            config = BackendConfig(
                solver="hybridz",
                device="cpu",
                dtype="float64",
                hybridz=HybridZConfig(
                    timeout=3.0,
                    engine="operator_hz_objbound",
                    operator_exact_budget=1,
                    property_residual_budget=1,
                    property_residual_time_limit=1.0,
                    property_residual_max_adjoint_cells=10,
                    property_residual_pool_per_rival=1,
                ),
            )
            result = verify_once(net, backend_cfg=config)[0]
            selector = result.metadata["property_residual_selector"]
            operator = result.metadata["operator_hz"]
            relu = next(
                item
                for item in operator["layers"]
                if item["layer_id"] == 2
            )
            self.assertEqual(selector["targets_selected"], 1)
            self.assertEqual(
                operator["exact_selection"],
                "property_gap_adjoint_facility_targets_v1",
            )
            self.assertEqual(operator["exact_budget_used"], 1)
            self.assertEqual(relu["exact_index_preview"], [0])
            self.assertEqual(
                relu["relu_exact_selection"],
                "property_gap_adjoint_facility_targets_v1",
            )
            self.assertEqual(
                operator["residual_target_receipts"][0]["status"],
                "skipped_exact",
            )
        finally:
            set_solver_mode(None)
            set_transfer_function_mode("interval")


if __name__ == "__main__":
    unittest.main(verbosity=2)
