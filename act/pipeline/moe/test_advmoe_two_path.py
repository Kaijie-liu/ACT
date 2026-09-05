from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.advmoe_two_path import (
    aggregate_lagrangian_grid_calls,
    aggregate_filters,
    comparison_budget_ledger,
    evaluate_path_lowering_equivalence,
    filter_witness_conflicts,
    lagrangian_mu0_call,
    resolve_selection_manifest,
    select_batched_static_path_logits,
    separate_interval_lagrangian_grid,
    top1_property_rows,
    validate_lagrangian_multiplier_protocol,
)
from act.pipeline.moe.audit_advmoe_two_path import (
    expected_crown_status,
    expected_filter_witness_conflicts,
    expected_separate_interval_control,
    independent_tables,
    lagrangian_branch_issues,
    selection_manifest_issues,
    summary_table_issues,
)
from act.pipeline.moe.lagrangian_guard_incompleteness_control import (
    fixed_multiplier_exact_minimum,
)


class AdvMoeTwoPathTests(unittest.TestCase):
    def test_fixed_multiplier_reduction_has_intrinsic_safe_gap(self) -> None:
        grid = np.linspace(0.0, 4.0, 4001)
        values = np.asarray(
            [fixed_multiplier_exact_minimum(value) for value in grid]
        )
        self.assertAlmostEqual(float(grid[np.argmax(values)]), 1.0, places=12)
        self.assertAlmostEqual(float(values.max()), -0.9, places=12)
        self.assertEqual(fixed_multiplier_exact_minimum(0.0), -1.9)
        self.assertEqual(fixed_multiplier_exact_minimum(2.0), -1.9)

    def test_selection_manifest_resolves_ranks_and_exclusion_independently(self):
        predictions = np.asarray([0, 1, 1, 0])
        labels = np.asarray([0, 0, 1, 1])
        with tempfile.TemporaryDirectory(dir="/data1/Kane/MOE/ACT") as temporary:
            root = Path(temporary)
            source = root / "development.json"
            source.write_text("{}\n", encoding="utf-8")
            source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
            manifest = {
                "schema_version": 1,
                "status": "FROZEN_NOT_RUN",
                "dataset_archive_sha256": "dataset",
                "checkpoint_sha256": "checkpoint",
                "clean_correct_ranks": [1],
                "ordered_dataset_indices": [2],
                "development_exclusion": {
                    "ordered_dataset_indices": [0],
                    "sources": [{"path": str(source), "sha256": source_hash}],
                },
            }
            manifest_path = root / "selection.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            manifest_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            config = {
                "schema_version": 2,
                "selection": {"samples": 1},
                "selection_manifest": str(manifest_path),
                "selection_manifest_sha256": manifest_hash,
            }
            indices, identity = resolve_selection_manifest(
                config,
                Path("/data1/Kane/MOE"),
                predictions,
                labels,
                checkpoint_sha256="checkpoint",
                dataset_sha256="dataset",
            )
            np.testing.assert_array_equal(indices, np.asarray([2]))
            expected, issues = selection_manifest_issues(
                config,
                Path("/data1/Kane/MOE"),
                {**identity, "dataset_indices": [2]},
                predictions,
                labels,
                checkpoint_sha256="checkpoint",
                dataset_sha256="dataset",
            )
            np.testing.assert_array_equal(expected, indices)
            self.assertEqual(issues, [])

    def test_schema_v2_refuses_implicit_first_n_selection(self):
        with self.assertRaisesRegex(ValueError, "selection manifest"):
            resolve_selection_manifest(
                {"schema_version": 2, "selection": {"samples": 1}},
                Path("/data1/Kane/MOE"),
                np.asarray([0]),
                np.asarray([0]),
                checkpoint_sha256="checkpoint",
                dataset_sha256="dataset",
            )

    def test_lagrangian_grid_selects_multiplier_per_property_row(self) -> None:
        calls = [
            {
                "status": "UNKNOWN_RELAXATION",
                "complete": True,
                "lower_bounds": [0.2, -0.4],
            },
            {
                "status": "UNKNOWN_RELAXATION",
                "complete": True,
                "lower_bounds": [-0.1, 0.3],
            },
        ]
        result = aggregate_lagrangian_grid_calls(
            calls, [0.0, 2.0], property_rows=2, tolerance=0.1
        )
        self.assertEqual(result["status"], "CERTIFIED_MARGIN_FILTER")
        self.assertEqual(result["lower_bounds"], [0.2, 0.3])
        self.assertEqual(result["selected_multipliers"], [0.0, 2.0])

    def test_lagrangian_grid_fails_closed_on_backend_error(self) -> None:
        result = aggregate_lagrangian_grid_calls(
            [{"status": "ERROR", "lower_bounds": []}],
            [0.0],
            property_rows=1,
            tolerance=1e-7,
        )
        self.assertEqual(result["status"], "ERROR")
        self.assertFalse(result["complete"])

    def test_lagrangian_grid_fails_closed_on_incomplete_call(self) -> None:
        result = aggregate_lagrangian_grid_calls(
            [
                {
                    "status": "CERTIFIED_MARGIN_FILTER",
                    "complete": False,
                    "lower_bounds": [1.0],
                }
            ],
            [0.0],
            property_rows=1,
            tolerance=1e-7,
        )
        self.assertEqual(result["status"], "UNKNOWN_INCOMPLETE")
        self.assertFalse(result["complete"])
        self.assertEqual(result["lower_bounds"], [])

    def test_graph_matched_mu0_is_unique(self) -> None:
        call = {"lagrangian_multiplier": 0.0, "status": "UNKNOWN_RELAXATION"}
        self.assertIs(lagrangian_mu0_call({"calls": [call]}), call)
        missing = lagrangian_mu0_call(
            {"calls": [{"lagrangian_multiplier": 1.0}]}
        )
        self.assertEqual(missing["status"], "UNKNOWN_INCOMPLETE")

    def test_router_scale_normalized_multiplier_grid_is_bound(self) -> None:
        protocol = validate_lagrangian_multiplier_protocol(
            {
                "multipliers": [0.0, 0.5, 1.0],
                "scale_normalization": {
                    "rule": "DEVELOPMENT_MEDIAN_CLEAN_ABS_ROUTER_MARGIN",
                    "scale": 2.0,
                    "normalized_coefficients": [0.0, 1.0, 2.0],
                    "development_source": "/data1/Kane/MOE/ACT/dev.json",
                    "development_source_sha256": "abc",
                },
            }
        )
        self.assertEqual(protocol["resolved_multipliers"], [0.0, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "frozen scale"):
            validate_lagrangian_multiplier_protocol(
                {
                    "multipliers": [0.0, 0.6],
                    "scale_normalization": {
                        "rule": "DEVELOPMENT_MEDIAN_CLEAN_ABS_ROUTER_MARGIN",
                        "scale": 2.0,
                        "normalized_coefficients": [0.0, 1.0],
                        "development_source_sha256": "abc",
                    },
                }
            )

    def test_separate_interval_control_loses_shared_relation(self) -> None:
        path = {
            "status": "UNKNOWN_RELAXATION",
            "complete": True,
            "lower_bounds": [-0.9],
        }
        observed = separate_interval_lagrangian_grid(
            path,
            margin_lower=-1.0,
            margin_upper=1.0,
            multipliers=[0.0, 1.0],
            tolerance=1e-7,
        )
        self.assertEqual(observed["status"], "UNKNOWN_RELAXATION")
        self.assertEqual(observed["lower_bounds"], [-0.9])
        expected = expected_separate_interval_control(
            path,
            margin_lower=-1.0,
            margin_upper=1.0,
            multipliers=[0.0, 1.0],
            tolerance=1e-7,
        )
        self.assertEqual(observed, expected)

    def test_common_budget_excludes_slow_complete_grid(self) -> None:
        def call(seconds: float, multiplier: float | None = None):
            value = {
                "status": "CERTIFIED_MARGIN_FILTER",
                "complete": True,
                "accounted_wall_seconds": seconds,
            }
            if multiplier is not None:
                value["lagrangian_multiplier"] = multiplier
            return value

        router = call(0.2)
        paths = [call(0.3), call(0.3)]
        eta = [call(0.4), call(0.4)]
        lagrangian = [
            {"calls": [call(0.3, 0.0), call(0.3, 1.0)]},
            {"calls": [call(0.3, 0.0), call(0.3, 1.0)]},
        ]
        statuses = {
            "route_invariance": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
            "route_a_two_path": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
            "eta_guard_ablation": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
            "lagrangian_mu0_graph_matched": (
                "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            ),
            "lagrangian_guard_ablation": (
                "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            ),
            "lagrangian_separate_interval": (
                "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
            ),
        }
        ledger = comparison_budget_ledger(
            clean_route=0,
            router_bound=router,
            path_bounds=paths,
            eta_bounds=eta,
            lagrangian_bounds=lagrangian,
            statuses=statuses,
            total_wall_budget_seconds=1.0,
        )
        self.assertTrue(ledger["methods"]["unguarded_two_path"]["within_budget"])
        self.assertFalse(ledger["methods"]["lagrangian_grid"]["within_budget"])
        self.assertEqual(
            ledger["methods"]["lagrangian_grid"]["budget_status"],
            "UNKNOWN_BUDGET_EXHAUSTED",
        )

    def test_independent_lagrangian_audit_binds_frozen_grid(self) -> None:
        branch = {
            "status": "CERTIFIED_MARGIN_FILTER",
            "complete": True,
            "lower_bounds": [0.2, 0.3],
            "selected_multipliers": [0.0, 2.0],
            "minimum_lower_bound": 0.2,
            "calls": [
                {
                    "status": "UNKNOWN_RELAXATION",
                    "complete": True,
                    "lower_bounds": [0.2, -0.4],
                    "upper_bounds": [1.0, 1.0],
                    "lagrangian_multiplier": 0.0,
                },
                {
                    "status": "UNKNOWN_RELAXATION",
                    "complete": True,
                    "lower_bounds": [-0.1, 0.3],
                    "upper_bounds": [1.0, 1.0],
                    "lagrangian_multiplier": 2.0,
                },
            ],
        }
        self.assertEqual(
            lagrangian_branch_issues(
                branch,
                expected_multipliers=[0.0, 2.0],
                property_rows=2,
                tolerance=0.1,
            ),
            [],
        )
        branch["calls"][1]["lagrangian_multiplier"] = 3.0
        self.assertIn(
            "Lagrangian call 1 multiplier mismatch",
            lagrangian_branch_issues(
                branch,
                expected_multipliers=[0.0, 2.0],
                property_rows=2,
                tolerance=0.1,
            ),
        )

    def test_top1_rows_cover_every_competitor(self) -> None:
        rows = top1_property_rows(3)
        self.assertEqual(len(rows), 9)
        for row, offset in rows:
            self.assertEqual(offset, 0.0)
            self.assertEqual(row[3], 1.0)
            self.assertEqual(int((row == -1.0).sum()), 1)

    def test_two_path_filter_does_not_require_router_filter(self) -> None:
        result = aggregate_filters(
            clean_route=0,
            router_status="UNKNOWN_RELAXATION",
            path_statuses=["CERTIFIED_MARGIN_FILTER", "CERTIFIED_MARGIN_FILTER"],
            eta_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            attack_prediction_flip=False,
        )
        self.assertEqual(result["route_invariance"], "UNKNOWN")
        self.assertEqual(
            result["route_a_two_path"],
            "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
        )
        self.assertEqual(
            result["portfolio"], "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE"
        )
        self.assertEqual(result["endpoint"], result["portfolio"])

    def test_portfolio_accepts_route_invariance_without_two_path_filter(self) -> None:
        result = aggregate_filters(
            clean_route=0,
            router_status="CERTIFIED_MARGIN_FILTER",
            path_statuses=["CERTIFIED_MARGIN_FILTER", "UNKNOWN_RELAXATION"],
            eta_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            attack_prediction_flip=False,
        )
        self.assertEqual(
            result["route_invariance"],
            "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
        )
        self.assertEqual(result["route_a_two_path"], "UNKNOWN")
        self.assertEqual(result["endpoint"], result["portfolio"])

    def test_portfolio_accepts_lagrangian_guard_filter(self) -> None:
        result = aggregate_filters(
            clean_route=0,
            router_status="UNKNOWN_RELAXATION",
            path_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            eta_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            lagrangian_statuses=[
                "CERTIFIED_MARGIN_FILTER",
                "CERTIFIED_MARGIN_FILTER",
            ],
            attack_prediction_flip=False,
        )
        self.assertEqual(
            result["lagrangian_guard_ablation"],
            "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
        )
        self.assertEqual(result["endpoint"], result["portfolio"])

    def test_concrete_prediction_flip_controls_endpoint(self) -> None:
        result = aggregate_filters(
            clean_route=1,
            router_status="UNKNOWN_RELAXATION",
            path_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            eta_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            attack_prediction_flip=True,
        )
        self.assertEqual(result["endpoint"], "UNSAFE_FULL_FORWARD_REPLAY")

    def test_conflicts_cover_router_and_every_output_filter(self) -> None:
        statuses = {
            "route_invariance": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
            "route_a_two_path": "UNKNOWN",
            "eta_guard_ablation": "UNKNOWN",
            "portfolio": "CERTIFIED_MARGIN_FILTER_NOT_FORMAL_SAFE",
        }
        conflicts = filter_witness_conflicts(
            router_status="CERTIFIED_MARGIN_FILTER",
            statuses=statuses,
            prediction_flip=True,
            route_flip=True,
        )
        self.assertEqual(
            conflicts,
            {
                "router_filter_route_conflict": True,
                "output_filter_prediction_conflict": True,
                "any": True,
            },
        )
        independently_rebuilt = expected_filter_witness_conflicts(
            {
                "clean_route": 0,
                "router_crown": {"status": "CERTIFIED_MARGIN_FILTER"},
                "statuses": statuses,
                "attack": {"prediction_flip": True, "attacked_route": 1},
            }
        )
        self.assertEqual(independently_rebuilt, conflicts)

    def test_independent_crown_status_recomputation(self) -> None:
        record = {
            "complete": True,
            "lower_bounds": [0.1, 0.2],
            "upper_bounds": [0.3, 0.4],
        }
        self.assertEqual(
            expected_crown_status(record, 1e-7), "CERTIFIED_MARGIN_FILTER"
        )
        record["lower_bounds"][0] = -0.1
        self.assertEqual(
            expected_crown_status(record, 1e-7), "UNKNOWN_RELAXATION"
        )

    def test_path_equivalence_uses_registered_absolute_tolerance(self) -> None:
        paths = [object(), object()]
        inputs = torch.zeros(1, 3, 32, 32)
        with patch(
            "act.pipeline.moe.advmoe_two_path.path_adapter_equivalence",
            side_effect=[{"outputs_close": True}, {"outputs_close": True}],
        ) as equivalence:
            rows = evaluate_path_lowering_equivalence(
                paths, inputs, absolute_tolerance=1e-6
            )
        self.assertEqual(len(rows), 2)
        self.assertEqual(equivalence.call_count, 2)
        for call in equivalence.call_args_list:
            self.assertEqual(call.kwargs["atol"], 1e-6)
            self.assertEqual(call.kwargs["rtol"], 0.0)

    def test_static_path_selection_preserves_batch_shape(self) -> None:
        class CountingPath(nn.Module):
            def __init__(self, offset: float):
                super().__init__()
                self.offset = offset
                self.batch_sizes: list[int] = []

            def forward(self, inputs: torch.Tensor) -> torch.Tensor:
                self.batch_sizes.append(inputs.shape[0])
                return inputs[:, :2] + self.offset

        paths = [CountingPath(0.0), CountingPath(10.0)]
        inputs = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        routes = torch.tensor([0, 1, 0])
        selected = select_batched_static_path_logits(paths, inputs, routes)
        torch.testing.assert_close(
            selected,
            torch.tensor([[1.0, 2.0], [13.0, 14.0], [5.0, 6.0]]),
        )
        self.assertEqual(paths[0].batch_sizes, [3])
        self.assertEqual(paths[1].batch_sizes, [3])

    def test_independent_tables_separate_prediction_and_route_flips(self) -> None:
        rows = [
            {
                "epsilon_over_255": 0.5,
                "clean_route": 0,
                "statuses": {
                    "route_invariance": "UNKNOWN",
                    "route_a_two_path": "UNKNOWN",
                    "eta_guard_ablation": "UNKNOWN",
                    "endpoint": "UNSAFE_FULL_FORWARD_REPLAY",
                },
                "attack": {
                    "prediction_flip": True,
                    "attacked_route": 0,
                },
            },
            {
                "epsilon_over_255": 0.5,
                "clean_route": 0,
                "statuses": {
                    "route_invariance": "UNKNOWN",
                    "route_a_two_path": "UNKNOWN",
                    "eta_guard_ablation": "UNKNOWN",
                    "endpoint": "UNKNOWN",
                },
                "attack": {
                    "prediction_flip": False,
                    "attacked_route": 1,
                },
            },
        ]
        table = independent_tables(rows, [0.5])["0.5"]
        self.assertEqual(table["prediction_flip_witnesses"], 1)
        self.assertEqual(table["route_flip_witnesses"], 1)
        self.assertEqual(table["both_flip_witnesses"], 0)

    def test_summary_schema_v1_and_v2_are_validated_separately(self) -> None:
        expected = {
            "samples": 2,
            "route_invariance": {"UNKNOWN": 2},
            "route_a_two_path": {"UNKNOWN": 2},
            "eta_guard_ablation": {"UNKNOWN": 2},
            "lagrangian_guard_ablation": {"UNKNOWN": 2},
            "lagrangian_mu0_graph_matched": {"UNKNOWN": 2},
            "lagrangian_separate_interval": {"UNKNOWN": 2},
            "portfolio": {"UNKNOWN": 2},
            "endpoint": {"UNKNOWN": 2},
            "prediction_flip_witnesses": 1,
            "route_flip_witnesses": 1,
            "both_flip_witnesses": 0,
        }
        v1 = {
            **{key: expected[key] for key in (
                "samples", "route_invariance", "route_a_two_path",
                "eta_guard_ablation", "endpoint",
            )},
            "route_attack_or_prediction_witnesses": 1,
        }
        self.assertEqual(summary_table_issues(v1, expected, 1), [])
        v2 = {**expected}
        self.assertEqual(summary_table_issues(v2, expected, 2), [])
        v2["route_flip_witnesses"] = 0
        self.assertIn(
            "summary table route_flip_witnesses mismatch",
            summary_table_issues(v2, expected, 2),
        )


if __name__ == "__main__":
    unittest.main()
