from __future__ import annotations

import unittest
from unittest.mock import patch

import torch
from torch import nn

from act.pipeline.moe.advmoe_two_path import (
    aggregate_filters,
    evaluate_path_lowering_equivalence,
    filter_witness_conflicts,
    select_batched_static_path_logits,
    top1_property_rows,
)
from act.pipeline.moe.audit_advmoe_two_path import (
    expected_crown_status,
    expected_filter_witness_conflicts,
    independent_tables,
    summary_table_issues,
)


class AdvMoeTwoPathTests(unittest.TestCase):
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
