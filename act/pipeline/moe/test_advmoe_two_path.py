from __future__ import annotations

import unittest

from act.pipeline.moe.advmoe_two_path import (
    aggregate_filters,
    top1_property_rows,
)
from act.pipeline.moe.audit_advmoe_two_path import expected_crown_status


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

    def test_concrete_prediction_flip_controls_endpoint(self) -> None:
        result = aggregate_filters(
            clean_route=1,
            router_status="UNKNOWN_RELAXATION",
            path_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            eta_statuses=["UNKNOWN_RELAXATION", "UNKNOWN_RELAXATION"],
            attack_prediction_flip=True,
        )
        self.assertEqual(result["endpoint"], "UNSAFE_FULL_FORWARD_REPLAY")

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


if __name__ == "__main__":
    unittest.main()
