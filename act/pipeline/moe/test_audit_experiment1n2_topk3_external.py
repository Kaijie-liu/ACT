"""Unit tests for the external N2 artifact audit."""

from __future__ import annotations

import unittest

from act.pipeline.moe.audit_experiment1n2_topk3_external import (
    route_structure_issues,
)


def _row():
    return {
        "sample_rank": 0,
        "status": "SAFE",
        "reason": "SAFE_WEIGHTED_RANGE",
        "route_set_enumeration_exact": True,
        "unresolved_route_sets": [],
        "exact_feasible_unordered_top3_set_count": 1,
        "products_per_property": 2,
        "route_sets": [
            {
                "route_set": [0, 2, 5],
                "status": "SAFE",
                "reason": "SAFE_WEIGHTED_RANGE",
                "property_rows": [
                    {
                        "property_index": index,
                        "product_count": 2,
                        "status": "SAFE",
                        "full_model_witness_valid": False,
                    }
                    for index in range(9)
                ],
            }
        ],
    }


class ExternalN2AuditTest(unittest.TestCase):
    def test_accepts_complete_two_product_safe_route(self):
        self.assertEqual(route_structure_issues(_row()), [])

    def test_rejects_product_count_mutation(self):
        row = _row()
        row["route_sets"][0]["property_rows"][3]["product_count"] = 1
        self.assertTrue(
            any("two products" in issue for issue in route_structure_issues(row))
        )

    def test_rejects_relaxation_unsafe_without_replay(self):
        row = _row()
        prop = row["route_sets"][0]["property_rows"][0]
        prop["status"] = "UNSAFE"
        prop["full_model_witness_valid"] = False
        self.assertTrue(
            any("promotes relaxation UNSAFE" in issue for issue in route_structure_issues(row))
        )

    def test_accepts_complete_gate_elimination_route(self):
        row = _row()
        row["reason"] = "SAFE_GATE_ELIMINATION"
        row["route_sets"][0] = {
            "route_set": [0, 2, 5],
            "status": "SAFE",
            "reason": "SAFE_GATE_ELIMINATION",
            "property_rows": [],
            "gate_elimination": [
                {
                    "expert": expert,
                    "status": "SAFE",
                    "solver_status": "certified",
                    "solver_reason": "expanded_violations_infeasible",
                }
                for expert in (0, 2, 5)
            ],
        }
        self.assertEqual(route_structure_issues(row), [])

    def test_accepts_early_full_forward_unsafe_prefix(self):
        row = _row()
        row["status"] = "UNSAFE"
        row["reason"] = "UNSAFE_FULL_FORWARD_FALLBACK"
        row["full_model_witness_valid"] = True
        route = row["route_sets"][0]
        route["status"] = "UNSAFE"
        route["reason"] = "UNSAFE_FULL_FORWARD_FALLBACK"
        route["full_model_witness_valid"] = True
        route["property_rows"] = [
            {
                "property_index": 0,
                "product_count": 2,
                "status": "UNSAFE",
                "full_model_witness_valid": True,
            }
        ]
        self.assertEqual(route_structure_issues(row), [])


if __name__ == "__main__":
    unittest.main()
