import unittest

from act.pipeline.moe.audit_experiment1n1_engineering import _safe_structure_issues


class AuditExperiment1N1EngineeringTests(unittest.TestCase):
    def test_safe_requires_every_segment_safe(self):
        row = {
            "n1": {
                "status": "SAFE",
                "feasible_pairs": [[0, 1]],
                "pairs": [
                    {
                        "pair": [0, 1],
                        "status": "SAFE",
                        "property_rows": [
                            {
                                "property_index": 0,
                                "status": "SAFE",
                                "reason": "SAFE_WEIGHTED_SEGMENTED",
                                "reused_parent": False,
                                "segments": [
                                    {"decision": {"status": "SAFE"}},
                                    {"decision": {"status": "UNKNOWN"}},
                                ],
                            }
                        ],
                    }
                ],
            }
        }
        issues = _safe_structure_issues(row)
        self.assertIn(
            "new segmented SAFE property contains a non-SAFE segment",
            issues,
        )

    def test_reused_safe_property_needs_no_segment_record(self):
        row = {
            "n1": {
                "status": "SAFE",
                "feasible_pairs": [[0, 1]],
                "pairs": [
                    {
                        "pair": [0, 1],
                        "status": "SAFE",
                        "property_rows": [
                            {
                                "property_index": 0,
                                "status": "SAFE",
                                "reused_parent": True,
                            }
                        ],
                    }
                ],
            }
        }
        self.assertEqual(_safe_structure_issues(row), [])

    def test_reused_safe_pair_propagates_to_nested_properties(self):
        row = {
            "n1": {
                "status": "SAFE",
                "feasible_pairs": [[0, 1]],
                "pairs": [
                    {
                        "pair": [0, 1],
                        "status": "SAFE",
                        "reused_parent": True,
                        "property_rows": [
                            {
                                "property_index": 0,
                                "status": "SAFE",
                                "reason": "SAFE_WEIGHTED_RANGE",
                                "segments": [],
                            }
                        ],
                    }
                ],
            }
        }
        self.assertEqual(_safe_structure_issues(row), [])

    def test_safe_must_cover_every_feasible_pair(self):
        row = {
            "n1": {
                "status": "SAFE",
                "feasible_pairs": [[0, 1], [0, 2]],
                "pairs": [
                    {
                        "pair": [0, 1],
                        "status": "SAFE",
                        "property_rows": [
                            {
                                "property_index": 0,
                                "status": "SAFE",
                                "reused_parent": True,
                            }
                        ],
                    }
                ],
            }
        }
        self.assertIn(
            "SAFE row does not cover every feasible pair exactly once",
            _safe_structure_issues(row),
        )

    def test_safe_must_cover_every_property(self):
        row = {
            "n1": {
                "status": "SAFE",
                "feasible_pairs": [[0, 1]],
                "pairs": [
                    {
                        "pair": [0, 1],
                        "status": "SAFE",
                        "property_rows": [
                            {
                                "property_index": 0,
                                "status": "SAFE",
                                "reused_parent": True,
                            }
                        ],
                    }
                ],
            }
        }
        self.assertIn(
            "SAFE pair [0, 1] does not cover every property row",
            _safe_structure_issues(row, expected_property_rows=2),
        )


if __name__ == "__main__":
    unittest.main()
