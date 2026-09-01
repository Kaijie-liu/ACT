import unittest

from act.pipeline.moe.exact_big_m_engineering import (
    _support_side_fell_back,
    paired_audit_issues,
    summarize_conditions,
)


def _condition(rank, mode, *, candidates=(0, 1), selectors=4, seconds=2.0):
    return {
        "sample_rank": rank,
        "mode": mode,
        "complete": True,
        "candidate_experts": list(candidates),
        "selector_binaries": selectors,
        "mip_nodes": selectors,
        "total_seconds": seconds,
        "feasibility_seconds": seconds / 2,
    }


def _expert(rank, mode, expert, value):
    return {
        "sample_rank": rank,
        "mode": mode,
        "expert": expert,
        "big_m_support_mode": mode,
        "big_m": {str((expert + 1) % 8): value},
    }


class ExactBigMEngineeringTests(unittest.TestCase):
    def test_support_status_accounting_matches_backend_labels(self):
        self.assertFalse(_support_side_fell_back("milp_optimal"))
        self.assertFalse(_support_side_fell_back("milp_optimal_capped_by_fast"))
        self.assertFalse(_support_side_fell_back("lp_optimal"))
        self.assertTrue(_support_side_fell_back("fallback_generator"))

    def test_summary_is_paired_not_solved_only(self):
        rows = [
            _condition(3, "fast", selectors=7, seconds=2.0),
            _condition(3, "exact", selectors=2, seconds=3.0),
        ]
        summary = summarize_conditions(rows)
        self.assertEqual(summary["paired_ranks"], 1)
        self.assertEqual(summary["total_selector_binary_reduction"], 5)
        self.assertAlmostEqual(summary["median_exact_over_fast_total_time"], 1.5)

    def test_audit_accepts_nonincreasing_exact_m(self):
        conditions = [_condition(4, "fast"), _condition(4, "exact", selectors=2)]
        experts = []
        for expert in range(8):
            experts.append(_expert(4, "fast", expert, 3.0))
            experts.append(_expert(4, "exact", expert, 2.0))
        self.assertEqual(
            paired_audit_issues(conditions, experts, expected_ranks=1), []
        )

    def test_audit_rejects_semantic_or_bound_regression(self):
        conditions = [
            _condition(5, "fast", candidates=(0, 1)),
            _condition(5, "exact", candidates=(0, 2), selectors=9),
        ]
        experts = []
        for expert in range(8):
            experts.append(_expert(5, "fast", expert, 1.0))
            experts.append(_expert(5, "exact", expert, 2.0))
        issues = paired_audit_issues(conditions, experts, expected_ranks=1)
        self.assertTrue(any("semantics differ" in issue for issue in issues))
        self.assertTrue(any("selector width increased" in issue for issue in issues))
        self.assertTrue(any("exact M increased" in issue for issue in issues))


if __name__ == "__main__":
    unittest.main()
