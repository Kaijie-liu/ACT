import copy
import unittest
from act.pipeline.moe.analyze_schedule_complementarity import stopping_location


class PhaseTests(unittest.TestCase):
    def test_terminal_label_alone_is_not_candidate_failure(self):
        e = {'verdict': {'reason': 'UNKNOWN_SOLVER_LIMIT'},
             'route_coverage': {'candidate_set_minimal': True, 'route_sets_exact': True, 'coverage_complete': True},
             'tier1': {'branches': [{'solver_reason': 'violation_region_undecided'}]}}
        self.assertEqual(stopping_location(e), 'TIER1_EXPERT_VIOLATION_REGION_UNDECIDED')
        bad = copy.deepcopy(e); bad['tier1']['branches'] = []
        self.assertEqual(stopping_location(bad), 'UNKNOWN_SOLVER_LIMIT_PHASE_UNRESOLVED')
        bad['route_coverage']['route_sets_exact'] = False
        self.assertEqual(stopping_location(bad), 'INCOMPLETE_ROUTE_ANALYSIS')
        self.assertEqual(stopping_location(None), 'UNAVAILABLE_PACKAGE')


if __name__ == '__main__': unittest.main()
