"""Controls for derived accounting; no network/solver queries."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from cohort_analysis.archive import ROOT, cost_record, local_precheck, sha, write_new


class Accounting(unittest.TestCase):
    def setUp(self):
        self.request = {'properties': [{}, {}, {}]}
        self.manifest = {'routes': {'feasible': [[0, 1]]}, 'positive_threshold': '1/10', 'generation_complete': True}
        self.value = {'required_obligations': 3, 'positive_obligations': 1,
                      'missing_obligations': 1, 'nonpositive_obligations': 1,
                      'route_pairs': [[0, 1]], 'status': 'UNKNOWN_MISSING_EVIDENCE',
                      'obligations': [
                          {'pair': [0, 1], 'property_index': 0, 'lower_bound': '1/3', 'state': 'CHECKED_RATIONAL_POSITIVE'},
                          {'pair': [0, 1], 'property_index': 1, 'lower_bound': '-2/3', 'state': 'CHECKED_NONPOSITIVE_OR_BELOW_THRESHOLD'},
                          {'pair': [0, 1], 'property_index': 2, 'lower_bound': None, 'state': 'MISSING_EVIDENCE'}]}

    def check(self, value=None):
        return local_precheck(value or self.value, self.request, self.manifest)

    def test_missing_and_nonpositive_not_unsafe(self):
        result = self.check()
        self.assertEqual((result['positive'], result['nonpositive'], result['missing']), (1, 1, 1))
        self.assertEqual(result['status'], 'UNKNOWN_MISSING_EVIDENCE')

    def test_missing_duplicate_and_property_mutations(self):
        for mutation in ('delete', 'duplicate', 'property'):
            v = deepcopy(self.value)
            if mutation == 'delete': v['obligations'].pop()
            elif mutation == 'duplicate': v['obligations'][1] = v['obligations'][0]
            else: v['obligations'][0]['property_index'] = 9
            with self.assertRaises(ValueError): self.check(v)

    def test_count_state_and_route_mutations(self):
        for mutation in ('count', 'status', 'state', 'route'):
            v = deepcopy(self.value)
            if mutation == 'count': v['positive_obligations'] = 3
            elif mutation == 'status': v['status'] = 'CHECKED_CONDITIONAL'
            elif mutation == 'state': v['obligations'][1]['state'] = 'CHECKED_RATIONAL_POSITIVE'
            else: v['route_pairs'] = [[1, 2]]
            with self.assertRaises(ValueError): self.check(v)

    def test_exact_threshold_is_not_positive(self):
        v = deepcopy(self.value)
        v['obligations'][0].update(lower_bound='1/10', state='CHECKED_NONPOSITIVE_OR_BELOW_THRESHOLD')
        v.update(positive_obligations=0, nonpositive_obligations=2)
        self.assertEqual(self.check(v)['nonpositive'], 2)

    def test_phase_censoring_and_unmeasured_are_not_zero(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe') as name:
            d = Path(name); write_new(d/'a.json', {'x': 1})
            t = {'stages': {'capture': {'state': 'COMPLETED', 'elapsed_seconds': 10},
                            'propose': {'state': 'BUDGET_EXHAUSTED', 'elapsed_seconds': 20}},
                 'censored_phase': None, 'wall_seconds': 31, 'dataset_index': 1,
                 'status': 'TIMEOUT', 'outer_timeout': False, 'artifact_sha256': {'a.json': sha(d/'a.json')}}
            r = cost_record(d, t, {'proof_size': None})
            self.assertEqual(r['stop_phase'], 'propose')
            self.assertEqual(r['unassigned_outer_seconds'], 1)
            self.assertIsNone(r['native_solver_exclusive_seconds'])
            self.assertIsNone(r['proposal_wrapper_seconds'])
            self.assertNotIn('propose', r['completed_phase_seconds'])
            t['stages'].pop('propose')
            t['censored_phase'] = {'phase': 'precheck', 'observed_phase_window_seconds': 20}
            self.assertEqual(cost_record(d, t, {'proof_size': None})['unassigned_outer_seconds'], 1)
            t['censored_phase']['observed_phase_window_seconds'] = 40
            with self.assertRaises(ValueError): cost_record(d, t, {'proof_size': None})

    def test_no_overwriting_archive(self):
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe') as name:
            p = Path(name)/'review.json'; write_new(p, {'status': 'PASS'})
            with self.assertRaises(FileExistsError): write_new(p, {'status': 'FAIL'})

    def test_reserve_can_expire_between_preflight_and_grant(self):
        # Analytic clock control, not a new real query or a production repair.
        from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired
        now = [219.98]
        budget = EvidenceBudget(0, clock=lambda: now[0])
        self.assertGreater(budget.deadline-budget.clock(), 80.01)
        now[0] += .03  # source decoding/construction/checking before grant
        with self.assertRaises(EvidenceBudgetExpired): budget.grant(60, 80)
        self.assertGreater(budget.remaining(2), 77)


if __name__ == '__main__':
    unittest.main()
