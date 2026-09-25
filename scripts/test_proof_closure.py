"""Accounting controls only; no neural execution, propagation or solving."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('closure', ROOT / 'scripts/summarize_proof_closure.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Closure(unittest.TestCase):
    def setUp(self):
        self.data = m.load_sources()

    def reject(self, change):
        data = copy.deepcopy(self.data)
        change(data)
        with self.assertRaises((ValueError, KeyError)):
            m.build(data)

    def test_recount_and_boundaries(self):
        d = m.build(self.data)
        self.assertEqual(d['real_summary']['calls'], 7)
        self.assertEqual(d['real_summary']['distinct_inputs'], 4)
        self.assertEqual(d['real_summary']['timeouts'], 6)
        self.assertEqual(d['real_summary']['resource_limits'], 1)
        self.assertEqual([r['online_discharged_duties'] for r in d['real_attempts']], [0]*6+[225])
        self.assertEqual(d['last_real_frontier']['unclosed_duties'], 27)
        self.assertEqual(d['last_real_frontier']['retained_potential_pairs'], [[4,7],[5,7],[6,7]])
        self.assertAlmostEqual(d['last_real_frontier']['time_before_work_deadline_at_return'], 15.0521659274)
        self.assertFalse(d['experiment_launched'])
        self.assertEqual(d['decision'], 'STOP_TIMING_LINE_NO_AUTOMATIC_REAL_FREEZE')

    def test_offline_is_not_online(self):
        d = m.build(self.data)['real_attempts']
        self.assertEqual([(r['offline_checked_excluded_pairs'], r['online_discharged_duties']) for r in d[-3:]],
                         [(24, 0), (25, 0), (25, 225)])
        for key, value in [('route_discharged_duties', 225), ('route_receipt_published', True),
                           ('route_check_phase_completed', True)]:
            self.reject(lambda d: d['residual']['calls'][0]['evidence'].update({key: value}))

    def test_missing_duplicate_wrong_input(self):
        self.reject(lambda d: d['parse']['calls'].pop())
        self.reject(lambda d: d['parse']['calls'].__setitem__(1, copy.deepcopy(d['parse']['calls'][0])))
        self.reject(lambda d: d['frontier'].update(dataset_index=98))
        self.reject(lambda d: d['residual'].update(required_per_call=27))

    def test_no_unearned_bounds_or_completion(self):
        for key, value in [('independently_checked_output_bounds', 27), ('positive_output_bounds', 27),
                           ('construction_published', True), ('complete_output_positive_proof', True),
                           ('native_float_proof', True), ('route_changing_established', True),
                           ('duties_without_positive_checked_evidence', 0), ('native_lp_calls_started', 27)]:
            self.reject(lambda d: d['residual']['calls'][1]['evidence'].update({key: value}))

    def test_cost_and_budget(self):
        for key, value in [('stage_seconds', -1), ('budget_seconds', 301), ('overhead_seconds', None),
                           ('end_to_end_seconds', float('nan')), ('complete_output_positive_proof', True)]:
            self.reject(lambda d: d['scoped']['cost'].update({key: value}))
        self.reject(lambda d: d['parse']['calls'][0].update(return_seconds=0))
        self.reject(lambda d: d['parse']['calls'][0]['stages'][1].update(phase='propose'))

    def test_incomplete_coverage_and_wrong_source(self):
        self.reject(lambda d: d['residual']['calls'][1]['evidence']['retained_pair_list'].pop())
        self.reject(lambda d: d['residual']['files']['shared/source.json'].update(sha256='0'*64))
        self.reject(lambda d: d['residual']['files'].update({'shared/construction.json': {}}))

    def test_synthetic_cannot_become_output_evidence(self):
        for key, value in [('lower_bounds_checked', 27), ('output_obligations', 0),
                           ('complete_output_positive_proof', True), ('source_sha256', '0'*64)]:
            self.reject(lambda d: d['upstream']['rows'][2]['profile']['source_check'].update({key: value}))
        self.reject(lambda d: d['upstream'].update(real_requests=4))
        d = m.build(self.data)
        self.assertLess(d['synthetic_cost_pairs'][0]['readonly_minus_direct_seconds'], 0)
        self.assertGreater(d['synthetic_cost_pairs'][1]['readonly_minus_direct_seconds'], 0)

    def test_strict_json(self):
        for raw in ('{"x":0,"x":1}', '{"x":NaN}', '{"x":Infinity}'):
            with self.assertRaises(ValueError):
                m.strict_json(raw)

    def test_isolated_relocated_rebuild_and_identity(self):
        with tempfile.TemporaryDirectory(prefix='proof-closure-', dir=ROOT.parent) as directory:
            root = Path(directory)
            for name in [*(v[0] for v in m.SOURCES.values()), m.OUTPUT, 'scripts/summarize_proof_closure.py']:
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((ROOT / name).read_bytes())
            done = subprocess.run([sys.executable, '-I', '-S', str(root/'scripts/summarize_proof_closure.py'), '--check'],
                                  cwd=root, capture_output=True, text=True, timeout=15)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(json.loads(done.stdout)['calls'], 7)
            source = root / m.SOURCES['scoped'][0]
            source.write_bytes(source.read_bytes() + b' ')
            with self.assertRaisesRegex(ValueError, 'archive identity'):
                m.load_sources(root)


if __name__ == '__main__':
    unittest.main()
