import copy
import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from act.pipeline.moe.common_fact_snapshot import check_snapshot, digest, fact_view, publish_snapshot
from act.pipeline.moe.test_route_complexity_schedule import model, run
from act.pipeline.moe import route_complexity_schedule as scheduler


class SnapshotTests(unittest.TestCase):
    def test_legacy_reference_preserves_registered_settings(self):
        configs = Path(__file__).parent/'configs'
        raw = (configs/'monolithic_legacy_reference_v1.json').read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(),
                         'bb05702ebcd61ae47db28f2d35fe03d2e84803b23fa3e2cc4328a954e00ff756')
        from act.pipeline.moe.paired_followup import method_config
        original = method_config(json.loads((configs/'staged_verifier_v1.json').read_text()), 'monolithic_f0', 300.)
        self.assertEqual(json.loads(raw), original)
        self.assertNotIn('route_complexity_schedule', original)
        self.assertNotIn('scoped_proof_reuse', original)

    def capture(self):
        values = []
        report = run(model(((2., 0.), (3., 0.), (4., 0.))), common_fact_callback=values.append)
        self.assertEqual(len(values), 1)
        return values[0], report

    def test_snapshot_matches_final_and_is_detached(self):
        value, report = self.capture()
        check_snapshot(value, expected_identity=report.evidence['identity'], evidence=report.evidence)
        original = copy.deepcopy(value)
        report.evidence['tier1']['branches'][0]['proof_output_bounds']['lower'][0] = -99
        self.assertEqual(value, original)
        with self.assertRaisesRegex(ValueError, 'facts mismatch'):
            check_snapshot(value, evidence=report.evidence)

    def test_mutations_rejected_even_with_rehashed_payload(self):
        value, _ = self.capture()
        for field in ('scope', 'pairs', 'branches', 'count', 'time', 'config'):
            bad = copy.deepcopy(value); b = bad['payload']
            if field == 'scope': b['scope']['model_state'] = {}
            elif field == 'pairs': b['feasible_route_sets'].append([0, 1])
            elif field == 'branches': b['branches'].pop()
            elif field == 'count': b['available_fact_count'] += 1
            elif field == 'time': b['completion_elapsed_seconds'] = -1
            else: b['config']['comparison_method'] = 'wrong'
            bad['payload_sha256'] = digest(b)
            with self.assertRaises(ValueError, msg=field): check_snapshot(bad)
        bad = copy.deepcopy(value); bad['payload_sha256'] = 'wrong'
        with self.assertRaises(ValueError): check_snapshot(bad)

    def test_common_across_arms_and_no_clobber(self):
        value, _ = self.capture(); other = []
        run(model(((2., 0.), (3., 0.), (4., 0.))), 'monolithic_f0', common_fact_callback=other.append)
        self.assertEqual(fact_view(value), fact_view(other[0]))
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            path = Path(tmp)/'prelude.json'
            publish_snapshot(path, value)
            before = path.read_bytes()
            with self.assertRaises(FileExistsError): publish_snapshot(path, other[0])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(list(Path(tmp).glob('*.pending-*')))

    def test_before_solver_and_survives_hard_kill(self):
        # A real SIGKILL immediately after durable publication: no final package
        # can exist, yet the observation remains parseable and hash-verifiable.
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            path = Path(tmp)/'prelude.json'
            code = '''
import os, signal, sys
from unittest.mock import patch
from act.pipeline.moe.common_fact_snapshot import publish_snapshot
from act.pipeline.moe.test_route_complexity_schedule import model, run
from act.pipeline.moe import route_complexity_schedule as scheduler
def emit(value):
    publish_snapshot(sys.argv[1], value)
    os.kill(os.getpid(), signal.SIGKILL)
with patch.object(scheduler, '_solve_output', side_effect=AssertionError('too early')):
    run(model(((-.2,0.),(1.,0.),(2.,0.))), common_fact_callback=emit)
'''
            child = subprocess.run([sys.executable, '-c', code, str(path)], capture_output=True, timeout=30)
            self.assertEqual(child.returncode, -signal.SIGKILL, child.stderr.decode())
            self.assertEqual(check_snapshot(json.loads(path.read_text()))['status'], 'PASS')

    def test_incomplete_analysis_emits_nothing(self):
        from dataclasses import replace
        original = scheduler.analyze_topk_sets
        values = []
        with patch.object(scheduler, 'analyze_topk_sets', side_effect=lambda *a, **kw: replace(original(*a, **kw), exact=False)):
            report = run(model(((2.,0.),(3.,0.))), common_fact_callback=values.append)
        self.assertEqual(values, [])
        self.assertEqual(report.status, 'UNKNOWN')


if __name__ == '__main__': unittest.main()
