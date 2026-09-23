"""Tiny native audit mutations and saved-only freeze controls; no real MoE."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.protected_hz_solver import ProtectedHZSolver
from act.front_end.specs import OutKind, OutputSpec
from audit_metamoe_protected_smoke_r1 import evaluation, audit
import metamoe_protected_smoke_r1 as driver
from metamoe_expert_trace import ExpertRecorder
from recent_moe_deployment import sha256


class AuditControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='protected-audit-', dir='/data1/Kane/MOE')
        self.folder = Path(self.tmp.name)/'eval'
        hz = sh.sparse_hz_from_bounds(Bounds(torch.tensor([[-1.]], dtype=torch.float64),
                                           torch.tensor([[1.]], dtype=torch.float64)), frame_id=923)
        spec = OutputSpec(kind=OutKind.LINEAR_LE, c=torch.tensor([[1.], [-1.]], dtype=torch.float64),
                          d=torch.tensor([2., 2.], dtype=torch.float64))
        ProtectedHZSolver(directory=self.folder, time_limit=10).evaluate_spec(
            hz, spec, batch_size=1, n_out=1, input_hz=hz, input_shape=(1, 1))

    def tearDown(self):
        self.tmp.cleanup()

    def modify(self, file, operation):
        path = self.folder/file
        value = json.loads(path.read_text())
        operation(value)
        path.write_text(json.dumps(value))

    def test_complete_saved_audit(self):
        out = evaluation(self.folder, 2)
        self.assertEqual(out['status'], 'certified')
        self.assertEqual(out['native_property_calls'], 2)

    def test_missing_property_rejected(self):
        self.modify('result.json', lambda d: d['metadata']['properties'].pop())
        with self.assertRaisesRegex(ValueError, 'coverage'):
            evaluation(self.folder, 2)

    def test_unknown_base_cannot_certify(self):
        self.modify('result.json', lambda d: d['metadata'].update(base_status='unknown'))
        with self.assertRaisesRegex(ValueError, 'nonvacuity'):
            evaluation(self.folder, 2)

    def test_changed_property_identity(self):
        self.modify('property_identity.json', lambda d: d.update(count=19))
        with self.assertRaisesRegex(ValueError, 'identity'):
            evaluation(self.folder, 2)

    def test_query_model_or_scope_tampering(self):
        self.modify('query_001/request.json', lambda d: d['scope'].update(row=1))
        with self.assertRaisesRegex(ValueError, 'integrity'):
            evaluation(self.folder, 2)

    def test_extra_base_allocation_refused(self):
        self.modify('plan.json', lambda d: d.update(base_fraction=.9))
        with self.assertRaisesRegex(ValueError, 'budget'):
            evaluation(self.folder, 2)

    def test_late_result_not_accepted(self):
        self.modify('result.json', lambda d: d.update(finished_monotonic=1e30))
        with self.assertRaisesRegex(ValueError, 'late'):
            evaluation(self.folder, 2)


class FreezeControls(unittest.TestCase):
    def test_same_input_budget_source_and_no_followup(self):
        cfg = driver.build_config()
        driver.contract(cfg)
        for change in (lambda c: c.update(seconds=600), lambda c: c.update(base_fraction=.2),
                lambda c: c.update(automatic_followup=True), lambda c: c.update(epsilon=1/255),
                lambda c: c['files'].update({'act/back_end/solver/solver_hz.py': 'changed'}),
                lambda c: c['requests'][0].update(label=-1), lambda c: c['roster'].append(['mnist_0', 'act'])):
            bad = copy.deepcopy(cfg)
            change(bad)
            with self.assertRaises(ValueError):
                driver.contract(bad)


class OuterFlowControls(unittest.TestCase):
    def fixture(self, mode):
        tmp = tempfile.TemporaryDirectory(prefix='protected-flow-', dir='/data1/Kane/MOE')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = driver.build_config()
        cfg['output_root'] = str(root/'run')
        path = root/'config.json'
        path.write_text(json.dumps(cfg))
        def fake(command, cwd, folder, seconds, rss):
            folder.mkdir()
            rec = ExpertRecorder(folder/'trace.jsonl', time.monotonic(), driver.identity(path))
            if mode == 'COMPLETED':
                rec.emit('WORKER_COMPLETE')
                (folder/'result.json').write_text(json.dumps({'status': 'UNKNOWN', 'request_id': 'mnist_0',
                    'arm': 'act', 'config_sha256': sha256(path), 'worker_seconds': .001,
                    'label': cfg['requests'][0]['label'], 'tensor_file_sha256': sha256(cfg['requests'][0]['tensor_file'])}))
            else:
                rec.emit('BEGIN', name='partial', parent=None, arguments={})
                (folder/'result.json').write_text('{"status":')
            rec.close()
            for stream in ('stdout', 'stderr'):
                (folder/f'{stream}.txt').write_text('synthetic')
            receipt = {'status': mode, 'command': command, 'deadline_seconds': seconds,
                'execution_including_preflight_seconds': .05, 'total_with_postflight_seconds': .051,
                'peak_sampled_group_rss_bytes': 1000, 'group_rss_limit_bytes': rss,
                'exit_code': 0 if mode == 'COMPLETED' else -9, 'error': None,
                'rss_poll_seconds': .05, 'postflight_in_execution_budget': False,
                'rss_is_sampled_not_instantaneous_cap': True, 'receipt_own_write_excluded_from_this_clock': True,
                'stdout_sha256': sha256(folder/'stdout.txt'), 'stderr_sha256': sha256(folder/'stderr.txt')}
            (folder/'receipt.json').write_text(json.dumps(receipt))
            time.sleep(.06)
            return receipt
        with patch.object(driver, 'validate'), patch.object(driver, 'supervise', side_effect=fake) as calls:
            driver.run(path)
            self.assertEqual(calls.call_count, 1)
        return path, Path(cfg['output_root'])

    def review(self, path):
        with patch('audit_metamoe_protected_smoke_r1.validate'):
            return audit(path)

    def test_single_completed_call_no_auto_followup_or_overwrite(self):
        path, _ = self.fixture('COMPLETED')
        result = self.review(path)
        self.assertEqual(result['audit'], 'PASS')
        self.assertFalse(result['opens_formal_cohort'])
        with patch.object(driver, 'validate'), patch.object(driver, 'supervise') as call:
            with self.assertRaises(FileExistsError):
                driver.run(path)
            call.assert_not_called()

    def test_outer_timeout_dominates_partial_candidate(self):
        path, _ = self.fixture('TIMEOUT')
        result = self.review(path)
        self.assertEqual(result['row']['status'], 'TIMEOUT')
        self.assertTrue(result['row']['result_parse_error'])
        self.assertEqual(len(result['trace']['open_span_ids']), 1)

    def test_postflight_inventory_cannot_be_unaccounted(self):
        path, root = self.fixture('COMPLETED')
        (root/'mnist_0_act/protected').mkdir()
        (root/'mnist_0_act/protected/unregistered').write_text('unbound')
        with self.assertRaisesRegex(ValueError, 'inventory'):
            self.review(path)


if __name__ == '__main__':
    unittest.main()
