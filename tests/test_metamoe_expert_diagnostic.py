"""Synthetic observation/terminal controls; no large model or native solve."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import metamoe_expert_diagnostic as driver
from metamoe_expert_trace import ExpertRecorder, TraceFailure, install, restore
from audit_conv_f0_timing import check_trace
from audit_metamoe_expert_diagnostic import audit, diagnose
from recent_moe_deployment import sha256


class TraceControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='meta-trace-control-', dir='/data1/Kane/MOE')
        self.root = Path(self.tmp.name)
        self.path = self.root/'trace.jsonl'
        self.rec = ExpertRecorder(self.path, time.monotonic(), {'control': True})

    def tearDown(self):
        self.rec.close()
        self.tmp.cleanup()

    def trace(self, killed=False):
        if not killed:
            self.rec.emit('WORKER_COMPLETE')
        return check_trace(self.path, 100, {'control': True}, killed=killed)

    def test_original_arguments_return_identity_and_one_call(self):
        calls, token = [], object()
        def fn(a, *, timelimit):
            calls.append((a, timelimit))
            return token
        result = self.rec.wrap(fn, 'fn')(token, timelimit=30.)
        self.assertIs(result, token)
        self.assertEqual(calls, [(token, 30.)])
        self.assertEqual(self.trace()['spans'][0]['arguments']['timelimit'], 30.)

    def test_exception_is_recorded_and_rethrown_unchanged(self):
        exc = RuntimeError('synthetic failure')
        def fail():
            raise exc
        with self.assertRaises(RuntimeError) as caught:
            self.rec.wrap(fail, 'fail')()
        self.assertIs(caught.exception, exc)
        self.assertEqual(self.trace(True)['spans'][0]['exception'], 'RuntimeError')

    def test_base_unknown_metadata_and_no_false_property_query(self):
        from act.util.stats import VerifyResult, VerifyStatus
        def early():
            return [VerifyResult(VerifyStatus.UNKNOWN, metadata={'reason': 'base_unknown'})]
        self.rec.wrap(early, 'HZSolver.evaluate_spec')()
        result = diagnose(self.trace())['expert_evaluations'][0]
        self.assertEqual(result['result']['verify_results'][0]['metadata']['reason'], 'base_unknown')
        self.assertEqual(result['property_query_count'], 0)

    def test_partial_trace_identity_and_tampering(self):
        self.rec.emit('BEGIN', name='open', parent=None, arguments={})
        with self.assertRaises(ValueError):
            self.trace()
        raw = self.path.read_bytes()
        self.path.write_bytes(raw+b'{"partial":')
        view = self.trace(True)
        self.assertEqual(view['partial_tail_bytes'], len(b'{"partial":'))
        self.assertTrue(view['spans'][0]['right_censored'])
        with self.assertRaises(ValueError):
            check_trace(self.path, 100, {'control': False}, killed=True)
        self.path.write_bytes(raw.replace(b'open', b'evil'))
        with self.assertRaises(ValueError):
            self.trace(True)

    def test_logging_failure_not_swallowed_as_solver_unknown(self):
        self.rec.close()
        with self.assertRaises(TraceFailure):
            self.rec.wrap(lambda: None, 'closed')()

    def test_installation_reaches_aliases_and_restores(self):
        from act.back_end.moe import route_a, class_separated_top1
        from act.back_end.solver import solver_hz
        from act.back_end.hybridz_tf import HybridzTF
        originals = (route_a.verify_once, solver_hz.milp, solver_hz.HZSolver.evaluate_spec, HybridzTF.apply)
        patches = install(self.rec)
        try:
            self.assertIs(route_a.verify_once.__wrapped__, originals[0])
            self.assertIs(solver_hz.milp.__wrapped__, originals[1])
            self.assertIs(solver_hz.HZSolver.evaluate_spec.__wrapped__, originals[2])
            self.assertIs(HybridzTF.apply.__wrapped__, originals[3])
            # The frontend imports support locally at call time, not at module load.
            from act.back_end.solver.solver_hz import hz_support_bounds
            self.assertIs(hz_support_bounds, solver_hz.hz_support_bounds)
            self.assertTrue(hasattr(hz_support_bounds, '__wrapped__'))
        finally:
            restore(patches)
        self.assertEqual((route_a.verify_once, solver_hz.milp, solver_hz.HZSolver.evaluate_spec, HybridzTF.apply), originals)
        self.trace()

    def native_control(self, x, status=1, error=None, expired=False):
        import numpy as np
        from scipy import sparse
        from act.back_end.solver import solver_hz as sh
        model = sh._HZMILP(np.zeros(1), sparse.csr_matrix([[1.]]),
            sparse.csr_matrix((0, 1)), np.zeros(0), np.zeros(0),
            np.array([0.]), np.array([1.]), np.ones(1, dtype=np.int32), 0, 1)
        proposed = SimpleNamespace(status=status, success=False, message='Time limit reached',
            x=np.array(x) if x is not None else None, mip_node_count=2, mip_gap=float('inf'))
        calls = []
        def native(c, *, integrality, bounds, constraints, options):
            calls.append(options.copy())
            if error:
                raise error
            return proposed
        native.__module__, native.__name__ = 'scipy.optimize', 'milp'
        wrapped = self.rec.wrap(native, 'solver.milp')
        original_check = sh._valid_milp_point
        def evaluate_spec():
            return self.rec.wrap(sh._solve_hz_feasibility, 'solver._solve_hz_feasibility')(
                model, time.monotonic()+(-1 if expired else 30))
        with patch.object(sh, 'milp', wrapped), patch.object(sh, '_valid_milp_point',
                self.rec.wrap(original_check, 'solver._valid_milp_point')):
            result = self.rec.wrap(evaluate_spec, 'HZSolver.evaluate_spec')()
        view = diagnose(self.trace())['expert_evaluations'][0]['queries'][0]
        return result, calls, view

    def test_status_one_valid_incumbent_stays_feasible(self):
        result, calls, view = self.native_control([1.])
        self.assertEqual(result.status, 'feasible')
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]['mip_rel_gap'], 0.)
        self.assertEqual(view['native'][0]['result']['status'], 1)
        self.assertIn('NATIVE_LIMIT_REPORTED', view['flags'])
        self.assertEqual(view['incumbent_checks'][0]['value'], True)

    def test_invalid_incumbent_preserved_as_unknown(self):
        result, _, view = self.native_control([.5])
        self.assertEqual(result.status, 'unknown')
        self.assertIn('INCUMBENT_REJECTED_BY_EXISTING_FLOAT_POLICY', view['flags'])

    def test_native_exception_and_exhausted_deadline_distinct(self):
        result, calls, view = self.native_control(None, error=RuntimeError('native error'))
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(len(calls), 1)
        self.assertIn('NATIVE_EXCEPTION', view['flags'])

    def test_exhausted_deadline_has_no_native_call(self):
        result, calls, view = self.native_control(None, expired=True)
        self.assertEqual(result.status, 'unknown')
        self.assertEqual(calls, [])
        self.assertIn('LOCAL_DEADLINE_EXHAUSTED_WITHOUT_NATIVE_CALL', view['flags'])


class ProtocolControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = driver.build_config()  # Saved hashes/environment only.

    def test_freeze_preserves_parent_and_rejects_drift(self):
        driver.contract(self.cfg)
        changes = [lambda c: c.update(seconds=301), lambda c: c.update(margin=0.),
            lambda c: c.update(epsilon=4/255), lambda c: c.update(automatic_followup=True),
            lambda c: c['requests'][0].update(label=-1), lambda c: c.update(arms=['author']),
            lambda c: c.update(roster=[['mnist_0', 'act']]*2),
            lambda c: c['files'].update({'act/back_end/solver/solver_hz.py': 'rebind'}),
            lambda c: c['hybridz'].update(sparse_representation_bytes=2**32),
            lambda c: c.update(group_rss_limit_bytes=9*2**30)]
        for mutate in changes:
            bad = copy.deepcopy(self.cfg)
            mutate(bad)
            with self.assertRaises(ValueError):
                driver.contract(bad)

    def fixture(self, mode):
        tmp = tempfile.TemporaryDirectory(prefix='meta-diag-outer-', dir='/data1/Kane/MOE')
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        cfg = copy.deepcopy(self.cfg)
        cfg['output_root'] = str(root/'run')
        path = root/'config.json'
        path.write_text(json.dumps(cfg))
        def fake(command, cwd, folder, seconds, rss):
            folder.mkdir()
            rec = ExpertRecorder(folder/'trace.jsonl', time.monotonic(), driver.identity(path))
            if mode == 'COMPLETED':
                rec.emit('WORKER_COMPLETE')
                result = {'status': 'UNKNOWN', 'request_id': 'mnist_0', 'arm': 'act',
                    'label': cfg['requests'][0]['label'], 'config_sha256': sha256(path),
                    'tensor_file_sha256': sha256(cfg['requests'][0]['tensor_file']),
                    'worker_seconds': .001}
                (folder/'result.json').write_text(json.dumps(result))
            else:
                rec.emit('BEGIN', name='partial', parent=None, arguments={})
                (folder/'result.json').write_text('{"status":')
            rec.close()
            for stream in ('stdout', 'stderr'):
                (folder/f'{stream}.txt').write_text('control')
            receipt = {'status': mode, 'exit_code': 0 if mode == 'COMPLETED' else -9,
                'error': None, 'command': command, 'deadline_seconds': seconds,
                'execution_including_preflight_seconds': .02, 'total_with_postflight_seconds': .021,
                'peak_sampled_group_rss_bytes': 1000, 'group_rss_limit_bytes': rss,
                'rss_poll_seconds': .05, 'postflight_in_execution_budget': False,
                'rss_is_sampled_not_instantaneous_cap': True, 'receipt_own_write_excluded_from_this_clock': True,
                'stdout_sha256': sha256(folder/'stdout.txt'), 'stderr_sha256': sha256(folder/'stderr.txt')}
            (folder/'receipt.json').write_text(json.dumps(receipt))
            time.sleep(.03)
            return receipt
        with patch.object(driver, 'validate'), patch.object(driver, 'supervise', side_effect=fake) as calls:
            driver.run(path)
            self.assertEqual(calls.call_count, 1)
        return path, Path(cfg['output_root'])

    def review(self, path):
        with patch('audit_metamoe_expert_diagnostic.validate'):
            return audit(path)

    def test_single_request_no_followup_and_no_overwrite(self):
        path, root = self.fixture('COMPLETED')
        result = self.review(path)
        self.assertEqual(result['audit'], 'PASS')
        self.assertFalse(result['opens_formal_cohort'])
        with patch.object(driver, 'validate'), patch.object(driver, 'supervise') as call:
            with self.assertRaises(FileExistsError):
                driver.run(path)
            call.assert_not_called()

    def test_timeout_partial_evidence_and_cost_preserved(self):
        path, _ = self.fixture('TIMEOUT')
        result = self.review(path)
        self.assertEqual(result['row']['status'], 'TIMEOUT')
        self.assertTrue(result['row']['result_parse_error'])
        self.assertEqual(len(result['trace']['open_span_ids']), 1)
        self.assertGreater(result['batch_cost']['batch_wall_through_summary_seconds'], .021)

    def test_audit_rejects_tampered_trace(self):
        path, root = self.fixture('COMPLETED')
        file = root/'mnist_0_act/trace.jsonl'
        file.write_bytes(file.read_bytes().replace(b'WORKER_COMPLETE', b'WORKER_REPLACED'))
        with self.assertRaisesRegex(ValueError, 'trace identity'):
            self.review(path)


if __name__ == '__main__':
    unittest.main()
