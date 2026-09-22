"""Freeze/terminal controls ONLY: no checkpoint load, inference or solver."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import metamoe_csr_paired_r4 as driver
from freeze_metamoe_csr_paired_r4 import build_config
from audit_metamoe_csr_paired_r4 import audit, check_receipt, check_candidate, complete_numerical, replay_binding
from recent_moe_deployment import sha256


class SmokeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.frozen = build_config()  # Only hashes/inventory/saved diagnostic recheck.

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='metamoe-smoke-freeze-', dir='/data1/Kane/MOE')
        self.root = Path(self.temp.name)
        self.cfg = copy.deepcopy(self.frozen)
        self.cfg['output_root'] = str(self.root/'run')
        self.path = self.root/'config.json'
        self.path.write_text(json.dumps(self.cfg))

    def tearDown(self):
        self.temp.cleanup()

    def test_frozen_request_budget_and_parent_mutations(self):
        driver.contract(self.frozen)
        mutations = [lambda c: c.update(seconds=301), lambda c: c.update(margin=0.),
            lambda c: c.update(epsilon=4/255), lambda c: c.update(group_rss_limit_bytes=9*2**30),
            lambda c: c['hybridz'].update(sparse_representation_bytes=4*2**30),
            lambda c: c['hybridz'].update(sparse_resource_policy='csr_bytes_v1'),
            lambda c: c['requests'].reverse(), lambda c: c['requests'][0].update(label=5),
            lambda c: c['requests'][0].update(tensor_file='other.npz'),
            lambda c: c.update(checkpoint='other.pth'), lambda c: c['python'].update(act='other-python'),
            lambda c: c.update(automatic_followup=True), lambda c: c['roster'].pop(),
            lambda c: c['prerequisites'].clear(), lambda c: c.update(parent_config_sha256='x'),
            lambda c: c['files'].pop('scripts/audit_metamoe_csr_paired_r4.py')]
        for mutate in mutations:
            bad = copy.deepcopy(self.frozen)
            mutate(bad)
            with self.assertRaises(ValueError):
                driver.contract(bad)

    def test_no_parent_source_rebinding(self):
        bad = copy.deepcopy(self.frozen)
        bad['files']['act/config/config.py'] = 'different'
        with self.assertRaisesRegex(ValueError, 'parent source'):
            driver.contract(bad)

    def candidate(self, request_id, arm):
        request = next(r for r in self.cfg['requests'] if r['id'] == request_id)
        result = {'status': 'UNKNOWN', 'config_sha256': sha256(self.path), 'request_id': request_id,
            'arm': arm, 'label': request['label'], 'tensor_file_sha256': sha256(request['tensor_file']),
            'worker_seconds': .001, 'reason': 'nonzero_or_global_output_obligation_unproved'}
        if arm == 'author':
            result.update(reason='backend_unresolved', backend_status='unknown')
        return result

    def fake_supervise(self, command, cwd, folder, seconds, rss):
        folder.mkdir()
        rid, arm = command[-3], command[-1]
        result = self.candidate(rid, arm)
        mode = self.modes.get((rid, arm), 'COMPLETED')
        result_file = folder/'result.json'
        if mode == 'PARTIAL':
            result_file.write_text('{"status":"POSITIVE"}')
        elif mode != 'ERROR':
            result_file.write_text(json.dumps(result))
        if arm == 'author' and mode not in ('PARTIAL', 'ERROR'):
            (folder/'backend.stdout').write_text('Result: unknown in 0.001 seconds\nFinal verified acc: 0.0% (total 1 examples)\n')
        for name in ('stdout', 'stderr'):
            (folder/f'{name}.txt').write_text('synthetic control only')
        state = 'TIMEOUT' if mode in ('TIMEOUT', 'PARTIAL') else mode
        receipt = {'status': state, 'command': command, 'deadline_seconds': seconds,
            'execution_including_preflight_seconds': .01, 'total_with_postflight_seconds': .012,
            'peak_sampled_group_rss_bytes': 1024, 'group_rss_limit_bytes': rss,
            'exit_code': 0 if state == 'COMPLETED' else -9, 'error': None,
            'rss_poll_seconds': .05, 'postflight_in_execution_budget': False,
            'rss_is_sampled_not_instantaneous_cap': True, 'receipt_own_write_excluded_from_this_clock': True,
            'stdout_sha256': sha256(folder/'stdout.txt'), 'stderr_sha256': sha256(folder/'stderr.txt')}
        (folder/'receipt.json').write_text(json.dumps(receipt))
        time.sleep(.02)  # Synthetic batch wall must include its charged/postflight times.
        return receipt

    def fixture(self, modes=None):
        self.modes = modes or {}
        with patch.object(driver, 'validate'), patch.object(driver, 'supervise', side_effect=self.fake_supervise) as calls:
            driver.run(self.path)
        return calls

    def review(self, replay=None):
        with patch('audit_metamoe_csr_paired_r4.validate'):
            return audit(self.path, replay)

    def test_four_call_order_no_auto_followup_and_separate_replay_gate(self):
        calls = self.fixture()
        self.assertEqual(calls.call_count, 4)
        self.assertEqual([(c.args[0][-3], c.args[0][-1]) for c in calls.call_args_list], driver.ROSTER)
        first = self.review()
        self.assertTrue(first['execution_control_pass'])
        self.assertFalse(first['smoke_gate_pass'])
        replay = self.root/'replay.json'
        replay.write_text(json.dumps({'audit': 'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS',
            'config_sha256': sha256(self.path), 'summary_sha256': first['summary_sha256'],
            'rows': [], 'separate_audit_seconds': .1}))
        final = self.review(replay)
        self.assertTrue(final['smoke_gate_pass'])
        self.assertFalse(final['opens_formal_cohort'])
        self.assertGreater(final['batch_cost']['batch_wall_through_summary_seconds'],
                           final['batch_cost']['charged_request_seconds'])

    def test_error_stops_and_preserves_all_four_denominators(self):
        self.assertEqual(self.fixture({driver.ROSTER[0]: 'ERROR'}).call_count, 1)
        out = self.review()
        self.assertEqual(len(out['rows']), 4)
        self.assertEqual([r['status'] for r in out['rows']], ['ERROR']+['NOT_STARTED_AFTER_ERROR']*3)
        self.assertFalse(out['execution_control_pass'])

    def test_timeout_and_partial_evidence_are_retained_without_upgrade(self):
        self.fixture({driver.ROSTER[0]: 'PARTIAL', driver.ROSTER[1]: 'TIMEOUT'})
        out = self.review()
        self.assertEqual([r['status'] for r in out['rows'][:2]], ['TIMEOUT', 'TIMEOUT'])
        self.assertTrue(out['rows'][0]['result_parse_error'])
        self.assertFalse(out['execution_control_pass'])

    def test_overwrite_refused(self):
        self.fixture()
        with patch.object(driver, 'validate'), patch.object(driver, 'supervise') as call:
            with self.assertRaises(FileExistsError):
                driver.run(self.path)
            call.assert_not_called()

    def test_roster_mutations_rejected(self):
        self.fixture()
        p = Path(self.cfg['output_root'])/'summary.json'
        original = json.loads(p.read_text())
        for rows in [original['rows'][:-1], list(reversed(original['rows'])), [original['rows'][0]]*4]:
            p.write_text(json.dumps({**original, 'rows': rows}))
            with self.assertRaisesRegex(ValueError, 'denominator'):
                self.review()

    def test_receipt_mutations_rejected(self):
        self.fixture()
        root = Path(self.cfg['output_root'])/'cifar10_0_act'
        receipt = json.loads((root/'receipt.json').read_text())
        row = json.loads((root/'terminal.json').read_text())
        check_receipt(receipt, row, self.cfg, receipt['command'])
        for key, value in [('execution_including_preflight_seconds', float('nan')),
            ('execution_including_preflight_seconds', 301.), ('total_with_postflight_seconds', -.1),
            ('exit_code', 1), ('error', 'bad'), ('peak_sampled_group_rss_bytes', 9*2**30),
            ('command', ['other']), ('deadline_seconds', 301), ('group_rss_limit_bytes', 9*2**30)]:
            with self.assertRaises(ValueError):
                check_receipt({**receipt, key: value}, row, self.cfg, receipt['command'])

    def test_artifact_tamper_rejected(self):
        self.fixture()
        p = Path(self.cfg['output_root'])/'cifar10_0_author'/'backend.stdout'
        p.write_text('tampered')
        with self.assertRaisesRegex(ValueError, 'artifact'):
            self.review()

    def test_cost_underreporting_rejected(self):
        self.fixture()
        p = Path(self.cfg['output_root'])/'batch_cost.json'
        old = json.loads(p.read_text())
        for key, value in [('charged_request_seconds', 0.), ('batch_wall_through_summary_seconds', .04)]:
            p.write_text(json.dumps({**old, key: value}))
            with self.assertRaises(ValueError):
                self.review()

    def test_raw_backend_timeout_is_not_complete(self):
        result = self.candidate('mnist_0', 'author')
        result['backend_status'] = 'timeout'
        check_candidate(result, 'author')
        self.assertFalse(complete_numerical({'status': 'UNKNOWN', 'arm': 'author'}, result))
        result = self.candidate('mnist_0', 'act')
        for reason in ('sparse_representation_resource_limit', 'missing_correlated_router_hz', 'unregistered'):
            result['reason'] = reason
            self.assertFalse(complete_numerical({'status': 'UNKNOWN', 'arm': 'act'}, result))

    def positive(self):
        return {'status': 'POSITIVE', 'worker_seconds': 1., 'evidence_grade': 'HZ_POLICY_ACCEPTED',
            'source_complete': False, 'class_counts': [10, 10], 'property_rows': 19,
            'semantics': 'class_separated_raw_top1_zero_fill_any_legal_ties',
            'candidates': [0, 1], 'excluded': [], 'unresolved': [],
            'expert_statuses': {'0': 'certified', '1': 'certified'},
            'nonzero_obligations': [{'expert': i, 'accepted': True, 'lower': 1., 'upper': 2.} for i in (0, 1)]}

    def test_positive_complete_obligations_required(self):
        check_candidate(self.positive(), 'act')
        mutations = [lambda r: r.update(property_rows=18), lambda r: r['nonzero_obligations'].pop(),
            lambda r: r['nonzero_obligations'][0].update(lower=-1.),
            lambda r: r['expert_statuses'].pop('1'), lambda r: r.update(source_complete=True),
            lambda r: r['unresolved'].append(1), lambda r: r['candidates'].append(0)]
        for mutate in mutations:
            result = self.positive()
            mutate(result)
            with self.assertRaises(ValueError):
                check_candidate(result, 'act')

    def test_wrong_arm_and_backend_upgrade_rejected(self):
        with self.assertRaises(ValueError):
            check_candidate(self.positive(), 'author')
        result = {'status': 'BACKEND_POSITIVE', 'worker_seconds': 1.,
                  'evidence_grade': 'AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER', 'backend_status': 'safe'}
        check_candidate(result, 'author')
        for raw in ('unknown', 'unsafe-bab', 'timeout', None):
            with self.assertRaises(ValueError):
                check_candidate({**result, 'backend_status': raw}, 'author')

    def test_replay_complete_unique_binding(self):
        review = {'config_sha256': 'c', 'summary_sha256': 's', 'rows': [
            {'id': 'mnist_0', 'arm': 'act', 'result_sha256': 'r', 'status': 'UNSAFE_REPLAYED'}]}
        replay = {'audit': 'INDEPENDENT_ORIGINAL_MODEL_REPLAY_PASS', 'config_sha256': 'c',
            'summary_sha256': 's', 'separate_audit_seconds': .1,
            'rows': [{'id': 'mnist_0', 'arm': 'act', 'result_sha256': 'r',
                      'label': 17, 'prediction': 17, 'minimum_margin': 0.}]}
        replay_binding(review, replay, self.cfg)
        for delta in [{'rows': []}, {'rows': replay['rows']*2}, {'summary_sha256': 'wrong'},
                      {'separate_audit_seconds': float('nan')}, {'config_sha256': 'other'}]:
            with self.assertRaises(ValueError):
                replay_binding(review, {**replay, **delta}, self.cfg)
        for key, value in [('label', 999), ('minimum_margin', None), ('minimum_margin', float('nan')),
                           ('minimum_margin', 1.), ('prediction', 20), ('prediction', True)]:
            bad = copy.deepcopy(replay)
            bad['rows'][0][key] = value
            with self.assertRaises(ValueError):
                replay_binding(review, bad, self.cfg)

    def test_preflight_validation_failure_never_spawns(self):
        with patch.object(driver, 'validate', side_effect=ValueError('identity')), patch.object(driver, 'supervise') as call:
            with self.assertRaises(ValueError):
                driver.run(self.path)
            call.assert_not_called()
            self.assertFalse(Path(self.cfg['output_root']).exists())


if __name__ == '__main__':
    unittest.main()
