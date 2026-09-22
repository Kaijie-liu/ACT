import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from metamoe_csr_execution import supervise, terminal, resource_config
from audit_metamoe_csr import check_terminal


class OuterControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='csr-control-', dir='/data1/Kane/MOE')
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def call(self, code, seconds=3, limit=2**30):
        folder = self.root/'call'
        receipt = supervise([sys.executable, '-c', code], str(self.root), folder, seconds, limit)
        self.assertGreaterEqual(receipt['total_with_postflight_seconds'], receipt['execution_including_preflight_seconds'])
        self.assertTrue((folder/'receipt.json').exists())
        return receipt, folder

    def test_success_and_cost(self):
        # A process may legitimately exit between RSS samples. Keep this
        # sampling control alive across several polls instead of asserting a
        # nonzero sample for an arbitrarily short process.
        receipt, _ = self.call('import time; print("completed"); time.sleep(.2)')
        self.assertEqual(receipt['status'], 'COMPLETED')
        self.assertGreater(receipt['peak_sampled_group_rss_bytes'], 0)

    def test_exception_and_missing_candidate(self):
        receipt, folder = self.call('raise RuntimeError("control")')
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertEqual(terminal(folder, receipt)['status'], 'ERROR')

    def test_deadline_truncated_and_late_candidate(self):
        receipt, folder = self.call('import time; time.sleep(5)', seconds=.2)
        self.assertEqual(receipt['status'], 'TIMEOUT')
        (folder/'result.json').write_text('{"status":')
        value = terminal(folder, receipt)
        self.assertEqual(value['status'], 'TIMEOUT')
        self.assertTrue(value['result_parse_error'])
        (folder/'result.json').write_text(json.dumps({'status': 'POSITIVE'}))
        self.assertEqual(terminal(folder, receipt)['status'], 'TIMEOUT')

    def test_partial_success_is_error(self):
        receipt, folder = self.call('pass')
        (folder/'result.json').write_text('{')
        self.assertEqual(terminal(folder, receipt)['status'], 'ERROR')

    def test_rss_refusal(self):
        receipt, _ = self.call('import time; a=bytearray(16000000); time.sleep(2)', limit=1)
        self.assertEqual(receipt['status'], 'RESOURCE_LIMIT')

    def test_descendant_in_group_is_killed(self):
        child = self.root/'child'
        code = ('import subprocess,sys,time; from pathlib import Path; '
                f'p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(10)"]); Path({str(child)!r}).write_text(str(p.pid)); time.sleep(10)')
        receipt, _ = self.call(code, seconds=.5)
        self.assertEqual(receipt['status'], 'TIMEOUT')
        pid = int(child.read_text())
        stat = Path(f'/proc/{pid}/stat')
        if stat.exists():
            self.assertEqual(stat.read_text().rsplit(')', 1)[1].split()[0], 'Z')

    def test_resource_freeze(self):
        cfg = {'hybridz': {'sparse_resource_policy': 'csr_bytes_v1', 'sparse_representation_bytes': 2**31},
               'group_rss_limit_bytes': 8*2**30}
        resource_config(cfg)
        cfg['group_rss_limit_bytes'] += 1
        with self.assertRaises(ValueError):
            resource_config(cfg)

    def test_independent_audit_rejects_late_act_positive(self):
        with self.assertRaises(ValueError):
            check_terminal({'status': 'POSITIVE', 'seconds': 301},
                {'status': 'COMPLETED', 'deadline_seconds': 300}, {'status': 'POSITIVE'})
        with self.assertRaises(ValueError):
            check_terminal({'status': 'POSITIVE', 'seconds': 2},
                {'status': 'TIMEOUT', 'deadline_seconds': 300}, {'status': 'POSITIVE'})

    def test_preflight_exhaustion_no_spawn(self):
        with (patch('metamoe_csr_execution.time.monotonic', side_effect=[0., 4., 4., 4.]),
              patch('metamoe_csr_execution.subprocess.Popen') as spawn):
            receipt, _ = self.call('pass', seconds=3)
        self.assertEqual(receipt['status'], 'TIMEOUT')
        spawn.assert_not_called()

    def test_spawn_failure_keeps_receipt(self):
        with patch('metamoe_csr_execution.subprocess.Popen', side_effect=OSError('control')):
            receipt, _ = self.call('pass')
        self.assertEqual(receipt['status'], 'ERROR')
        self.assertIn('control', receipt['error'])


if __name__ == '__main__':
    unittest.main()
