"""Finite standard-library controls for S1 accounting and online acceptance."""
import copy
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import time
import unittest

from architecture_s1_execution import Artifacts, catalog, audit_value, receive, digest, read_record
from scoped_proof.io import save, sha, load

CONTROL_ROOT = None


def positive():
    return dict(status='POSITIVE', expected_histories=1, attempted_histories=1,
        returned_histories=1, verified_histories=1, pruned_histories=0,
        attempted_prefix_queries=0, prefix_queries=0, prefix_bounds=[], exclusions=[],
        complete=True, evidence_grade='ALPHA_BETA_CROWN_NUMERICAL_FILTER',
        histories=[dict(history=[], outcome=dict(status='verified', actual_raw_spec=[[[[1.]], [0.]]]))],
        evaluation=dict(family='dense', mode='complete'))


class Controls(unittest.TestCase):
    def setUp(self):
        self.root = CONTROL_ROOT/self._testMethodName
        self.store = Artifacts.create(self.root, cap=128*1024, terminal_reserve=4096)

    def test_parallel_account_and_immutable_files(self):
        values = [dict(k=i, data='a'*100) for i in range(40)]
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: self.store.write(self.root/f'{i}.json.gz', values[i]), range(40)))
        for i in range(40):
            self.assertEqual(read_record(self.root/f'{i}.json.gz'), values[i])
        self.assertEqual(load(self.store.account)['used'],
            4096+sum((self.root/f'{i}.json.gz').stat().st_size for i in range(40)))
        with self.assertRaises(ValueError):
            self.store.write(self.root/'0.json.gz', values[0])
        with self.assertRaises(ValueError):
            self.store.write(self.root/'0.json.gz', values[0], replace=True)

    def test_capacity_rejection_preserves_terminal_and_progress(self):
        before = load(self.store.account)['used']
        with self.assertRaises(BufferError):
            self.store.write(self.root/'large.json', dict(text='x'*128*1024))
        self.assertEqual(load(self.store.account)['used'], before)
        self.store.write(self.root/'progress.json.gz', dict(counter=1), replace=True)
        self.store.write(self.root/'progress.json.gz', dict(counter=200), replace=True)
        self.assertEqual(load(self.store.account)['used'], before+(self.root/'progress.json.gz').stat().st_size)
        self.store.write(self.root/'error.json.gz', dict(status='RESOURCE_LIMIT'), terminal=True)

    def test_queue_denominators_and_no_selection(self):
        models = [dict(id=f'm{i}') for i in range(12)]
        for mode, expected, count in [('complete', 4800, 200), ('incomplete', 24000, 1000)]:
            rows = catalog(models, mode)
            self.assertEqual(len(rows), expected)
            self.assertEqual(len({r['id'] for r in rows}), expected)
            self.assertEqual({(r['index'], r['radius'], r['model']) for r in rows},
                {(i, e, m['id']) for i in range(count) for e in (2, 8) for m in models})
            self.assertIn(98, {r['index'] for r in rows})  # Explicit new-study prefix.
        with self.assertRaises(ValueError):
            catalog([dict(id='x'), dict(id='x')], 'complete')

    def test_partial_and_missing_properties_rejected(self):
        result = positive()
        cov = audit_value(result, family='dense', experts=[], properties=1)
        self.assertTrue(cov['proof_complete'])
        bad = copy.deepcopy(result); bad['histories'] = []; bad['returned_histories'] = 0
        with self.assertRaises(ValueError):
            audit_value(bad, family='dense', experts=[], properties=1)
        with self.assertRaises(ValueError):
            audit_value(result, family='dense', experts=[], properties=2)
        bad = copy.deepcopy(result); bad['expected_histories'] = 2
        with self.assertRaises(ValueError):
            audit_value(bad, family='dense', experts=[], properties=1)

    def test_received_identity_deadline_cleanup_and_replay(self):
        now = time.monotonic(); folder = self.root/'job'; folder.mkdir()
        spec = dict(synthetic_control=True, backend_sources_sha256='source')
        save(self.root/'manifest.json', spec)
        query = dict(manifest=str(self.root/'manifest.json'), manifest_sha256=sha(self.root/'manifest.json'),
            worker_deadline=now+100, mode='complete')
        self.store.write(folder/'request.json.gz', query)
        result = positive()
        value = dict(result=result, family='dense', experts=[], properties=1,
            coverage=audit_value(result, family='dense', experts=[], properties=1),
            backend_sources_sha256='source', replayed=False)
        h = self.store.write(folder/'outcome.json.gz', dict(request_sha256=sha(folder/'request.json.gz'),
            clock=now, value=value))
        self.store.write(folder/'published.json.gz', dict(sha256=h, clock=now+1))
        session = dict(status='COMPLETED', returncode=0, execution_error=None,
            cleanup_error=None, cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
        self.assertEqual(receive(folder, query, session, now+200)[0], 'POSITIVE')
        self.assertEqual(receive(folder, query, session, now-1), ('TIMEOUT', None))
        bad = dict(session, cleanup_status='CLEANUP_INCOMPLETE')
        self.assertEqual(receive(folder, query, bad, now+200), ('ERROR', None))
        bad = dict(query, worker_deadline=now+300)
        with self.assertRaises(ValueError):
            receive(folder, bad, session, now+400)
        # Separate immutable bad-replay artifact, never mutate the normal record.
        bad_folder = self.root/'bad'; bad_folder.mkdir()
        self.store.write(bad_folder/'request.json.gz', query)
        bad_value = copy.deepcopy(value)
        bad_value['result'].update(status='UNSAFE_REPLAYED', witness=[[0.]])
        h = self.store.write(bad_folder/'outcome.json.gz', dict(
            request_sha256=sha(bad_folder/'request.json.gz'), clock=now, value=bad_value))
        self.store.write(bad_folder/'published.json.gz', dict(sha256=h, clock=now+1))
        with self.assertRaises(ValueError):
            receive(bad_folder, query, session, now+200)

    def test_returned_timeout_before_outer_cutoff_is_partial_not_error(self):
        result = positive()
        result.update(status='TIMEOUT', complete=False, evidence_grade='NONE',
            histories=[], returned_histories=0, verified_histories=0)
        cov = audit_value(result, family='dense', experts=[], properties=1)
        self.assertEqual(cov['in_flight'], 1)
        self.assertFalse(cov['proof_complete'])
        result['evidence_grade'] = 'ALPHA_BETA_CROWN_NUMERICAL_FILTER'
        with self.assertRaises(ValueError):
            audit_value(result, family='dense', experts=[], properties=1)
