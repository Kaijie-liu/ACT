"""Queue and receipt controls; no trained model, dataset or solver calls."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('b1_observed_batch',
    Path(__file__).resolve().parents[1]/'scripts/b1_observed_batch.py')
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


class BatchControls(unittest.TestCase):
    def test_order_and_sealed_exclusion(self):
        rows = [dict(method='adaptive',status='UNKNOWN',dataset_index=i,model=s)
                for i,s in [(4150,'seed2'),(4096,'seed1'),(4141,'seed2'),(4150,'seed1')]]
        rows += [dict(method='matched',status='UNKNOWN',dataset_index=4100,model='seed0'),
                 dict(method='adaptive',status='SAFE',dataset_index=4100,model='seed0')]
        self.assertEqual([(r['dataset_index'],r['model']) for r in m.select(rows)],
                         [(4141,'seed2'),(4150,'seed1'),(4150,'seed2')])

    def test_outer_terminal_cannot_be_replaced_by_outcome_file(self):
        r = dict(status='COMPLETED', returncode=0, execution_error=None,
                 cleanup_error=None,cleanup_status='LEADER_REAPED_NO_LIVE_SESSION')
        self.assertTrue(m.session_ok(r))
        for k,v in [('status','TIMEOUT'),('returncode',1),('execution_error',{}),
                    ('cleanup_error',{}),('cleanup_status','CLEANUP_INCOMPLETE')]:
            self.assertFalse(m.session_ok(dict(r,**{k:v})))


if __name__ == '__main__': unittest.main()
