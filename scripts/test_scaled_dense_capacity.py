"""Supervisor-only controls; never execute the frozen dense source."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.run_scaled_dense_capacity import supervise, received, reserve, save, PHASES, RSS_LIMIT
from scripts.audit_scaled_dense_capacity import prefix_audit, SOURCE


class Controls(unittest.TestCase):
    def test_deadline_partial_and_exception(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d); partial=root/'partial'
            cmd=[sys.executable,'-B','-c',
                 'import sys,time;from pathlib import Path;Path(sys.argv[1]).write_text("raw");time.sleep(2)',str(partial)]
            r=supervise(root,'cut',cmd,cwd=root,env=os.environ.copy(),budget=.25)
            self.assertEqual(r['status'],'TIMEOUT'); self.assertEqual(partial.read_text(),'raw')
            self.assertEqual(received(r,root),'INCOMPLETE_QUERY')
            r=supervise(root,'err',[sys.executable,'-B','-c','print("prefix",flush=True);raise ValueError("injected")'],
                        cwd=root,env=os.environ.copy(),budget=2)
            self.assertEqual(r['status'],'ERROR'); self.assertIn(b'prefix',(root/'err.stdout').read_bytes())

    def test_memory_cutoff(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            with patch('scripts.run_scaled_dense_capacity.rss',return_value=101):
                r=supervise(d,'mem',[sys.executable,'-B','-c','import time;time.sleep(2)'],
                            cwd=d,env=os.environ.copy(),budget=2,memory=100)
            self.assertEqual(r['status'],'MEMORY_LIMIT'); self.assertEqual(r['sampled_peak_rss_bytes'],101)

    def test_receiving_is_budgeted(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            def slow(_): time.sleep(.3); return 'not timely'
            r=supervise(d,'late',[sys.executable,'-B','-c','pass'],cwd=d,env=os.environ.copy(),
                        budget=.2,receiver=slow)
            self.assertEqual(r['status'],'TIMEOUT'); self.assertEqual(received(r,d),'INCOMPLETE_QUERY')

    def test_complete_receipt_cost_and_missing_evidence(self):
        counts=lambda x:dict(row_visits=x,coefficient_visits=x,arithmetic=x)
        r=dict(status='COMPLETE_DENSE_QUERY',worker_before_receipt_seconds=2.,worker_peak_rss_bytes=1024,
               phases={k:.1 for k in PHASES},source_work=counts(1),production_work=counts(2),work=counts(2))
        c=dict(status='COMPLETE',parent_observed_seconds=3.)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            p=Path(d)/'receipt.json'; self.assertEqual(received(c,d),'INCOMPLETE_QUERY')
            save(p,r); self.assertEqual(received(c,d),'COMPLETE_QUERY')
            mutations=[lambda x:x['phases'].pop('source_copy'),lambda x:x.__setitem__('production_work',counts(0)),
                       lambda x:x['work'].__setitem__('arithmetic',8000001),
                       lambda x:x['phases'].__setitem__('source_copy',4),
                       lambda x:x.__setitem__('worker_peak_rss_bytes',RSS_LIMIT+1)]
            for fn in mutations:
                bad=copy.deepcopy(r); fn(bad)
                with patch('scripts.run_scaled_dense_capacity.json.loads',return_value=bad):
                    with self.assertRaises(ValueError): received(c,d)
            # A rejecting Work counter is retained exactly; never clamped for a PASS.
            bad=dict(r,status='WORK_LIMIT',work=counts(8000001))
            with patch('scripts.run_scaled_dense_capacity.json.loads',return_value=bad):
                self.assertEqual(received(c,d),'INCOMPLETE_QUERY')

    def test_reserve_before_publication_and_owned_process_cleanup(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            reserve(d,1)
            with patch('scripts.run_scaled_dense_capacity.LIMIT',2*2**20):
                with self.assertRaises(ValueError): reserve(d,1)
            r=supervise(d,'badlaunch',['/not/an/executable'],cwd=d,env={},budget=1)
            self.assertEqual(r['status'],'ERROR'); self.assertTrue((Path(d)/'badlaunch.call.json').exists())

    def test_interrupted_progress_tail_is_not_complete_or_repaired(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d); c=dict(parent_observed_seconds=1.)
            save(root/'progress-01.json',dict(source=SOURCE,stage='imports',elapsed_seconds=.1,
                phases=dict(imports=.01),work=dict(row_visits=1,coefficient_visits=0,arithmetic=0)))
            (root/'progress-02.json').write_bytes(b'{"interrupted":')
            r=prefix_audit(root,c,complete=False)
            self.assertEqual(r['unreadable_interrupted_tail'],['progress-02.json'])
            self.assertEqual((root/'progress-02.json').read_bytes(),b'{"interrupted":')
            with self.assertRaises(ValueError): prefix_audit(root,c,complete=True)
            bad=dict(source='0'*64,stage='imports',elapsed_seconds=.1,phases={},work={})
            with patch('scripts.audit_scaled_dense_capacity.json.loads',return_value=bad):
                with self.assertRaises(ValueError): prefix_audit(root,c,complete=False)


if __name__=='__main__': unittest.main()
