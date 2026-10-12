"""Small parent-deadline, partial-evidence and audit ledger controls."""
import copy
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.run_scaled_endpoint_controls import call
from scripts.audit_scaled_endpoint_controls import accounting,code_inventory,PHASES,NATIVE_PHASES


class Controls(unittest.TestCase):
    def test_hard_cutoff_preserves_partial_without_late_success(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d); p=root/'partial.json'
            r=call(root,'cut',[sys.executable,'-B','-c',
                'import sys,time;from pathlib import Path;Path(sys.argv[1]).write_text("partial");time.sleep(1)',str(p)],
                cwd=root,env=os.environ.copy(),budget=.2)
            self.assertEqual(r['status'],'TIMEOUT'); self.assertEqual(p.read_text(),'partial')
            self.assertTrue((root/'cut.stdout').exists()); self.assertTrue((root/'cut.stderr').exists())

    def test_exception_retains_logs_and_late_zero_exit_is_timeout(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d)
            r=call(root,'error',[sys.executable,'-B','-c','print("partial",flush=True);raise ValueError("injected")'],
                   cwd=root,env=os.environ.copy(),budget=2)
            self.assertEqual(r['status'],'ERROR'); self.assertIn(b'partial',(root/'error.stdout').read_bytes())
            result=subprocess.CompletedProcess([],0,b'complete',b'')
            with (patch('scripts.run_scaled_endpoint_controls.time.monotonic',side_effect=[0.,.001,1.1]),
                  patch('scripts.run_scaled_endpoint_controls.subprocess.run',return_value=result)):
                r=call(root,'late',[],cwd=root,env={},budget=1)
            self.assertEqual(r['status'],'TIMEOUT'); self.assertEqual(r['returncode'],0)

    def test_cost_ledger_rejects_reset_missing_and_free_nested_costs(self):
        counts=lambda n:dict(row_visits=n,coefficient_visits=n,arithmetic=n)
        r=dict(status='COMPLETE_FINITE_SCALED_CONTROL',real_models=0,gpu_calls=0,worker_before_receipt_seconds=2.,
            phases={k:float(k=='complete_candidate_and_check_seconds') for k in PHASES},
            source_work=counts(1),production_work=counts(2),work=counts(3),
            native=dict(device='cpu',dtype='float64',expanded_constraint_entries=0,
                algorithm='CPU_SCALED_REFERENCE_PROJECTED_SUBGRADIENT_128_V3',
                **{k:(.5 if k=='preparation_seconds' else 0.) for k in NATIVE_PHASES}))
        c=dict(parent_observed_seconds=3.); accounting(r,c)
        for mutate in (lambda q:q.__setitem__('production_work',counts(0)),
                       lambda q:q['native'].__setitem__('preparation_seconds',2.),
                       lambda q:q['work'].__setitem__('arithmetic',8000001),
                       lambda q:q['phases'].__setitem__('source_seconds',9.),
                       lambda q:q['phases'].pop('capture_and_place_seconds'),
                       lambda q:q['native'].pop('exact_check_seconds'),
                       lambda q:q.__setitem__('worker_before_receipt_seconds',4.)):
            q=copy.deepcopy(r); mutate(q)
            with self.assertRaises(ValueError): accounting(q,c)

    def test_code_inventory_cannot_drop_or_add_unchecked_files(self):
        from scripts.run_scaled_endpoint_controls import FILES
        from scoped_source.hz_request_store import digest
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as d:
            root=Path(d); code={k:digest(b'control') for k in FILES}
            for directory in ('code','relocated/code'):
                for k in FILES:
                    p=root/directory/k; p.parent.mkdir(parents=True,exist_ok=True); p.write_bytes(b'control')
            code_inventory(root,code); missing=dict(code); missing.pop(FILES[0])
            with self.assertRaises(ValueError): code_inventory(root,missing)
            (root/'code/extra.py').write_bytes(b'not listed')
            with self.assertRaises(ValueError): code_inventory(root,code)


if __name__=='__main__': unittest.main()
