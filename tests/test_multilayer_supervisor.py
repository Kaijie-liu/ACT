import json
from pathlib import Path
import tempfile
import time
import unittest
import torch

from act.back_end.moe.multilayer_supervisor import supervise_call, verify_multilayer_box_supervised, _atomic
from test_multilayer_verifier import example


def controlled_worker(kind, deadline, directory):
    if kind=='error':
        raise RuntimeError('intentional-control')
    if kind=='partial':
        _atomic(directory/'progress.json',{'records':[{'status':'UNKNOWN','phase':'property'}]})
        time.sleep(10.)
    if kind=='late':
        while time.monotonic() < deadline+.2:
            time.sleep(.02)
    return {'status':'POSITIVE','reason':'control','evidence_grade':'CONTROL_ONLY'}


class SupervisorControls(unittest.TestCase):
    def setUp(self):
        root=Path('/data1/Kane/MOE/baseline_runs')
        root.mkdir(exist_ok=True)
        self.temp=tempfile.TemporaryDirectory(prefix='multilayer_test_',dir=root)
        self.addCleanup(self.temp.cleanup)

    def run_control(self,kind,budget=4.,accept=None):
        return supervise_call(controlled_worker,kind,output_dir=Path(self.temp.name)/kind,
            total_seconds=budget,accept=accept or (lambda r:{'status':'PASS','issues':[]}))

    def test_complete_and_no_overwrite(self):
        result=self.run_control('normal')
        self.assertEqual(result['status'],'POSITIVE',result)
        self.assertGreaterEqual(result['return_wall_seconds'],result['decision_seconds'])
        self.assertIn('candidate.json',result['artifacts'])
        with self.assertRaises(FileExistsError): self.run_control('normal')

    def test_exception_retained(self):
        result=self.run_control('error')
        self.assertEqual(result['status'],'ERROR')
        self.assertIn('error.json',result['artifacts'])

    def test_partial_and_hard_deadline(self):
        result=self.run_control('partial')
        self.assertEqual(result['status'],'TIMEOUT',result)
        self.assertEqual(result['partial_histories'],1)
        self.assertEqual(result['evidence_grade'],'NONE')
        self.assertLess(result['return_wall_seconds'],6.)

    def test_late_and_audit_failure(self):
        self.assertEqual(self.run_control('late')['status'],'TIMEOUT')
        self.assertEqual(self.run_control('bad-audit',accept=lambda _:{'status':'FAIL','issues':['control']})['status'],'ERROR')

    def test_startup_timeout(self):
        result=self.run_control('startup',budget=.001)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertEqual(result['evidence_grade'],'NONE')

    def test_real_supervised_positive_and_replay(self):
        x=torch.zeros(1,1,dtype=torch.float64)
        for unsafe in (False,True):
            result=verify_multilayer_box_supervised(example(unsafe),output_dir=Path(self.temp.name)/str(unsafe),
                center=x,lower=x-1,upper=x+1,rows=torch.tensor([[1.,-1.]],dtype=torch.float64),
                thresholds=torch.zeros(1,dtype=torch.float64),total_seconds=20.)
            self.assertEqual(result['status'],'UNSAFE_REPLAYED' if unsafe else 'POSITIVE',result)
            self.assertEqual(result['audit']['status'],'PASS')


if __name__=='__main__': unittest.main()
