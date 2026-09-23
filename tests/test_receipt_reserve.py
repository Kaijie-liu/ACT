"""Tiny native controls only; no model checkpoint or dataset is loaded."""
import io
import json
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch
from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.solver.isolated_feasibility import NativeSession
from act.back_end.solver.native_feasibility_worker import native_fraction, serve
from act.back_end.solver.receipt_reserve import command_with_fraction, reserve_on_checked_experts
from act.back_end.moe.checked_execution import CheckedExecutionOptions, checked_execution
from act.front_end.specs import OutKind, OutputSpec
from act.util.stats import VerifyStatus


class ReserveControls(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE', prefix='reserve-controls-')
        self.root = Path(self.tmp.name)
        self.hz = sh.sparse_hz_from_bounds(Bounds(torch.tensor([[-1.]], dtype=torch.float64),
            torch.tensor([[1.]], dtype=torch.float64)), frame_id=987)

    def tearDown(self):
        self.tmp.cleanup()

    def test_output_only_and_registered_fractions(self):
        for phase in ('base', 'router', 'unknown'):
            self.assertEqual(native_fraction({'phase':phase}, .8), 1.)
        for phase in ('expanded', 'contracted'):
            self.assertEqual(native_fraction({'phase':phase}, .8), .8)
        for bad in (.9, 0., True, '0.8', float('nan')):
            with self.assertRaises(ValueError):native_fraction({},bad)

    def test_invocation_identity_not_arbitrary_worker(self):
        from act.back_end.solver import native_feasibility_worker as w
        cmd = [sys.executable, w.__file__, '--model','m','--sha256','h']
        self.assertEqual(command_with_fraction(cmd,.8)[-2:], ['--output-budget-fraction','0.8'])
        for bad in ([*cmd,'extra'], [sys.executable,'other.py',*cmd[2:]], [*cmd[:2],'--other',*cmd[3:]]):
            with self.assertRaises(ValueError):command_with_fraction(bad,.8)

    def test_checked_composition_same_obligations_and_native_matrices(self):
        sources = []
        original = sh.HZSolver
        for fraction in (1.,.8):
            folder = self.root/str(fraction)
            with checked_execution(folder,request_sha256='a'*64,input_sha256='b'*64,
                    options=CheckedExecutionOptions(expert_base=True)) as cls:
                init = cls.__init__
                with reserve_on_checked_experts(cls,fraction):
                    solver = cls(time_limit=10)
                    spec = OutputSpec(kind=OutKind.LINEAR_LE,
                        c=torch.tensor([[1.],[-1.]],dtype=torch.float64),d=torch.tensor([2.,2.],dtype=torch.float64))
                    out = solver.evaluate_spec(self.hz,spec,batch_size=1,n_out=1,
                        input_hz=self.hz,input_shape=(1,1))[0]
                    self.assertEqual(out.status,VerifyStatus.CERTIFIED)
                    self.assertEqual(len(out.metadata['properties']),2)
                self.assertIs(cls.__init__,init)
            self.assertIs(sh.HZSolver,original)
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
            from audit_metamoe_checked_base_control import evaluate
            from audit_metamoe_receipt_reserve import native_caps
            for evaluation in (folder/'protected').glob('evaluation_*'):
                self.assertEqual(evaluate(evaluation,2,'a'*64,'b'*64)['status'],'certified')
            self.assertEqual(native_caps(folder,fraction)['native_cost_missing'],0)
            query_files = sorted(folder.rglob('request.json'))
            self.assertEqual(len(query_files),2)  # base is checked, every property remains native
            matrices = []
            for p in query_files:
                req = json.loads(p.read_text())
                raw = json.loads((p.parent/'native_result.json').read_text())
                budget = json.loads((p.parent/'native_budget.json').read_text())
                self.assertEqual(raw['output_budget_fraction'],fraction)
                self.assertEqual(raw['applied_fraction'],fraction)
                self.assertLessEqual(raw['effective_options']['time_limit'],budget['proposed_native_seconds'])
                self.assertLessEqual(budget['proposed_native_seconds'],
                    budget['remaining_before_budget_publication']*fraction)
                self.assertEqual(budget['deadline_monotonic'],req['deadline_monotonic'])
                matrices.append((req['model_sha256'],req['query_sha256'],req['scope']))
            sources.append(matrices)
        self.assertEqual(*sources)

    def test_scope_restores_after_exception(self):
        with checked_execution(self.root,request_sha256='a'*64,input_sha256='b'*64,
                options=CheckedExecutionOptions(expert_base=True)) as cls:
            original=cls.__init__
            with self.assertRaises(RuntimeError):
                with reserve_on_checked_experts(cls,.8):raise RuntimeError('control')
            self.assertIs(cls.__init__,original)
        with self.assertRaises(ValueError):
            with reserve_on_checked_experts(None,.8):pass

    def test_status_one_returns_without_authorizing_infeasibility_and_keeps_worker(self):
        # Exercise the real worker publication and real parent acceptance. The
        # candidate generator alone is synthetic, returning no incumbent.
        folder=self.root/'native';folder.mkdir()
        session=NativeSession(sh._lower_hz_milp(self.hz),folder)
        self.addCleanup(session.close)
        calls=[]
        def fake_milp(**kwargs):
            calls.append(kwargs['options'].copy())
            return SimpleNamespace(status=1,success=False,message='synthetic time limit',x=None)
        stdin=SimpleNamespace(write=lambda s:None,flush=lambda:None,close=lambda:None)
        def start():
            q=folder/f'query_{len(session.queries):03d}'
            req=json.loads((q/'request.json').read_text())
            line=json.dumps({'folder':str(q),'token':req['token']})+'\n'
            with patch('sys.stdin',io.StringIO(line)),patch('scipy.optimize.milp',side_effect=fake_milp):
                serve(session.model_file,session.model_hash,output_budget_fraction=.8)
            if session.proc is None:
                session.proc=SimpleNamespace(stdin=stdin,poll=lambda:None,kill=lambda:None,wait=lambda:0)
        with patch.object(session,'start',side_effect=start):
            for phase in ('base','expanded','contracted'):
                result=session.query(time.monotonic()+5,scope={'phase':phase})
                self.assertEqual(result.status,'unknown')
                self.assertEqual(session.queries[-1]['terminal'],'COMPLETED')
                self.assertIsNotNone(session.proc)
        for i,expected in enumerate((1.,.8,.8)):
            budget=json.loads((folder/f'query_{i:03d}/native_budget.json').read_text())
            self.assertEqual(budget['applied_fraction'],expected)
            self.assertLessEqual(calls[i]['time_limit'],budget['proposed_native_seconds'])


if __name__=='__main__':unittest.main()
