import json
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
import numpy as np
import scipy.sparse as sp
import torch
from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.solver import checked_base_session as cb
from act.back_end.solver.current_assignment import AssignmentScope, propose_current_assignment
from act.back_end.solver.isolated_feasibility import NativeSession
from act.back_end.solver.protected_hz_solver import ProtectedHZSolver
from act.front_end.specs import OutputSpec, OutKind
from act.util.stats import VerifyStatus


class CheckedBaseControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='checked-base-',dir='/data1/Kane/MOE')
        self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
        self.hz=sh.sparse_hz_from_bounds(Bounds(torch.tensor([[-1.]],dtype=torch.float64),
            torch.tensor([[1.]],dtype=torch.float64)),frame_id=791)
        self.model=sh._lower_hz_milp(self.hz)
        self.scope=AssignmentScope('1'*64,'2'*64,'current')

    def session(self, model=None):
        folder=self.root/'session';folder.mkdir()
        s=cb.CheckedBaseSession(model or self.model,folder,assignment_scope=self.scope)
        self.addCleanup(s.close);return s

    def evaluate(self, solver, spec=None):
        return solver.evaluate_spec(self.hz,spec or OutputSpec(kind=OutKind.LINEAR_LE,
            c=torch.tensor([[1.],[-1.]],dtype=torch.float64),d=torch.tensor([2.,2.],dtype=torch.float64)),
            batch_size=1,n_out=1,input_hz=self.hz,input_shape=(1,1))[0]

    def test_safe_differential_all_obligations_still_solved(self):
        old=self.evaluate(ProtectedHZSolver(directory=self.root/'native',time_limit=10))
        factory=lambda m,p:cb.CheckedBaseSession(m,p,assignment_scope=self.scope)
        new=self.evaluate(ProtectedHZSolver(directory=self.root/'fast',time_limit=10,session_factory=factory))
        self.assertEqual(old.status,new.status);self.assertEqual(new.status,VerifyStatus.CERTIFIED)
        self.assertEqual(old.metadata['native_queries_started'],3)
        self.assertEqual(new.metadata['native_queries_started'],2)
        self.assertEqual([q['scope'] for q in old.metadata['queries']],[q['scope'] for q in new.metadata['queries']])
        self.assertEqual([r['status'] for r in new.metadata['properties']],['infeasible']*2)
        self.assertEqual(new.metadata['queries'][0]['evidence_kind'],'CURRENT_FULL_MATRIX_POINT')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from audit_metamoe_checked_base_control import evaluate
        audit=evaluate(self.root/'fast',2,self.scope.request_sha256,self.scope.input_sha256)
        self.assertEqual(audit['status'],'certified')
        with self.assertRaisesRegex(ValueError,'current request scope'):
            evaluate(self.root/'fast',2,'3'*64,self.scope.input_sha256)
        # Keep this mutation confined to the disposable test artifact.
        path=self.root/'fast/result.json';record=json.loads(path.read_text())
        record['metadata']['properties'].pop();path.write_text(json.dumps(record))
        with self.assertRaisesRegex(ValueError,'all properties retained'):
            evaluate(self.root/'fast',2,self.scope.request_sha256,self.scope.input_sha256)

    def test_unsafe_still_needs_property_query_and_recoverable_input(self):
        factory=lambda m,p:cb.CheckedBaseSession(m,p,assignment_scope=self.scope)
        spec=OutputSpec(kind=OutKind.LINEAR_LE,c=torch.tensor([[1.]],dtype=torch.float64),
                       d=torch.tensor([-.5],dtype=torch.float64))
        result=self.evaluate(ProtectedHZSolver(directory=self.root/'fast',time_limit=10,session_factory=factory),spec)
        self.assertEqual(result.status,VerifyStatus.FALSIFIED)
        self.assertGreater(result.metadata['native_queries_started'],0)
        self.assertGreater(result.counterexample.item(),-.5)

    def test_center_outside_guard_falls_back_on_same_deadline(self):
        m=replace(self.model,A=sp.csr_matrix([[1.]]),row_lb=np.array([.5]),row_ub=np.array([1.]))
        s=self.session(m);deadline=time.monotonic()+5
        out=s.query(deadline,scope={'phase':'base'})
        self.assertEqual(out.status,'feasible');self.assertGreaterEqual(out.x[0],.5-1e-7)
        self.assertEqual(s.queries[0]['deadline_monotonic'],deadline)
        record=json.loads((s.root/'checked_base/return.json').read_text())
        self.assertFalse(record['check']['accepted']);self.assertEqual(record['returned_status'],'unknown')

    def bad_proposal(self, transform):
        s=self.session();deadline=time.monotonic()+5
        p=propose_current_assignment(self.model,self.scope,deadline)
        with patch.object(cb,'propose_current_assignment',return_value=transform(p)),patch.object(
                NativeSession,'query',return_value=sh._MILPResult('unknown',None)) as fallback:
            out=s.query(deadline,scope={'phase':'base'})
        self.assertEqual(out.status,'unknown');self.assertEqual(fallback.call_args.args[0],deadline)
        self.assertFalse(json.loads((s.root/'checked_base/return.json').read_text())['check']['accepted'])

    def test_foreign_nonce_rejected(self):
        self.bad_proposal(lambda p:replace(p,scope=replace(p.scope,evaluation_nonce='foreign')))
    def test_partial_point_rejected(self):self.bad_proposal(lambda p:replace(p,point=p.point[:-1]))
    def test_invalid_point_rejected(self):self.bad_proposal(lambda p:replace(p,point=np.array([2.])))
    def test_wrong_matrix_rejected(self):self.bad_proposal(lambda p:replace(p,model_sha256='a'*64))

    def test_expired_no_proposal_no_native(self):
        s=self.session()
        with patch.object(cb,'propose_current_assignment',side_effect=AssertionError('no time')) as proposal:
            result=s.query(time.monotonic()-1,scope={'phase':'base'})
        self.assertEqual(result.status,'unknown');self.assertFalse(proposal.called);self.assertEqual(s.launches,0)

    def test_late_publication_rejects_valid_point_and_counts_time(self):
        s=self.session();clock=[100.];original=cb.publish
        def slow(path,value):
            original(path,value)
            if Path(path).name=='return.json':clock[0]=104.
        with patch.object(cb.time,'monotonic',side_effect=lambda:clock[0]),patch.object(cb,'publish',side_effect=slow):
            out=s.query(103.,scope={'phase':'base'})
        self.assertEqual(out.status,'unknown');self.assertEqual(s.launches,0)
        self.assertTrue((s.root/'checked_base/late_return_rejected.json').exists())
        self.assertEqual(s.queries[0]['deadline_monotonic'],103.)

    def test_exception_keeps_partial_evidence_and_remaining_budget(self):
        s=self.session();deadline=time.monotonic()+5
        with patch.object(cb,'propose_current_assignment',side_effect=RuntimeError('synthetic')),patch.object(
                NativeSession,'query',return_value=sh._MILPResult('unknown',None)) as fallback:
            result=s.query(deadline,scope={'phase':'base'})
        self.assertEqual(result.status,'unknown');self.assertEqual(fallback.call_args.args[0],deadline)
        self.assertTrue((s.root/'base_model.npz').exists())
        self.assertIn('synthetic',json.loads((s.root/'checked_base/return.json').read_text())['error'])

    def test_no_attempt_for_property_query(self):
        s=self.session()
        with patch.object(cb,'propose_current_assignment',side_effect=AssertionError('base only')),patch.object(
                NativeSession,'query',return_value=sh._MILPResult('unknown',None)):
            result=s.query(time.monotonic()+1,scope={'phase':'expanded','row':0},
                           extra_A=sp.csr_matrix([[1.]]),extra_lb=np.array([2.]),extra_ub=np.array([np.inf]))
        self.assertEqual(result.status,'unknown');self.assertFalse((s.root/'checked_base').exists())

    def test_publication_exception_cannot_produce_certified(self):
        factory=lambda m,p:cb.CheckedBaseSession(m,p,assignment_scope=self.scope)
        original=cb.publish
        def broken(path,value):
            if Path(path).name=='return.json':raise OSError('synthetic partial publication')
            original(path,value)
        with patch.object(cb,'publish',side_effect=broken):
            result=self.evaluate(ProtectedHZSolver(directory=self.root/'partial',time_limit=10,session_factory=factory))
        self.assertEqual(result.status,VerifyStatus.UNKNOWN)
        self.assertTrue((self.root/'partial/checked_base/candidate.npz').exists())
        self.assertTrue(all(p['status']=='NOT_STARTED' for p in result.metadata['properties']))

    def test_no_second_proposal_and_no_relaxed_tolerance(self):
        s=self.session();deadline=time.monotonic()+5
        self.assertEqual(s.query(deadline,scope={'phase':'base'}).status,'feasible')
        with patch.object(cb,'propose_current_assignment',side_effect=AssertionError('one only')),patch.object(
                NativeSession,'query',return_value=sh._MILPResult('unknown',None)) as native:
            self.assertEqual(s.query(deadline,scope={'phase':'base'}).status,'unknown')
            self.assertTrue(native.called)
        with self.assertRaisesRegex(ValueError,'original tolerance'):
            s.query(deadline,scope={'phase':'base'},tolerance=1e-5)

    def test_base_feasible_cannot_skip_unresolved_properties(self):
        def native(this,deadline,**kwargs):
            this.queries.append({'scope':kwargs['scope'],'native_started':False});return sh._MILPResult('unknown',None)
        with (cb.checked_base_experts(self.root/'ctx',request_sha256='1'*64,input_sha256='2'*64) as cls,
                patch.object(NativeSession,'query',native)):
            result=self.evaluate(cls(time_limit=10))
        self.assertEqual(result.status,VerifyStatus.UNKNOWN);self.assertEqual(result.metadata['base_status'],'feasible')
        self.assertEqual(len(result.metadata['properties']),2)

    def test_context_restores_on_exception_and_rejects_nesting(self):
        old=sh.HZSolver
        with self.assertRaisesRegex(RuntimeError,'outer'):
            with cb.checked_base_experts(self.root,request_sha256='1'*64,input_sha256='2'*64):
                with self.assertRaises(RuntimeError):
                    with cb.checked_base_experts(self.root,request_sha256='1'*64,input_sha256='2'*64):pass
                raise RuntimeError('outer')
        self.assertIs(sh.HZSolver,old)


if __name__=='__main__':unittest.main()
