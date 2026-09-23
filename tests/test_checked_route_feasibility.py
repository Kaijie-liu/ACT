import json
import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import threading
import sys
import time
import unittest
from unittest.mock import patch
import numpy as np
import torch
from act.back_end.core import Bounds
from act.back_end.moe import hz_routing as routing
from act.back_end.solver import solver_hz as sh
from act.back_end.solver import checked_route_feasibility as cr
from act.back_end.solver.current_assignment import propose_current_assignment,AssignmentScope


class CheckedRoutingControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='checked-router-',dir='/data1/Kane/MOE')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)
        self.input=sh.sparse_hz_from_bounds(Bounds(torch.tensor([[-1.]],dtype=torch.float64),
            torch.tensor([[1.]],dtype=torch.float64)),frame_id=1229)

    def router(self,weights,bias):return sh.sparse_hz_linear(self.input,np.array(weights,dtype=float),np.array(bias,dtype=float))
    def runner(self,enabled=True):return cr.CheckedRouteFeasibility(self.root/'route',request_sha256='1'*64,input_sha256='2'*64,enabled=enabled)
    def read(self,runner,n=0):return json.loads((runner.root/f'query_{n:03d}/result.json').read_text())

    def test_differential_infeasible_and_feasible_branch(self):
        router=self.router([[.1],[.1]],[-1.,1.]);run=self.runner()
        for i,status in enumerate(('infeasible','feasible')):
            guarded=routing.condition_topk_membership(router,i,1).hz
            old=sh.hz_check_feasibility(guarded,time_limit=2)
            new=run(guarded,time_limit=2)
            self.assertEqual(old.status,status);self.assertEqual(new.status,status)
        self.assertTrue(self.read(run,0)['native_invoked'])
        self.assertFalse(self.read(run,1)['native_invoked'])

    def test_center_not_in_guard_must_not_drop_guard(self):
        router=self.router([[1.],[0.]],[0.,.5]);guard=routing.condition_topk_membership(router,0,1).hz
        run=self.runner();result=run(guard,time_limit=2)
        self.assertEqual(result.status,'feasible');self.assertTrue(self.read(run)['native_invoked'])
        self.assertFalse(self.read(run)['proposal']['check']['accepted'])

    def test_ties_all_routes_and_all_top2_sets_retained(self):
        router=self.router([[0.],[0.],[0.]],[1.,1.,1.])
        with cr.checked_route_feasibility(self.root/'ctx',request_sha256='1'*64,input_sha256='2'*64) as runner:
            with patch.object(sh,'milp',side_effect=AssertionError('tied center suffices')):
                one=routing.analyze_candidates(router,1,time_limit_per_expert=2)
                two=routing.analyze_topk_sets(router,2,time_limit_per_set=2)
        self.assertEqual(one.candidates,(0,1,2));self.assertFalse(one.unresolved)
        self.assertEqual(two.feasible,((0,1),(0,2),(1,2)));self.assertTrue(two.exact)
        self.assertEqual(len(runner.records),6)
        self.assertEqual(len({r['scope']['evaluation_nonce'] for r in runner.records}),6)

    def test_failed_proposal_is_not_infeasible(self):
        run=self.runner()
        with patch.object(cr,'propose_current_assignment',side_effect=ValueError('unsupported encoding')),patch.object(
            sh,'_solve_hz_feasibility',return_value=sh._MILPResult('unknown',None)):
            result=run(self.input,time_limit=2)
        self.assertEqual(result.status,'unknown');self.assertTrue(self.read(run)['native_invoked'])

    def test_wrong_scope_or_matrix_and_partial_point_rejected(self):
        for index,mutate in enumerate((lambda p:replace(p,scope=replace(p.scope,request_sha256='a'*64)),
                lambda p:replace(p,scope=replace(p.scope,input_sha256='b'*64)),
                lambda p:replace(p,scope=replace(p.scope,evaluation_nonce='foreign')),
                lambda p:replace(p,model_sha256='c'*64),lambda p:replace(p,point=p.point[:-1]))):
            run=cr.CheckedRouteFeasibility(self.root/f'case{index}',request_sha256='1'*64,input_sha256='2'*64,enabled=True)
            def propose(m,s,d):return mutate(propose_current_assignment(m,s,d))
            with patch.object(cr,'propose_current_assignment',side_effect=propose),patch.object(
                    sh,'_solve_hz_feasibility',return_value=sh._MILPResult('unknown',None)):
                result=run(self.input,time_limit=2)
            self.assertEqual(result.status,'unknown');self.assertFalse(self.read(run)['proposal']['check']['accepted'])

    def test_disabled_native_and_support_never_shortcut(self):
        run=self.runner(False)
        with patch.object(cr,'propose_current_assignment',side_effect=AssertionError('disabled')):
            self.assertEqual(run(self.input,time_limit=2).status,'feasible')
        self.assertTrue(self.read(run)['native_invoked'])
        support=routing.hz_support_bounds;native=sh.hz_check_feasibility
        with cr.checked_route_feasibility(self.root/'ctx',request_sha256='1'*64,input_sha256='2'*64):
            self.assertIs(routing.hz_support_bounds,support);self.assertIs(sh.hz_check_feasibility,native)
            bound=support(self.input,[0],time_limit=2,relax_binaries=False)
            self.assertEqual(bound.solves,2)

    def test_repeated_query_reconstructs_and_rechecks_no_cache(self):
        run=self.runner()
        with patch.object(cr,'check_current_assignment',wraps=cr.check_current_assignment) as checker:
            for _ in range(2):self.assertEqual(run(self.input,time_limit=2).status,'feasible')
        self.assertEqual(checker.call_count,2)
        self.assertNotEqual(self.read(run,0)['scope'],self.read(run,1)['scope'])

    def test_expired_does_not_query(self):
        run=self.runner()
        with patch.object(sh,'_solve_hz_feasibility',side_effect=AssertionError('expired')):
            self.assertEqual(run(self.input,time_limit=0).status,'unknown')
        self.assertFalse(self.read(run)['native_invoked'])

    def test_saved_point_audit_rejects_scope_and_missing_guard_binding(self):
        run=self.runner();self.assertEqual(run(self.input,time_limit=2).status,'feasible')
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from audit_metamoe_checked_routing_control import routing_query
        args=(run.root/'query_000',0,True,'1'*64,'2'*64,self.read(run)['started_monotonic']-1,{'spans':[]},False)
        self.assertEqual(routing_query(*args)['status'],'feasible')
        with self.assertRaisesRegex(ValueError,'current routing scope'):
            routing_query(*args[:3],'a'*64,*args[4:])
        path=run.root/'query_000/result.json';record=json.loads(path.read_text());record['model_sha256']='b'*64
        path.write_text(json.dumps(record))  # Disposable control artifact only.
        with self.assertRaisesRegex(ValueError,'guarded model identity'):routing_query(*args)

    def test_partial_publication_cannot_return_feasible(self):
        run=self.runner();original=cr.publish
        def broken(p,v):
            if p.name=='result.json':raise OSError('synthetic partial publication')
            original(p,v)
        with patch.object(cr,'publish',side_effect=broken),self.assertRaises(OSError):
            run(self.input,time_limit=2)
        self.assertTrue((run.root/'query_000/proposal.npz').exists())
        self.assertFalse((run.root/'query_000/result.json').exists());self.assertFalse(run.records)

    def test_proposal_publication_cap_falls_back_with_remaining_original_budget(self):
        run=self.runner();clock=[100.];original=cr.publish
        def slow(p,v):
            original(p,v)
            if p.name=='attempt.json':clock[0]=104.
        with patch.object(cr.time,'monotonic',side_effect=lambda:clock[0]),patch.object(cr,'publish',side_effect=slow),patch.object(
                sh,'_solve_hz_feasibility',return_value=sh._MILPResult('unknown',None)) as native:
            result=run(self.input,time_limit=30)
        self.assertEqual(result.status,'unknown');self.assertEqual(native.call_args.args[1],130.)
        self.assertTrue(self.read(run)['proposal_deadline_rejected'])

    def test_same_remaining_native_deadline_after_proposal_failure(self):
        run=self.runner();clock=[100.]
        def slow(*args):clock[0]=101.;raise ValueError('synthetic')
        with patch.object(cr.time,'monotonic',side_effect=lambda:clock[0]),patch.object(cr,'propose_current_assignment',side_effect=slow),patch.object(
                sh,'_solve_hz_feasibility',return_value=sh._MILPResult('unknown',None)) as native:
            self.assertEqual(run(self.input,time_limit=2).status,'unknown')
        self.assertEqual(native.call_args.args[1],102.)
        self.assertEqual(self.read(run)['native_started_monotonic'],101.)

    def test_late_proposal_publication_cannot_accept_point(self):
        run=self.runner();clock=[100.];original=cr.publish
        def late(p,v):
            original(p,v)
            if p.name=='result.json':clock[0]=104.
        with patch.object(cr.time,'monotonic',side_effect=lambda:clock[0]),patch.object(cr,'publish',side_effect=late):
            result=run(self.input,time_limit=10)
        self.assertEqual(result.status,'unknown')
        self.assertTrue((run.root/'query_000/late_result_rejected.json').exists())

    def test_late_native_return_is_explicit_same_old_policy(self):
        run=self.runner(False);clock=[100.]
        def native(*args,**kw):clock[0]=103.;return sh._MILPResult('feasible',np.array([0.]))
        with patch.object(cr.time,'monotonic',side_effect=lambda:clock[0]),patch.object(sh,'_solve_hz_feasibility',side_effect=native):
            result=run(self.input,time_limit=2)
        self.assertEqual(result.status,'feasible')
        self.assertTrue(self.read(run)['native_returned_after_local_deadline'])
        self.assertGreater(result.elapsed,2.)

    def test_bad_tolerance_and_budget_rejected(self):
        run=self.runner()
        for kw in ({'tolerance':1e-5},{'time_limit':31},{'time_limit':float('nan')}):
            with self.assertRaises(ValueError):run(self.input,**kw)

    def test_context_restores_and_rejects_nested_and_threaded(self):
        old=routing.hz_check_feasibility
        with self.assertRaisesRegex(RuntimeError,'synthetic'):
            with cr.checked_route_feasibility(self.root/'ctx',request_sha256='1'*64,input_sha256='2'*64):
                with self.assertRaises(RuntimeError):
                    with cr.checked_route_feasibility(self.root/'other',request_sha256='1'*64,input_sha256='2'*64):pass
                raise RuntimeError('synthetic')
        self.assertIs(routing.hz_check_feasibility,old)
        outcomes=[]
        def secondary():
            try:
                with cr.checked_route_feasibility(self.root/'thread',request_sha256='1'*64,input_sha256='2'*64):pass
            except RuntimeError:outcomes.append('refused')
        t=threading.Thread(target=secondary);t.start();t.join();self.assertEqual(outcomes,['refused'])


class RoutingOuterControls(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import metamoe_checked_routing_control as control
        import audit_metamoe_checked_routing_control as reviewer
        self.control,self.reviewer=control,reviewer
        tmp=tempfile.TemporaryDirectory(prefix='routing-outer-',dir='/data1/Kane/MOE')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)
        self.cfg=control.build();self.cfg['execution_commit']='test-only-ancestor'
        self.cfg['output_root']=str(self.root/'run')
        self.config=self.root/'config.json';self.config.write_text(json.dumps(self.cfg))
        for attr,value in (('CONFIG',self.config),('OUTPUT',self.root/'run')):
            context=patch.object(control,attr,value);context.start();self.addCleanup(context.stop)

    def fixture(self,mode):
        from metamoe_expert_trace import ExpertRecorder
        from recent_moe_deployment import sha256
        def fake(command,cwd,folder,seconds,rss):
            folder.mkdir(parents=True)
            clock=time.monotonic();variant=folder.parent.name
            identity=self.control.identity(self.cfg,variant)
            rec=ExpertRecorder(folder/'trace.jsonl',clock,identity)
            (folder/'runtime.json').write_text(json.dumps({'identity':identity,'worker_started_monotonic':clock}))
            if mode=='COMPLETED':
                rec.emit('WORKER_COMPLETE')
                (folder/'result.json').write_text(json.dumps({'status':'UNKNOWN','request_id':'mnist_0',
                    'arm':'act','config_sha256':sha256(self.config),'worker_seconds':.001,
                    'label':self.cfg['requests'][0]['label'],
                    'tensor_file_sha256':sha256(self.cfg['requests'][0]['tensor_file'])}))
            else:
                rec.emit('BEGIN',name='partial',parent=None,arguments={})
                (folder/'routing/query_000').mkdir(parents=True)
                (folder/'routing/query_000/begin.json').write_text('{}')
                (folder/'result.json').write_text('{"status":')
            rec.close()
            for stream in ('stdout','stderr'):(folder/f'{stream}.txt').write_text('synthetic')
            receipt={'status':mode,'command':command,'deadline_seconds':seconds,
                'execution_including_preflight_seconds':.01,'total_with_postflight_seconds':.011,
                'peak_sampled_group_rss_bytes':1000,'group_rss_limit_bytes':rss,
                'exit_code':0 if mode=='COMPLETED' else -9,'error':None,
                'rss_poll_seconds':.05,'postflight_in_execution_budget':False,
                'rss_is_sampled_not_instantaneous_cap':True,'receipt_own_write_excluded_from_this_clock':True,
                'stdout_sha256':sha256(folder/'stdout.txt'),'stderr_sha256':sha256(folder/'stderr.txt')}
            (folder/'receipt.json').write_text(json.dumps(receipt));time.sleep(.02)
            return receipt
        with patch.object(self.control,'validate'),patch.object(self.control,'require_clean'),patch.object(
                self.control.subprocess,'run'),patch.object(self.control,'supervise',side_effect=fake) as calls:
            self.control.run(self.cfg)
        return calls.call_count

    def review(self):
        with patch.object(self.control,'validate'):return self.reviewer.audit()

    def test_complete_cost_and_no_overwrite(self):
        self.assertEqual(self.fixture('COMPLETED'),2)
        out=self.review();self.assertEqual(out['audit'],'PASS');self.assertEqual(len(out['rows']),2)
        self.assertEqual(out['cost']['charged_request_seconds'],.02)
        with patch.object(self.control,'validate'),patch.object(self.control,'require_clean'),patch.object(
                self.control.subprocess,'run'),patch.object(self.control,'supervise') as call:
            with self.assertRaises(FileExistsError):self.control.run(self.cfg)
            call.assert_not_called()

    def test_outer_deadline_keeps_partial_evidence_not_success(self):
        self.assertEqual(self.fixture('TIMEOUT'),2);out=self.review()
        for row in out['rows']:
            self.assertEqual(row['status'],'TIMEOUT');self.assertIsNone(row['result'])
            self.assertEqual(row['routing'],[{'status':'PARTIAL','accepted':False}])
            self.assertEqual(len(row['right_censored_spans']),1)

    def test_exception_stops_second_arm(self):
        self.assertEqual(self.fixture('ERROR'),1);out=self.review()
        self.assertEqual(len(out['rows']),1);self.assertEqual(out['rows'][0]['status'],'ERROR')
        path=self.control.OUTPUT/'summary.json';summary=json.loads(path.read_text())
        summary['rows'][1]={'variant':'router_checked','status':'UNKNOWN'}
        path.write_text(json.dumps(summary))
        with self.assertRaisesRegex(ValueError,'executed after stop-on-error'):self.review()

    def test_extra_unaccounted_artifact_rejected(self):
        self.fixture('TIMEOUT')
        (self.control.OUTPUT/'router_native/mnist_0_act/routing/query_000/extra').write_text('unbound')
        with self.assertRaisesRegex(ValueError,'evidence inventory'):self.review()

    def test_frozen_configuration_drift_refused(self):
        # Validation must reject changes before any execution. Real source
        # manifests are never rebound for these disposable mutations.
        base=self.control.build()
        for mutate in (lambda c:c.update(seconds=600),lambda c:c.update(proposal_seconds=4),
                       lambda c:c.update(support_unchanged=False),lambda c:c.update(automatic_followup=True)):
            bad=copy.deepcopy(base);mutate(bad)
            with self.assertRaisesRegex(ValueError,'routing control binding drift'):self.control.validate(bad)


if __name__=='__main__':unittest.main()
