import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch,Mock
import numpy as np
from scipy import sparse
import torch
from act.back_end.core import Bounds
from act.back_end.solver import solver_hz as sh
from act.back_end.solver import checked_nonzero_support as nz
from act.back_end.moe import class_separated_top1 as entry


class NonzeroControls(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='nonzero-control-',dir='/data1/Kane/MOE')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name)

    def box(self,lo,hi):
        return sh.sparse_hz_from_bounds(Bounds(torch.tensor([lo],dtype=torch.float64),
            torch.tensor([hi],dtype=torch.float64)),frame_id=975)

    def runner(self,enabled=True,fallback=None,name='run'):
        return nz.CheckedNonzeroSupport(self.root/name,request_sha256='1'*64,input_sha256='2'*64,
            enabled=enabled,fallback=fallback or entry.selected_score_support)

    def read(self,run,index=0):return json.loads((run.root/f'query_{index:03d}/result.json').read_text())

    def test_both_signs_keep_original_fallback_range_no_solver(self):
        for k,(lo,hi) in enumerate(((2.,4.),(-4.,-2.),(.25,.25),(-.25,-.25))):
            hz=self.box([lo],[hi]);run=self.runner(name=str(k));old=sh.hz_support_bounds(hz,[0],time_limit=0,relax_binaries=False)
            with patch.object(sh,'milp',side_effect=AssertionError('no native solve')):
                out=run(hz,0,time_limit=2)
            self.assertTrue(torch.equal(old.bounds.lb,out.bounds.lb));self.assertTrue(torch.equal(old.bounds.ub,out.bounds.ub))
            self.assertEqual(out.solves,0);self.assertFalse(out.exact)
            self.assertEqual(self.read(run)['evidence'],'CURRENT_GENERATOR_BOX_NONZERO')

    def test_zero_touching_and_crossing_fall_back(self):
        for k,(lo,hi) in enumerate(((0.,0.),(0.,1.),(-1.,0.),(-1.,1.))):
            hz=self.box([lo],[hi]);fallback=Mock(wraps=entry.selected_score_support)
            run=self.runner(fallback=fallback,name=str(k));out=run(hz,0,time_limit=2)
            self.assertTrue(self.read(run)['native_invoked']);fallback.assert_called_once()
            self.assertFalse(nz.nonzero(out.bounds.lb.item(),out.bounds.ub.item()))

    def test_guard_only_positivity_needs_native_unchanged(self):
        hz=self.box([-1.],[1.]);hz.Auc=sparse.csr_matrix([[-1.]]);hz.Aub=sparse.csr_matrix((1,0));hz.ub=np.array([-.5])
        run=self.runner();out=run(hz,0,time_limit=2)
        self.assertTrue(self.read(run)['native_invoked']);self.assertGreater(out.bounds.lb.item(),0)
        self.assertEqual(out.lower_status,('milp_optimal',))

    def test_binary_factor_sign_and_all_rows(self):
        hz=sh.SparseHZono(np.array([3.,-3.,0.]),sparse.csr_matrix((3,0)),sparse.csr_matrix([[1.],[1.],[1.]]),
            sparse.csr_matrix((0,0)),sparse.csr_matrix((0,1)),np.zeros(0),frame_id=3)
        run=self.runner()
        for row in range(3):run(hz,row,time_limit=2)
        self.assertEqual([r['native_invoked'] for r in run.records],[False,False,True])
        self.assertEqual(len({r['scope']['evaluation_nonce'] for r in run.records}),3)

    def test_duplicate_generators_conservative_exact_sign(self):
        hz=self.box([2.],[4.]);hz.Gc=sparse.csr_matrix((np.array([1.,-1.]),np.array([0,0]),np.array([0,2])),shape=(1,1))
        # Separate duplicate occurrences produce a conservative radius, never a
        # falsely stronger sign. Exact checker acts before scipy normalization.
        sig=nz.exact_generator_sign(hz,0,time.monotonic()+2)
        self.assertEqual(sig['lower'],'1');self.assertEqual(sig['upper'],'5')

    def test_exact_sign_disagreement_only_vetoes(self):
        hz=self.box([-1.],[1.]);fake=sh.HZSupportBoundsResult((0,),Bounds(torch.tensor([[1.]]),torch.tensor([[2.]])),
            ('fast_fallback',),('fast_fallback',),(None,),0.,0,False)
        fallback=Mock(return_value=nz.unknown(0,time.monotonic()));run=self.runner(fallback=fallback)
        with patch.object(sh,'hz_support_bounds',return_value=fake):out=run(hz,0,time_limit=2)
        fallback.assert_called_once();self.assertEqual(self.read(run)['exact_sign']['sign'],0)
        self.assertFalse(nz.describe(out)['accepted'])

    def test_wrong_fast_row_and_nonfinite_refuse_shortcut(self):
        hz=self.box([2.],[3.]);fallback=Mock(return_value=nz.unknown(0,time.monotonic()))
        good=sh.hz_support_bounds(hz,[0],time_limit=0,relax_binaries=False)
        for k,fast in enumerate((replace(good,rows=(1,)),replace(good,bounds=Bounds(
                torch.tensor([[float('nan')]]),torch.tensor([[float('inf')]]))))):
            run=self.runner(fallback=fallback,name=str(k))
            with patch.object(sh,'hz_support_bounds',return_value=fast):run(hz,0,time_limit=2)
            self.assertNotEqual(self.read(run)['evidence'],'CURRENT_GENERATOR_BOX_NONZERO')
        self.assertEqual(fallback.call_count,2)

    def test_changed_hz_recomputed_and_no_cross_request_cache(self):
        hz=self.box([2.],[3.]);run=self.runner();run(hz,0,time_limit=2)
        hz.c[0]=-hz.c[0];run(hz,0,time_limit=2)
        a,b=[self.read(run,i) for i in (0,1)]
        self.assertNotEqual(a['source_hz_sha256'],b['source_hz_sha256'])
        self.assertEqual([a['exact_sign']['sign'],b['exact_sign']['sign']],[1,-1])

    def test_disabled_and_dense_use_native(self):
        hz=self.box([2.],[3.]);run=self.runner(False)
        with patch.object(nz,'exact_generator_sign',side_effect=AssertionError('disabled')):run(hz,0,time_limit=2)
        self.assertTrue(self.read(run)['native_invoked'])
        dense=sh.sparse_hz_to_dense(hz);run2=self.runner(name='dense');out=run2(dense,0,time_limit=2)
        self.assertTrue(nz.describe(out)['accepted']);self.assertTrue(self.read(run2)['native_invoked'])

    def test_fallback_uses_original_remaining_budget(self):
        clock=[100.];hz=self.box([0.],[1.]);original=nz.publish;fallback=Mock(return_value=nz.unknown(0,time.monotonic()))
        def slow(p,r):
            original(p,r)
            if p.name=='attempt.json':clock[0]=101.
        run=self.runner(fallback=fallback)
        with patch.object(nz.time,'monotonic',side_effect=lambda:clock[0]),patch.object(nz,'publish',side_effect=slow):
            run(hz,0,time_limit=30)
        self.assertEqual(fallback.call_args.kwargs['time_limit'],29.)

    def test_precheck_cap_and_late_publication_no_acceptance(self):
        for k,target in enumerate(('attempt.json','result.json')):
            clock=[100.];original=nz.publish;fallback=Mock(return_value=nz.unknown(0,time.monotonic()))
            def slow(p,r):
                original(p,r)
                if p.name==target:clock[0]=104.
            run=self.runner(fallback=fallback,name=str(k))
            with patch.object(nz.time,'monotonic',side_effect=lambda:clock[0]),patch.object(nz,'publish',side_effect=slow):
                out=run(self.box([2.],[3.]),0,time_limit=30)
            self.assertFalse(nz.describe(out)['accepted'])
            if target=='attempt.json':self.assertEqual(fallback.call_args.kwargs['time_limit'],26.)
            else:self.assertTrue((run.root/'query_000/late_result_rejected.json').exists());fallback.assert_not_called()

    def test_expired_or_partial_publication_cannot_accept(self):
        run=self.runner();out=run(self.box([2.],[3.]),0,time_limit=0)
        self.assertFalse(nz.describe(out)['accepted']);self.assertFalse(self.read(run)['native_invoked'])
        original=nz.publish
        def broken(p,r):
            if p.name=='result.json':raise OSError('partial')
            original(p,r)
        run2=self.runner(name='broken')
        with patch.object(nz,'publish',side_effect=broken),self.assertRaises(OSError):run2(self.box([2.],[3.]),0,time_limit=2)
        self.assertFalse(run2.records);self.assertFalse((run2.root/'query_000/result.json').exists())

    def test_deadline_inside_exact_check(self):
        with self.assertRaises(TimeoutError):nz.exact_generator_sign(self.box([2.],[3.]),0,time.monotonic()-1)

    def test_hook_only_definedness_restores_on_error(self):
        original=entry.selected_score_support;general=sh.hz_support_bounds
        with self.assertRaisesRegex(ValueError,'synthetic'):
            with nz.checked_nonzero_support(self.root/'ctx',request_sha256='1'*64,input_sha256='2'*64):
                self.assertIs(sh.hz_support_bounds,general)
                with self.assertRaises(RuntimeError):
                    with nz.checked_nonzero_support(self.root/'nest',request_sha256='1'*64,input_sha256='2'*64):pass
                raise ValueError('synthetic')
        self.assertIs(entry.selected_score_support,original)
        seen=[]
        def second():
            try:
                with nz.checked_nonzero_support(self.root/'thread',request_sha256='1'*64,input_sha256='2'*64):pass
            except RuntimeError:seen.append('refused')
        t=threading.Thread(target=second);t.start();t.join();self.assertEqual(seen,['refused'])

    def test_default_calls_original_configuration(self):
        hz=self.box([2.],[3.])
        with patch.object(sh,'hz_support_bounds',wraps=sh.hz_support_bounds) as f:
            entry.selected_score_support(hz,0,time_limit=2)
        self.assertEqual(f.call_args.args,(hz,[0]));self.assertEqual(f.call_args.kwargs,{'time_limit':2,'relax_binaries':False})

    def test_saved_audit_scope_guard_and_sign_mutations(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from audit_metamoe_nonzero_precheck_control import nonzero_query
        hz=self.box([2.],[3.]);run=self.runner();run(hz,0,time_limit=2)
        model=sh._lower_hz_milp(hz)
        z={**nz.csr_arrays(model.A),'row_lb':model.row_lb,'row_ub':model.row_ub,'var_lb':model.var_lb,'var_ub':model.var_ub,
            'integrality':model.integrality,'value_center':model.value_center,
            **{'value_'+k:v for k,v in nz.csr_arrays(model.value_matrix).items()}}
        args=(run.root/'query_000',0,True,'1'*64,'2'*64,0,{'spans':[]},False,z)
        with patch.object(nz,'exact_generator_sign',side_effect=AssertionError('independent checker')):
            self.assertTrue(nonzero_query(*args)['accepted'])
        with self.assertRaisesRegex(ValueError,'request binding'):nonzero_query(*args[:3],'3'*64,*args[4:])
        bad=copy.deepcopy(z);bad['value_center'][0]+=1
        with self.assertRaisesRegex(ValueError,'guarded routing object'):nonzero_query(*args[:-1],bad)
        path=run.root/'query_000/result.json';r=json.loads(path.read_text());r['exact_sign']['lower']='-1';path.write_text(json.dumps(r))
        with self.assertRaisesRegex(ValueError,'exact sign'):nonzero_query(*args)

    def test_saved_native_support_trace_audits(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        from metamoe_expert_trace import ExpertRecorder,install,restore
        from audit_conv_f0_timing import check_trace
        from audit_metamoe_nonzero_precheck_control import nonzero_query
        clock=time.monotonic();rec=ExpertRecorder(self.root/'trace.jsonl',clock,{'control':'native'})
        patches=install(rec)
        try:
            run=self.runner(False);run(self.box([2.],[3.]),0,time_limit=2)
            rec.emit('WORKER_COMPLETE')
        finally:restore(patches);rec.close()
        trace=check_trace(self.root/'trace.jsonl',time.monotonic()-clock,{'control':'native'})
        value=nonzero_query(run.root/'query_000',0,False,'1'*64,'2'*64,clock,trace,False)
        self.assertTrue(value['accepted']);self.assertTrue(value['native_invoked'])

    def test_all_tie_legal_definedness_obligations_use_scoped_hook(self):
        from tests.test_class_separated_top1 import affine
        from act.util.device_manager import initialize_device
        dtype=torch.get_default_dtype();self.addCleanup(torch.set_default_dtype,dtype)
        initialize_device('cpu','float64')
        model=entry.ClassSeparatedTop1(affine([0.,0.],[1.,1.]),
            [affine([0.,0.,0.],[2.,1.,.5]),affine([0.,0.],[3.,2.])],(3,2))
        center=torch.zeros((1,1),dtype=torch.float64)
        with nz.checked_nonzero_support(self.root/'ctx',request_sha256='1'*64,input_sha256='2'*64) as run:
            result=entry.verify_class_separated_box(model,center=center,lower=center-.1,upper=center+.1,
                rows=torch.ones((1,5),dtype=torch.float64),thresholds=torch.tensor([1e-7],dtype=torch.float64),total_seconds=20)
        self.assertEqual(result['status'],'POSITIVE');self.assertEqual(result['candidates'],[0,1])
        self.assertEqual([r['row'] for r in run.records],[0,1]);self.assertTrue(all(not r['native_invoked'] for r in run.records))

    def test_nonfinite_source_and_unsupported_checker_fall_back(self):
        for k,value in enumerate((float('nan'),float('inf'))):
            hz=self.box([2.],[3.]);hz.c[0]=value
            fallback=Mock(return_value=nz.unknown(0,time.monotonic()));run=self.runner(fallback=fallback,name=str(k))
            run(hz,0,time_limit=2);fallback.assert_called_once()
            self.assertFalse(self.read(run)['result']['accepted'])
        run=self.runner(name='fail',fallback=Mock(return_value=nz.unknown(0,time.monotonic())))
        with patch.object(nz,'exact_generator_sign',side_effect=TimeoutError('synthetic')):run(self.box([2.],[3.]),0,time_limit=2)
        self.assertTrue(self.read(run)['native_invoked'])

    def test_unresolved_full_entry_remains_json_serializable(self):
        from tests.test_class_separated_top1 import affine
        from act.util.device_manager import initialize_device
        dtype=torch.get_default_dtype();self.addCleanup(torch.set_default_dtype,dtype)
        initialize_device('cpu','float64')
        model=entry.ClassSeparatedTop1(affine([0.,0.],[1.,-1.]),
            [affine([0.,0.],[2.,1.]),affine([0.],[3.])],(2,1))
        x=torch.zeros((1,1),dtype=torch.float64)
        with patch.object(entry,'selected_score_support',return_value=nz.unknown(0,time.monotonic())):
            result=entry.verify_class_separated_box(model,center=x,lower=x-.1,upper=x+.1,
                rows=entry.classification_rows(3,0),thresholds=torch.full((2,),1e-7,dtype=torch.float64),total_seconds=20)
        self.assertEqual(result['status'],'UNKNOWN');self.assertIsNone(result['nonzero_obligations'][0]['lower'])
        json.dumps(result,allow_nan=False)


class OuterControls(unittest.TestCase):
    def setUp(self):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
        import metamoe_nonzero_precheck_control as control
        import audit_metamoe_nonzero_precheck_control as reviewer
        self.control,self.reviewer=control,reviewer
        tmp=tempfile.TemporaryDirectory(prefix='nonzero-outer-',dir='/data1/Kane/MOE')
        self.addCleanup(tmp.cleanup);self.root=Path(tmp.name);self.cfg=control.build()
        self.cfg['execution_commit']='test-only';self.cfg['output_root']=str(self.root/'run')
        self.config=self.root/'config.json';self.config.write_text(json.dumps(self.cfg))
        for name,value in (('CONFIG',self.config),('OUTPUT',self.root/'run')):
            ctx=patch.object(control,name,value);ctx.start();self.addCleanup(ctx.stop)

    def fixture(self,mode):
        from metamoe_expert_trace import ExpertRecorder
        from recent_moe_deployment import sha256
        def fake(command,cwd,folder,seconds,rss):
            folder.mkdir(parents=True);clock=time.monotonic();identity=self.control.identity(self.cfg,folder.parent.name)
            rec=ExpertRecorder(folder/'trace.jsonl',clock,identity)
            (folder/'runtime.json').write_text(json.dumps({'identity':identity,'worker_started_monotonic':clock}))
            if mode=='COMPLETED':
                rec.emit('WORKER_COMPLETE')
                (folder/'result.json').write_text(json.dumps({'status':'UNKNOWN','request_id':'mnist_0','arm':'act',
                    'config_sha256':sha256(self.config),'worker_seconds':.001,'label':self.cfg['requests'][0]['label'],
                    'tensor_file_sha256':sha256(self.cfg['requests'][0]['tensor_file'])}))
            else:
                rec.emit('BEGIN',name='partial',parent=None,arguments={})
                (folder/'nonzero/query_000').mkdir(parents=True)
                (folder/'nonzero/query_000/begin.json').write_text(json.dumps({'row':1}))
                (folder/'result.json').write_text('{"status":')
            rec.close()
            for s in ('stdout','stderr'):(folder/f'{s}.txt').write_text('synthetic')
            receipt={'status':mode,'command':command,'deadline_seconds':seconds,
                'execution_including_preflight_seconds':.01,'total_with_postflight_seconds':.011,
                'peak_sampled_group_rss_bytes':1000,'group_rss_limit_bytes':rss,
                'exit_code':0 if mode=='COMPLETED' else -9,'error':None,'rss_poll_seconds':.05,
                'postflight_in_execution_budget':False,'rss_is_sampled_not_instantaneous_cap':True,
                'receipt_own_write_excluded_from_this_clock':True,
                'stdout_sha256':sha256(folder/'stdout.txt'),'stderr_sha256':sha256(folder/'stderr.txt')}
            (folder/'receipt.json').write_text(json.dumps(receipt));time.sleep(.02);return receipt
        with patch.object(self.control,'validate'),patch.object(self.control,'require_clean'),patch.object(
                self.control.subprocess,'run'),patch.object(self.control,'supervise',side_effect=fake) as calls:
            self.control.run(self.cfg)
        return calls.call_count

    def review(self):
        with patch.object(self.control,'validate'):return self.reviewer.audit()

    def test_complete_cost_and_no_overwrite(self):
        self.assertEqual(self.fixture('COMPLETED'),2);v=self.review()
        self.assertEqual(v['audit'],'PASS');self.assertEqual(v['cost']['charged_request_seconds'],.02)
        with patch.object(self.control,'validate'),patch.object(self.control,'require_clean'),patch.object(
                self.control.subprocess,'run'),patch.object(self.control,'supervise') as calls:
            with self.assertRaises(FileExistsError):self.control.run(self.cfg)
            calls.assert_not_called()

    def test_outer_timeout_preserves_partial_not_positive(self):
        self.assertEqual(self.fixture('TIMEOUT'),2)
        for r in self.review()['rows']:
            self.assertEqual(r['status'],'TIMEOUT');self.assertEqual(r['nonzero'],[{'accepted':False,'status':'PARTIAL'}])
            self.assertEqual(len(r['right_censored_spans']),1)

    def test_error_stops_second_arm(self):
        self.assertEqual(self.fixture('ERROR'),1);self.assertEqual(len(self.review()['rows']),1)

    def test_extra_evidence_and_configuration_tampering_rejected(self):
        self.fixture('TIMEOUT');(self.control.OUTPUT/'support_native/mnist_0_act/nonzero/unbound').write_text('extra')
        with self.assertRaisesRegex(ValueError,'evidence inventory'):self.review()
        base=self.control.build()
        for mutate in (lambda c:c.update(seconds=600),lambda c:c.update(nonzero_precheck_seconds=4),
                       lambda c:c.update(router_checked_both=False),lambda c:c.update(automatic_followup=True)):
            bad=copy.deepcopy(base);mutate(bad)
            with self.assertRaisesRegex(ValueError,'binding drift'):self.control.validate(bad)


if __name__=='__main__':unittest.main()
