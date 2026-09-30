"""Bounded kernel controls, fake native backend only. No new solver calls."""
from copy import deepcopy
from fractions import Fraction as F
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from scoped_source import rowwise_bound as new
from scoped_source import rowwise_native as native
from scoped_source.sparse_check import check_bound as old_check
from scoped_proof.io import ROOT, save, sha
from scripts.check_h2_rowwise import derive, legacy_native, FILES, CONFIG, PROTOCOL_SHA


def fixture(offset='1/5'):
    lp={'matrix_format':'csr_v1','c':[0.1,'-2/7'],'offset':offset,'lower':['-1','0'],'upper':['2','3'],
        'A':{'shape':[3,2],'data':['1','2','-1','-0.0'],'indices':[0,1,0,1],'indptr':[0,2,2,4]},
        'b':['4','0','1'], 'E':{'shape':[1,2],'data':['1/3','0.1'],'indices':[0,1],'indptr':[0,2]},'h':['1']}
    cert={'lp_sha256':new.identity(lp),'inequality_dual':[-1,0,0],'equality_dual':[2],'claimed_lower_bound':'-1000'}
    return lp,cert


def empty(n):
    return {'shape':[0,n],'data':[],'indices':[],'indptr':[0]}


class RowwiseControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keep=os.environ.get('H2_ROWWISE_ROOT')
        if keep:
            cls.root=Path(keep)
            if not cls.root.resolve().is_relative_to(Path('/data1/Kane/MOE')): raise ValueError('output outside project')
            cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp=tempfile.TemporaryDirectory(prefix='h2-rowwise-',dir=ROOT/'data/moe/tmp'); cls.root=Path(cls.tmp.name)
        if sha(ROOT/CONFIG)!=PROTOCOL_SHA: raise ValueError('control protocol changed')
        names=set(FILES)|{'act/back_end/solver/lp_certificate.py','act/back_end/solver/sparse_lp_certificate.py',
                          'scoped_source/sparse_check.py','upstream_source/checker.py','source_enclosure/format.py'}
        bindings={}
        for name in sorted(names):
            p=cls.root/'implementation'/name; p.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,p); bindings[name]=sha(p)
        save(cls.root/'implementation.json',bindings); save(cls.root/'protocol.json',json.loads((ROOT/CONFIG).read_text()))
        cls.old=legacy_native()

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def verify(self,lp,cert):
        return new.check_bound(lp,cert,deadline=time.monotonic()+30)

    def test_full_archived_inventory_not_only_positive(self):
        result=derive(); save(self.root/'archive_differential.json',result)
        self.assertEqual((result['target_lps'],result['dual_certificates'],result['box_facts']),(75,72,3))
        self.assertEqual((result['positive_bounds'],result['negative_bounds'],result['zero_bounds']),(59,13,0))
        self.assertEqual(result['new_solves'],0)

    def test_exact_bound_and_residual_three_kernels(self):
        records=[]
        for offset in ('1/5','10'):
            lp,cert=fixture(offset); original=deepcopy((lp,cert))
            expected,residual=self.old.evaluate(lp,cert); actual=self.verify(lp,cert)
            self.assertEqual(actual['checked_lower_bound'],str(expected))
            self.assertEqual(actual['residual'],list(map(str,residual)))
            self.assertEqual(actual['checked_lower_bound'],old_check(lp,cert)['checked_lower_bound'])
            self.assertEqual((actual['rows_checked'],actual['entries_checked'],actual['zero_dual_rows_checked']),(4,6,2))
            self.assertEqual((lp,cert),original)
            records.append(actual)
        self.assertLess(F(records[0]['checked_lower_bound']),0); self.assertGreater(F(records[1]['checked_lower_bound']),0)
        save(self.root/'exact_differential.json',records)

    def test_empty_matrices_and_zero_width_box_binary64(self):
        lp={'matrix_format':'csr_v1','c':[0.1],'offset':'0','lower':['1'],'upper':['1'],
            'A':empty(1),'b':[],'E':empty(1),'h':[]}
        cert={'lp_sha256':new.identity(lp),'inequality_dual':[],'equality_dual':[],'claimed_lower_bound':'0'}
        result=self.verify(lp,cert)
        self.assertEqual(F(result['checked_lower_bound']),F.from_float(0.1))
        self.assertNotEqual(F(result['checked_lower_bound']),F(1,10))
        self.assertEqual(result['checked_lower_bound'],old_check(lp,cert)['checked_lower_bound'])
        self.assertEqual(result['rows_checked'],0)

    def test_malformed_final_zero_dual_row_rejected_after_rebinding(self):
        mutations=[
            lambda p:p['A']['indices'].__setitem__(-1,0),
            lambda p:p['A']['indices'].__setitem__(-1,2),
            lambda p:p['A']['indices'].__setitem__(-1,True),
            lambda p:p['A']['data'].__setitem__(-1,'not-rational'),
            lambda p:p['A']['data'].__setitem__(-1,'1/0'),
            lambda p:p['A']['shape'].__setitem__(1,True),
            lambda p:p['A']['shape'].__setitem__(0,-1),
            lambda p:p['A']['indptr'].__setitem__(-1,True),
            lambda p:p['A']['indptr'].__setitem__(0,1),
            lambda p:p['A']['indptr'].__setitem__(2,1),
            lambda p:p['A']['indptr'].pop(),
            lambda p:p['A']['data'].pop(),
            lambda p:p['b'].pop(),
            lambda p:p['b'].__setitem__(-1,'bad'),
            lambda p:p['upper'].__setitem__(0,'-2'),
            lambda p:p['c'].__setitem__(0,True),
        ]
        records=[]
        for i,change in enumerate(mutations):
            with self.subTest(mutation=i):
                lp,cert=fixture(); change(lp); cert['lp_sha256']=new.identity(lp)
                for checker in (self.verify,old_check):
                    with self.assertRaises((ValueError,ZeroDivisionError)): checker(lp,cert)
                with self.assertRaises((ValueError,ZeroDivisionError)): native.validate_native(lp,deadline=time.monotonic()+30)
                records.append({'mutation':i,'new_and_legacy_source_rejected':True,'native_prevalidation_rejected':True})
        save(self.root/'malformed_controls.json',records)

    def test_binding_duals_claim_and_nonfinite_rejected(self):
        changes=[lambda c:c.update(lp_sha256='0'*64), lambda c:c.update(claimed_lower_bound='100000'),
                 lambda c:c['inequality_dual'].__setitem__(-1,1),lambda c:c['inequality_dual'].pop(),
                 lambda c:c['equality_dual'].append(0),lambda c:c['inequality_dual'].__setitem__(-1,True)]
        for change in changes:
            lp,cert=fixture(); change(cert)
            for check in (self.verify,old_check):
                with self.assertRaises(ValueError): check(lp,cert)
        for bad in (float('nan'),float('inf'),float('-inf')):
            lp,cert=fixture(); lp['A']['data'][-1]=bad
            with self.assertRaises(ValueError): self.verify(lp,cert)
            with self.assertRaises(ValueError): native.validate_native(lp,deadline=time.monotonic()+30)

    def test_missing_required_fields_never_partial_acceptance(self):
        for key in ('offset','A','h'):
            lp,cert=fixture(); del lp[key]; cert['lp_sha256']=new.identity(lp)
            with self.assertRaises(ValueError): self.verify(lp,cert)
            with self.assertRaises(ValueError): native.validate_native(lp,deadline=time.monotonic()+30)
        lp,cert=fixture(); del cert['claimed_lower_bound']
        with self.assertRaises(ValueError): self.verify(lp,cert)

    def test_deadline_at_every_check_tick_no_return(self):
        lp,cert=fixture(); calls=[]
        def clock(_):
            return lambda:calls.append(None)
        with patch.object(new,'clock',clock): self.verify(lp,cert)
        count=len(calls)
        for limit in range(1,count+1):
            counter=0
            def tick():
                nonlocal counter
                counter+=1
                if counter==limit: raise TimeoutError('injected exact cutoff')
            with patch.object(new,'clock',lambda _:tick):
                with self.assertRaisesRegex(TimeoutError,'injected exact cutoff'): self.verify(lp,cert)
        save(self.root/'deadline_checkpoints.json',{'exact_check_tick_positions':count,'all_rejected':True,'hard_supervisor':False})

    def test_real_expired_and_invalid_deadlines(self):
        lp,cert=fixture()
        for function,args in ((new.check_bound,(lp,cert)),(native.validate_native,(lp,)),(native.propose,(lp,))):
            with self.assertRaises(TimeoutError): function(*args,deadline=time.monotonic()-1)
            for value in (True,float('inf'),float('nan'),time.monotonic()+301):
                with self.assertRaises(ValueError): function(*args,deadline=value)

    def test_alias_pollution_lp_and_certificate_rejected(self):
        for target in ('lp','cert'):
            lp,cert=fixture(); count=0
            def tick():
                nonlocal count
                count+=1
                if count==2:
                    if target=='lp': lp['offset']='9'
                    else: cert['inequality_dual'][0]=-2
            with patch.object(new,'clock',lambda _:tick):
                with self.assertRaisesRegex(ValueError,'changed during check'): self.verify(lp,cert)
        lp,_=fixture(); count=0
        def tick():
            nonlocal count
            count+=1
            if count==2: lp['offset']='9'
        with patch.object(native,'_clock',lambda _:tick):
            with self.assertRaisesRegex(ValueError,'changed during native validation'):
                native.validate_native(lp,deadline=time.monotonic()+30)

    def test_row_objects_not_retained_as_full_matrix(self):
        lp,cert=fixture(); n=100
        lp['A']={'shape':[n,2],'data':['1']*n,'indices':[0]*n,'indptr':list(range(n+1))}
        lp['b']=['0']*n; cert['inequality_dual']=[0]*n; cert['lp_sha256']=new.identity(lp)
        observations=[]
        for module,name,call in ((new,'rows',lambda:self.verify(lp,cert)),
                                  (native,'native_rows',lambda:native.validate_native(lp,deadline=time.monotonic()+30))):
            active=peak=0; original=getattr(module,name)
            class Tracked(list):
                def __init__(self,values):
                    nonlocal active,peak
                    super().__init__(values); active+=1; peak=max(peak,active)
                def __del__(self):
                    nonlocal active
                    active-=1
            def wrapped(*args,**kwargs):
                for row in original(*args,**kwargs):
                    current=Tracked(row); yield current; del current
            with patch.object(module,name,wrapped): call()
            # enumerate may hold its preceding result while next() builds the
            # following row. Two rows is still bounded by width, not row count.
            self.assertEqual(active,0); self.assertLessEqual(peak,2); observations.append({'path':name,'peak_tracked_rows':peak})
        save(self.root/'row_lifetimes.json',{'observations':observations,'measured_RSS':False,'full_path_capacity':False})

    def fake_backend(self,lp,*,success=True,late=False,exception=False,mutate=False,bad_dual=False):
        calls=[]; sparse=ModuleType('scipy.sparse'); optimize=ModuleType('scipy.optimize')
        def matrix(args,shape):
            return {'data':list(args[0]),'indices':list(args[1]),'indptr':list(args[2]),'shape':shape}
        def linprog(c,**kwargs):
            calls.append({'c':c,**kwargs})
            if exception: raise RuntimeError('controlled fake native error')
            if mutate: lp['offset']='19'
            if late: self.fake_now+=31
            y=[0.]*len(lp['b']); z=[0.]*len(lp['h'])
            if bad_dual: y[-1]=float('nan')
            return SimpleNamespace(success=success,ineqlin=SimpleNamespace(marginals=y),eqlin=SimpleNamespace(marginals=z))
        sparse.csr_matrix=matrix; optimize.linprog=linprog
        return {'scipy':ModuleType('scipy'),'scipy.sparse':sparse,'scipy.optimize':optimize},calls

    def test_native_stub_preserves_matrix_and_remaining_clock(self):
        lp,_=fixture(); modules,calls=self.fake_backend(lp)
        with patch.dict(sys.modules,modules): cert=native.propose(lp,deadline=time.monotonic()+30)
        self.assertEqual(len(calls),1); self.assertEqual(cert['lp_sha256'],new.identity(lp))
        c=calls[0]; self.assertEqual(c['A_ub']['indptr'],lp['A']['indptr'])
        self.assertEqual(c['A_ub']['indices'],lp['A']['indices'])
        self.assertEqual(c['A_ub']['data'],list(map(float,lp['A']['data'])))
        self.assertTrue(0<c['options']['time_limit']<=30); self.assertEqual(c['method'],'highs')
        self.assertEqual(self.verify(lp,cert)['checked_lower_bound'],old_check(lp,cert)['checked_lower_bound'])
        save(self.root/'native_stub.json',{'calls':calls,'certificate':cert,'real_solves':0})

    def test_native_failures_and_late_result_never_return_candidate(self):
        for option in ('success','exception','mutate','bad_dual','late'):
            lp,_=fixture(); self.fake_now=time.monotonic()
            options={option:False if option=='success' else True}; modules,_=self.fake_backend(lp,**options)
            with patch.dict(sys.modules,modules),patch.object(native.time,'monotonic',lambda:self.fake_now):
                with self.assertRaises((ValueError,RuntimeError,TimeoutError)):
                    native.propose(lp,deadline=self.fake_now+30)

    def test_native_deadline_each_validation_tick(self):
        lp,_=fixture(); calls=[]
        with patch.object(native,'_clock',lambda _:lambda:calls.append(None)):
            native.validate_native(lp,deadline=time.monotonic()+30)
        for limit in range(1,len(calls)+1):
            n=0
            def tick():
                nonlocal n
                n+=1
                if n==limit: raise TimeoutError('injected native validation cutoff')
            with patch.object(native,'_clock',lambda _:tick):
                with self.assertRaises(TimeoutError): native.validate_native(lp,deadline=time.monotonic()+30)

    def test_native_bad_tail_stops_before_backend(self):
        lp,_=fixture(); lp['A']['data'][-1]='bad'; modules,calls=self.fake_backend(lp)
        with patch.dict(sys.modules,modules):
            with self.assertRaises(ValueError): native.propose(lp,deadline=time.monotonic()+30)
        self.assertEqual(calls,[])

    def test_no_solver_or_model_loaded_by_checks(self):
        lp,cert=fixture(); before=set(sys.modules)
        self.verify(lp,cert); native.validate_native(lp,deadline=time.monotonic()+30)
        self.assertFalse({'torch','scipy','numpy','act','highspy'}&(set(sys.modules)-before))


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RowwiseControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    root=os.environ.get('H2_ROWWISE_ROOT')
    if root and Path(root).exists():
        save(Path(root)/'test_outcome.json',{'tests':result.testsRun,'success':result.wasSuccessful(),
            'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],
            'skipped':[(str(t),s) for t,s in result.skipped],'new_solves':0,'real_requests':0})
    sys.exit(not result.wasSuccessful())
