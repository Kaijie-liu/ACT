"""Fixed analytic amortization controls. No real LP/native calls."""
from copy import deepcopy
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
import json
import random
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from single_check_portable.execution import ROOT,ACT,save_new
from portable_proof.runtime import digest
from lp_sandwich.check import identity,check
from lp_sandwich.tests import csr
from sparse_basis.tests import hints,col,zero
from plan_basis.tests import solve as old_solve
from .engine import Budget,POLICY,statistics,eliminate,Limit,Unresolved
from .session import Session
from .immutable import Handle
from . import field
from .propose import propose

MODES=((False,False),(True,False),(False,True),(True,True))
ARTIFACT_ROOT=None
OBSERVATIONS=[]


def solve(system,mode=(True,True),seconds=20,scope='control'):
    b=Budget(time.monotonic()+seconds);st=statistics();answer=None;error=None;start=time.monotonic()
    try:
        answer=eliminate(deepcopy(system),b,st,amortize_source=mode[0],amortize_plan=mode[1],scope=scope)
        status='CANDIDATE_ONLY'
    except Limit as exc:status,error='LIMIT',str(exc)
    except Unresolved as exc:status,error='UNRESOLVED',str(exc)
    except TimeoutError as exc:status,error='TIMEOUT',str(exc)
    except ValueError as exc:status,error='ERROR',str(exc)
    return {'status':status,'error':error,'solution':None if answer is None else list(map(str,answer)),
            'stats':st,'costs':b.costs(),'operations':b.operations,'arithmetic':b.arithmetic,
            'seconds':time.monotonic()-start,'mode':list(mode),'network_SAFE':False,'network_UNSAFE':False}


def session(system,mode=(True,True),scope='control'):
    b=Budget(time.monotonic()+20);st=statistics()
    s=Session(system,b,st,scope=scope,amortize_source=mode[0],amortize_plan=mode[1])
    return s,b,st


def use(s,p,handle=None):return s.solve(p,scope='control',source_id=s.source_id,handle=handle)


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT
        ARTIFACT_ROOT=Path(tempfile.mkdtemp(prefix='amortized_basis_controls_',dir=ROOT/'data/moe/results'))
        cls.root=ARTIFACT_ROOT

    def record(self,name,system,result,reference=None):
        folder=self.root/name;folder.mkdir()
        save_new(folder/'system.json',{'rows':[[[j,str(v)] for j,v in sorted(r.items())] for r,_ in system],
                                      'rhs':[str(v) for _,v in system]})
        save_new(folder/'result.json',result)
        if reference is not None:save_new(folder/'reference.json',reference)
        OBSERVATIONS.append({'case':name,'status':result['status'],'mode':result['mode'],
                             'operations':result['operations'],'costs':result['costs'],
                             'seconds':result['seconds'],'stats':result['stats']})

    def test_four_modes_fixed_random_and_frozen_differential(self):
        rng=random.Random(90201)
        for n in (2,4,8):
            for repeat in range(2):
                mat=[[F(rng.randint(-3,3),rng.randint(1,5)) for _ in range(n)] for _ in range(n)]
                for i,row in enumerate(mat):row[i]=1+sum(abs(v) for j,v in enumerate(row) if i!=j)
                x=[F(rng.randint(-2,2)*2**35,7) for _ in range(n)]
                system=[({j:v for j,v in enumerate(row) if v},sum(v*w for v,w in zip(row,x))) for row in mat]
                ref=old_solve(system);numerical=None
                for mode in MODES:
                    out=solve(system,mode);self.record(f'random_{n}_{repeat}_{int(mode[0])}{int(mode[1])}',system,out,{'frozen':ref})
                    self.assertEqual(out['solution'],list(map(str,x)))
                    self.assertEqual(out['solution'],ref['solution'])
                    current={k:v for k,v in out['costs']['operations'].items() if k.startswith(('finite_field/','CRT/','reconstruction/','exact_residual/'))}
                    if numerical is None:numerical=current
                    self.assertEqual(current,numerical)

    def test_owned_source_is_deeply_immutable_and_caller_detached(self):
        v=F(1,3);system=[({0:v},F(1))];s,_,_=session(system);before=s.source_id
        system[0][0][0]=F(7);v._numerator=99
        self.assertEqual(use(s,11),[3]);self.assertEqual(s.source_id,before)
        row,rhs=s.system[0];ratio=row.entries[0][1]
        for obj,key,val in [(ratio,'numerator',0),(row,'entries',()),(rhs,'denominator',0)]:
            with self.assertRaises(AttributeError):object.__setattr__(obj,key,val)
        with self.assertRaises(TypeError):s.system[0][0].entries[0]=(0,F(0))
        changed,_,_=session(system)
        self.assertNotEqual(changed.source_id,s.source_id)
        with self.assertRaises(ValueError):s.solve(13,scope='control',source_id=changed.source_id)

    def test_exported_wrapper_mutation_does_not_pollute_owned_plan(self):
        s,_,st=session([({0:F(3)},F(1))]);use(s,7);raw=s.export_plan()
        object.__setattr__(raw,'checksum','wrong')
        self.assertEqual(use(s,11),[4]);self.assertEqual(st['plan_replays'],1)
        with self.assertRaises(ValueError):s.admit_plan(raw,scope='control',source_id=s.source_id)

    def test_scope_rhs_matrix_identity_and_foreign_handles(self):
        a,_,_=session([({0:F(3)},F(1))]);use(a,7);h=a.handle
        b,_,_=session([({0:F(3)},F(1))]);use(b,7)
        with self.assertRaises(ValueError):use(b,11,h)
        for system in ([({0:F(4)},F(1))],[({0:F(3)},F(2))]):
            c,_,_=session(system)
            with self.assertRaises(ValueError):c.admit_plan(a.export_plan(),scope='control',source_id=c.source_id)
        c,_,_=session([({0:F(3)},F(1))],scope='other')
        with self.assertRaises(ValueError):c.admit_plan(a.export_plan(),scope='other',source_id=c.source_id)
        with self.assertRaises(ValueError):a.solve(11,scope='wrong',source_id=a.source_id)
        with self.assertRaises(ValueError):a.solve(11,scope='control',source_id='wrong')

    def test_stale_forged_closed_and_reimported_handles(self):
        s,b,st=session([({0:F(3)},F(1))]);use(s,7);h=s.handle;raw=s.export_plan()
        before=b.operations;checks=st['plan_full_checks']
        new=s.admit_plan(raw,scope='control',source_id=s.source_id)
        self.assertGreater(b.operations,before);self.assertEqual(st['plan_full_checks'],checks+1)
        with self.assertRaises(ValueError):use(s,11,h)
        with self.assertRaises(ValueError):use(s,11,Handle(new.owner,new.generation))
        self.assertEqual(use(s,11,new),[4])
        s.close()
        with self.assertRaises(ValueError):use(s,13,new)

    def test_import_requires_full_checks_not_a_trusted_flag(self):
        s,_,st=session([({0:F(3)},F(1))]);use(s,7);raw=s.export_plan();before=st['plan_full_checks']
        for bad in ({'validated':True,'plan':raw},replace(raw,binding='wrong'),replace(raw,checksum='wrong'),
                    replace(raw,initial=list(raw.initial))):
            with self.assertRaises(ValueError):s.admit_plan(bad,scope='control',source_id=s.source_id)
        self.assertEqual(st['plan_full_checks'],before)
        self.assertEqual(use(s,11),[4])

    def test_cancellation_and_zero_pivot_replacement_validate_again(self):
        cases=[([({0:F(11),1:F(1)},F(12)),({0:F(1),1:F(1)},F(2))],'initial_modular_cancellation'),
               ([({j:F(v) for j,v in enumerate(r)},F(sum(r))) for r in [[1,1,1],[1,12,2],[1,2,4]]],'zero_planned_pivot')]
        for system,reason in cases:
            for mode in MODES:
                s,_,st=session(system,mode);use(s,7);old=s.handle;old_units=s.export_plan().units
                self.assertEqual(use(s,11),[1]*len(system))
                self.assertEqual(st['plan_invalidations'][0]['reason'],reason)
                self.assertGreaterEqual(st['plan_full_checks'],2)
                self.assertGreaterEqual(st['peak_plan_units'],old_units+s.export_plan().units)
                with self.assertRaises(ValueError):use(s,13,old)

    def test_failed_prime_recovery_does_not_accumulate_dead_plan(self):
        # Determinant 11: singular at 11, usable at 7 and 13.
        system=[({0:F(1),1:F(1)},F(2)),({0:F(1),1:F(12)},F(13))]
        from .engine import Singular
        s,b,st=session(system);use(s,7);old=s.handle;units=s.export_plan().units
        with self.assertRaises(Singular):use(s,11)
        self.assertEqual(s.handle,old)
        self.assertEqual(use(s,13),[1,1]);self.assertEqual(b.plan_units,units)

    def test_full_validation_call_counts_and_numerical_work_equal(self):
        system=[({0:F(1)},F(2**40))]
        outputs=[solve(system,mode) for mode in MODES]
        for mode,out in zip(MODES,outputs):
            st=out['stats'];rounds=st['primes_tried']
            self.assertEqual(st['source_full_checks'],1 if mode[0] else rounds)
            self.assertEqual(st['plan_full_checks'],st['plan_builds'] if mode[1] else st['plan_builds']+st['plan_replays'])
            self.assertEqual(st['request_guards'],rounds+st['plan_builds'])
            self.assertEqual(st['plan_guard_hits'],st['plan_replays'])
            self.assertEqual(sum(out['costs']['operations'].values()),out['operations'])
            self.assertEqual(out['solution'],[str(2**40)])

    def test_fixed_1024_four_modes_cost_decomposition(self):
        x=F(2**40,7);n=1024
        system=[({i:F(3),**({i-1:F(1)} if i else {})},(4 if i else 3)*x) for i in range(n)]
        reference=old_solve(system,reuse=False);outputs=[]
        for mode in MODES:
            out=solve(system,mode);self.record(f'sparse1024_{int(mode[0])}{int(mode[1])}',system,out,{'old_no_reuse':reference})
            self.assertEqual(out['solution'],[str(x)]*n);outputs.append(out)
        base=outputs[0]['costs']['operations']
        for out in outputs[1:]:
            counts=out['costs']['operations']
            delta=sum(base.get(k,0)-counts.get(k,0) for k in set(base)|set(counts))
            validation_delta=sum(base.get(k,0)-counts.get(k,0) for k in ('plan/binding','plan/validate'))
            self.assertEqual(delta,validation_delta)
        # No speed or complete-request superiority assertion.

    def test_live_cap_includes_plan_on_dynamic_path(self):
        from plan_basis import field as oldfield
        from plan_basis.engine import Budget as OldBudget,statistics as oldstats,POLICY as oldpolicy
        system=[({0:F(1)},F(1))]
        b=Budget(time.monotonic()+10);st=statistics()
        with patch.dict(POLICY,live_nnz=3):
            with self.assertRaises(Limit):field.dynamic(system,7,b,st,capture=True)
        ob=OldBudget(time.monotonic()+10);os=oldstats()
        with patch.dict(oldpolicy,live_nnz=3):oldfield.dynamic(system,7,ob,os,capture=True)
        self.assertGreater(os['peak_live_nnz'],3)
        save_new(self.root/'legacy_live_cap_control.json',{'old_peak':os['peak_live_nnz'],'cap':3,
                 'old_returned':True,'new_limit':b.arithmetic['first_limit'],'scope':'analytic accounting gap, not a false LP certificate'})

    def test_source_copy_and_old_new_plan_storage_caps(self):
        with patch.dict(POLICY,live_nnz=2):
            with self.assertRaises(Limit):session([({0:F(1)},F(1))])
        system=[({0:F(11),1:F(1)},F(12)),({0:F(1),1:F(1)},F(2))]
        s,_,_=session(system);use(s,7);units=s.export_plan().units
        with patch.dict(POLICY,plan_entries=units):
            with self.assertRaises(Limit):use(s,11)

    def test_deadlines_admission_use_validation_and_partial_reconstruction(self):
        system=[({0:F(1)},F(2**40))]
        self.assertEqual(solve(system,seconds=-1)['status'],'TIMEOUT')
        s,b,st=session(system);use(s,7);b.deadline=time.monotonic()-1
        with self.assertRaises(TimeoutError):use(s,11)
        from .engine import reconstruct
        calls=[0]
        def interrupted(value,modulus,b,detail=None):
            calls[0]+=1
            if calls[0]==2:raise TimeoutError('controlled reconstruction deadline')
            return reconstruct(value,modulus,b,detail)
        sys=[({0:F(1)},F(0)),({1:F(1)},F(1))]
        with patch('amortized_basis.engine.reconstruct',side_effect=interrupted):out=solve(sys)
        self.record('partial_reconstruction',sys,out)
        self.assertEqual(out['status'],'TIMEOUT');self.assertIsNone(out['solution'])
        progress=out['stats']['rounds'][0]['reconstruction_progress']
        self.assertEqual(progress['successful_prefix'],1);self.assertEqual(progress['failed_coordinate'],1)
        self.assertEqual(progress['status'],'INTERRUPTED')

    def test_source_plan_operation_bit_limits_do_not_reset(self):
        system=[({0:F(3),1:F(1)},F(4)),({1:F(2)},F(2))]
        for key,val in [('operations',1),('plan_entries',1),('max_bits',1)]:
            with patch.dict(POLICY,{key:val}):out=solve(system)
            self.assertEqual(out['status'],'LIMIT');self.assertIsNone(out['solution'])
            self.assertEqual(sum(out['costs']['operations'].values()),out['operations'])

    def test_late_admission_deadline_rolls_back_receipt_not_costs(self):
        s,b,st=session([({0:F(3)},F(1))]);use(s,7)
        old=s.handle;raw=s.export_plan();start=b.operations;tick=b.tick
        def late():
            if s.handle is not old:raise TimeoutError('controlled publication deadline')
            tick()
        with patch.object(b,'tick',side_effect=late):
            with self.assertRaises(TimeoutError):s.admit_plan(raw,scope='control',source_id=s.source_id)
        self.assertIs(s.handle,old);self.assertGreater(b.operations,start)
        self.assertEqual(use(s,11,old),[4])

    def test_corrupt_replay_and_exact_alias_checks_still_run(self):
        s,_,st=session([({0:F(3)},F(1))]);use(s,7)
        with patch('amortized_basis.field.replay',return_value=[0]):self.assertEqual(use(s,11),[4])
        self.assertEqual(st['plan_invalidations'][0]['reason'],'field_residual_mismatch')
        system=[({0:F(1)},F(1073741790))]
        for mode in MODES:
            out=solve(system,mode);self.record(f'alias_{int(mode[0])}{int(mode[1])}',system,out)
            self.assertEqual(out['solution'],['1073741790']);self.assertGreater(out['stats']['residual_rejections'],0)

    def test_nonpositive_and_interrupted_evidence_not_promoted(self):
        system=[({0:F(2**2100)},F(1))]
        with patch.dict(POLICY,primes=2):out=solve(system)
        self.record('unresolved_denominator',system,out)
        self.assertEqual(out['status'],'UNRESOLVED');self.assertIsNone(out['solution'])
        self.assertFalse(out['network_SAFE']);self.assertFalse(out['network_UNSAFE'])

    def test_multiround_original_lp_and_portable_independent_check(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[2**41],
            'E':csr([[7]],1),'h':[2**40],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('x',0)],[zero('E_residual',0)])
        for mode in MODES:
            out=propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+20,
                        amortize_source=mode[0],amortize_plan=mode[1])
            self.assertEqual(out['status'],'CANDIDATE_ONLY');self.assertGreater(out['stats']['plan_replays'],0)
            self.assertEqual(check(out['bundle'],identity(s))['upper_bound'],str(-F(2**40,7)))
            folder=self.root/f'lp_{int(mode[0])}{int(mode[1])}';folder.mkdir()
            save_new(folder/'proposal.json',out);save_new(folder/'bundle.json',out['bundle'])
            save_new(folder/'check_reference.json',check(out['bundle'],identity(s)))
        folder=self.root/'lp_11'
        shutil.copyfile(ROOT/'lp_sandwich/check.py',folder/'verify.py')
        run=subprocess.run([ACT,'-I','-S',str(folder/'verify.py'),str(folder/'bundle.json'),
             '--bundle-sha256',digest((folder/'bundle.json').read_bytes()),'--statement-sha256',identity(s),
             '--timeout-seconds','10'],cwd=folder,capture_output=True,text=True,check=True,timeout=15)
        checked=json.loads(run.stdout);save_new(folder/'checked.json',checked)
        self.assertFalse(checked['solver_or_model_imported']);self.assertEqual(checked['upper_bound'],str(-F(2**40,7)))

    def test_original_lp_bad_residual_and_source_binding_rejected(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3]],1),'h':[1],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('E_residual',0)],[{'column':col('x',0),'at':'lower'}])
        out=propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+20)
        self.assertEqual(out['status'],'CANDIDATE_ONLY');self.assertFalse(out['feasibility_certified'])
        checked=check(out['bundle'],identity(s));self.assertEqual(checked['primal_status'],'NOT_EXACTLY_FEASIBLE')
        folder=self.root/'rejected_lp';folder.mkdir();save_new(folder/'bundle.json',out['bundle']);save_new(folder/'check_reference.json',checked)
        bad=deepcopy(h);bad['rows']=[]
        result=propose(lp,s,c,bad,identity(s),identity(bad),deadline=time.monotonic()+20)
        self.assertEqual(result['status'],'ERROR');self.assertIsNone(result['bundle'])

    def test_empty_source_and_no_accepted_flags(self):
        self.assertEqual(solve([])['solution'],[])
        with self.assertRaises(ValueError):session([({0:F(1)},F(1))],mode=(1,True))


if __name__=='__main__':unittest.main()
