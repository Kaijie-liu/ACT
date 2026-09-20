"""Analytic-only controls; all fixtures and attempts retained by controls.py."""
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
from single_check_portable.execution import ROOT, ACT, save_new
from lp_sandwich.check import identity, check
from lp_sandwich.tests import csr
from portable_proof.runtime import digest
from sparse_basis.tests import hints, col, zero
from modular_basis.engine import solve_field as reference_field, Budget as OldBudget, statistics as old_stats
from modular_basis.tests import solve as reference_solve
from .engine import Budget, POLICY, statistics, eliminate, Limit, Unresolved, solve_field
from .field import Plan, payload, validate
from .propose import propose

ARTIFACT_ROOT=None
OBSERVATIONS=[]


def solve(system,*,reuse=True,seconds=20,scope='control'):
    b=Budget(time.monotonic()+seconds);st=statistics();answer=None;error=None
    begin=time.monotonic()
    try:answer=eliminate(deepcopy(system),b,st,reuse=reuse,scope=scope);status='CANDIDATE_ONLY'
    except Limit as exc:status,error='LIMIT',str(exc)
    except Unresolved as exc:status,error='UNRESOLVED',str(exc)
    except TimeoutError as exc:status,error='TIMEOUT',str(exc)
    except ValueError as exc:status,error='ERROR',str(exc)
    return {'status':status,'error':error,'solution':None if answer is None else list(map(str,answer)),
            'stats':st,'costs':b.costs(),'operations':b.operations,'arithmetic':b.arithmetic,
            'seconds':time.monotonic()-begin,'reuse':reuse,'network_SAFE':False,'network_UNSAFE':False}


def field(system,p,plan=None,scope='control',capture=True):
    b=Budget(time.monotonic()+20);st=statistics()
    x,plan=solve_field(deepcopy(system),p,b,st,plan=plan,scope=scope,capture=capture)
    return x,plan,st,b


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT
        ARTIFACT_ROOT=Path(tempfile.mkdtemp(prefix='plan_basis_controls_',dir=ROOT/'data/moe/results'))
        cls.root=ARTIFACT_ROOT

    def record(self,name,system,result,reference=None):
        folder=self.root/name;folder.mkdir()
        save_new(folder/'system.json',{'rows':[[[j,str(v)] for j,v in sorted(r.items())] for r,_ in system],
                                      'rhs':[str(v) for _,v in system]})
        save_new(folder/'result.json',result)
        if reference is not None:save_new(folder/'reference.json',reference)
        OBSERVATIONS.append({'case':name,'status':result['status'],'operations':result['operations'],
                             'seconds':result['seconds'],'stats':result['stats'],'costs':result['costs']})

    def test_fixed_random_rational_field_and_full_differential(self):
        rng=random.Random(90201)
        for n in range(2,10):
            for rep in range(3):
                mat=[[F(rng.randint(-3,3),rng.randint(1,5)) for _ in range(n)] for _ in range(n)]
                for i,row in enumerate(mat):row[i]=1+sum(abs(v) for j,v in enumerate(row) if i!=j)
                x=[F(rng.randint(-2,2)*2**35,7) for _ in range(n)]
                system=[({j:v for j,v in enumerate(row) if v},sum(v*w for v,w in zip(row,x))) for row in mat]
                on=solve(system);off=solve(system,reuse=False);old=reference_solve(system)
                self.record(f'random_{n}_{rep}',system,on,{'off':off,'old':old})
                self.assertEqual(on['solution'],list(map(str,x)))
                self.assertEqual(on['solution'],off['solution']);self.assertEqual(on['solution'],old['solution'])
                plan=None
                for p in (1073741789,1073741783,1073741741):
                    y,plan,_,_=field(system,p,plan)
                    ref=reference_field(deepcopy(system),p,OldBudget(time.monotonic()+20),old_stats())
                    self.assertEqual(y,ref)

    def test_initial_modular_cancellation_invalidates_and_rebuilds(self):
        system=[({0:F(11),1:F(1)},F(12)),({0:F(1),1:F(1)},F(2))]
        _,plan,_,_=field(system,7)
        x,new,st,_=field(system,11,plan)
        self.assertEqual(x,[1,1]);self.assertEqual(st['plan_builds'],1)
        self.assertEqual(st['plan_invalidations'][0]['reason'],'initial_modular_cancellation')
        self.assertNotEqual(new.initial,plan.initial)
        self.assertGreaterEqual(st['peak_plan_units'],plan.units+new.units)

    def test_fallback_retains_old_plan_storage_and_shared_operations(self):
        system=[({0:F(11),1:F(1)},F(12)),({0:F(1),1:F(1)},F(2))]
        _,plan,_,_=field(system,7)
        _,_,_,full=field(system,11,plan)
        _,_,_,fresh=field(system,11)
        self.assertGreater(full.operations,fresh.operations)
        with patch.dict(POLICY,plan_entries=plan.units):
            with self.assertRaises(Limit):field(system,11,plan)
        with patch.dict(POLICY,operations=fresh.operations):
            with self.assertRaises(Limit):field(system,11,plan)

    def test_zero_planned_pivot_with_nonsingular_prime_rebuilds(self):
        mat=[[1,1,1],[1,12,2],[1,2,4]]
        system=[({j:F(v) for j,v in enumerate(r)},F(sum(r))) for r in mat]
        _,plan,_,_=field(system,7)
        y,_,st,_=field(system,11,plan)
        self.assertEqual(y,[1,1,1])
        self.assertEqual(st['plan_invalidations'][0]['reason'],'zero_planned_pivot')
        self.assertEqual(st['plan_builds'],1)

    def test_binding_matrix_rhs_scope_and_integrity_fail_closed(self):
        system=[({0:F(3)},F(1))];_,plan,_,_=field(system,7)
        cases=[([({0:F(4)},F(1))],plan,'control'),([({0:F(3)},F(2))],plan,'control'),
               (system,plan,'other_request'),(system,replace(plan,checksum='wrong'),'control')]
        for sys,pl,scope in cases:
            with self.assertRaises(ValueError):field(sys,11,pl,scope)
        bad=replace(plan,steps=((0,0,(0,), (0,)),))
        with self.assertRaises(ValueError):field(system,11,bad)

    def test_rehashed_malicious_schedule_cannot_bypass_field_residual(self):
        system=[({0:F(1),1:F(1)},F(2)),({0:F(1),1:F(2)},F(3))]
        _,plan,_,_=field(system,7)
        steps=list(plan.steps);rid,col,keys,_=steps[0];steps[0]=(rid,col,keys,())
        forged=replace(plan,steps=tuple(steps),checksum=identity(payload(plan.binding,plan.initial,tuple(steps))))
        x,_,st,_=field(system,11,forged)
        self.assertEqual(x,[1,1]);self.assertEqual(len(st['plan_invalidations']),1)

    def test_source_and_cached_plan_not_mutated(self):
        system=[({0:F(3),1:F(1)},F(4)),({1:F(2)},F(2))]
        before=deepcopy(system);_,plan,_,_=field(system,7);old=deepcopy(plan)
        field(system,11,plan)
        self.assertEqual(system,before);self.assertEqual(plan,old)

    def test_mutable_or_illegal_plan_schema_rejected_even_rehashed(self):
        system=[({0:F(3)},F(1))];_,plan,_,_=field(system,7)
        for initial,steps in [(list(plan.initial),plan.steps),(plan.initial,list(plan.steps)),
                              (((False,),),plan.steps),(((1,),),plan.steps),
                              (plan.initial,((0,0,(),()),)),(plan.initial,((0,0,(0,),(0,)),))]:
            bad=replace(plan,initial=initial,steps=steps,checksum=identity(payload(plan.binding,initial,steps)))
            with self.assertRaises(ValueError):field(system,11,bad)

    def test_map_elimination_backsub_and_costs_are_explicit(self):
        system=[({0:F(1)},F(2**40))];out=solve(system)
        self.record('cost_accounting',system,out)
        ops=out['costs']['operations']
        self.assertEqual(sum(ops.values()),out['operations'])
        for k in ('finite_field/map','finite_field/elimination','finite_field/back_substitution',
                  'finite_field/residual_check','reconstruction/euclidean','exact_residual/original_rational_equations'):
            # Empty arithmetic loops are measured in time even if operation count is zero.
            self.assertIn(k,out['costs']['seconds'])
            self.assertIn(k,ops)
        self.assertTrue(all(v>=0 for v in out['costs']['seconds'].values()))

    def test_reconstruction_failure_coordinate_and_success_prefix(self):
        from .engine import reconstruct
        system=[({0:F(1)},F(0)),({1:F(1)},F(1)),({2:F(1)},F(2))]
        counter=[0]
        def fail_second(residue,modulus,b,detail=None):
            counter[0]+=1
            if counter[0]==2:
                detail.update(bound_bits=15,failure_reason='denominator_bound');return None
            return reconstruct(residue,modulus,b,detail)
        with patch('plan_basis.engine.reconstruct',side_effect=fail_second):out=solve(system)
        self.record('prefix_control',system,out)
        r=out['stats']['rounds'][0]['reconstruction_progress']
        self.assertEqual(r['successful_prefix'],1);self.assertEqual(r['failed_coordinate'],1)
        self.assertEqual(r['failure_reason'],'denominator_bound');self.assertFalse(r['complete_vector'])
        self.assertEqual(out['solution'],['0','1','2'])

    def test_deadline_in_reconstruction_preserves_partial_diagnostics(self):
        system=[({0:F(1)},F(1))]
        def stop(*args,**kwargs):raise TimeoutError('controlled deadline')
        with patch('plan_basis.engine.reconstruct',side_effect=stop):out=solve(system)
        self.record('partial_reconstruction',system,out)
        self.assertEqual(out['status'],'TIMEOUT');self.assertIsNone(out['solution'])
        p=out['stats']['rounds'][0]['reconstruction_progress']
        self.assertEqual(p['failed_coordinate'],0);self.assertEqual(p['status'],'INTERRUPTED')

    def test_early_alias_still_requires_exact_rational_residual(self):
        system=[({0:F(1)},F(1073741790))];out=solve(system)
        self.record('early_alias',system,out)
        self.assertEqual(out['solution'],['1073741790'])
        self.assertGreater(out['stats']['residual_rejections'],0)
        self.assertGreater(out['stats']['plan_replays'],0)

    def test_corrupted_field_vector_cannot_bypass_exact_residual(self):
        system=[({0:F(1)},F(1))]
        with patch.dict(POLICY,primes=2),patch('plan_basis.engine.solve_field',return_value=([0],None)):
            out=solve(system)
        self.record('corrupt_candidate',system,out)
        self.assertEqual(out['status'],'UNRESOLVED');self.assertEqual(out['stats']['residual_rejections'],2)

    def test_caps_and_timeout_never_publish_candidate(self):
        system=[({0:F(3),1:F(1)},F(4)),({1:F(2)},F(2))]
        self.assertEqual(solve(system,seconds=-1)['status'],'TIMEOUT')
        for key,val in [('operations',1),('plan_entries',1),('live_nnz',1),('max_bits',1)]:
            with patch.dict(POLICY,{key:val}):out=solve(system)
            self.assertEqual(out['status'],'LIMIT',key);self.assertIsNone(out['solution'])
            self.assertEqual(sum(out['costs']['operations'].values()),out['operations'])

    def test_replay_deadline_and_plan_memory_are_charged(self):
        system=[({0:F(3)},F(1))];_,plan,_,_=field(system,7)
        b=Budget(time.monotonic()-1)
        with self.assertRaises(TimeoutError):solve_field(system,11,b,statistics(),plan=plan)
        with patch.dict(POLICY,plan_entries=1):
            with self.assertRaises(Limit):field(system,11,plan)

    def test_corrupt_replay_and_inside_replay_deadline(self):
        from .field import replay
        system=[({0:F(3)},F(1))];_,plan,_,_=field(system,7)
        with patch('plan_basis.field.replay',return_value=[0]):
            x,_,st,_=field(system,11,plan)
        self.assertEqual(x,[4]);self.assertEqual(st['plan_invalidations'][0]['reason'],'field_residual_mismatch')
        def expire(system,p,b,st,pl):
            b.deadline=time.monotonic()-1
            return replay(system,p,b,st,pl)
        b=Budget(time.monotonic()+20);st=statistics()
        with patch('plan_basis.field.replay',side_effect=expire):
            with self.assertRaises(TimeoutError):solve_field(system,11,b,st,plan=plan,scope='control')
        self.assertEqual(st['plan_builds'],0);self.assertFalse(st['plan_invalidations'])

    def test_multiround_reuse_original_lp_moved_check(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[2**41],
            'E':csr([[7]],1),'h':[2**40],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('x',0)],[zero('E_residual',0)])
        out=propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+20,reuse=True)
        self.assertEqual(out['status'],'CANDIDATE_ONLY');self.assertGreater(out['stats']['plan_replays'],0)
        folder=self.root/'multiround_lp';folder.mkdir()
        save_new(folder/'proposal.json',out);save_new(folder/'bundle.json',out['bundle'])
        save_new(folder/'check_reference.json',check(out['bundle'],identity(s)))
        shutil.copyfile(ROOT/'lp_sandwich/check.py',folder/'verify.py')
        run=subprocess.run([ACT,'-I','-S',str(folder/'verify.py'),str(folder/'bundle.json'),
             '--bundle-sha256',digest((folder/'bundle.json').read_bytes()),'--statement-sha256',identity(s),
             '--timeout-seconds','10'],cwd=folder,capture_output=True,text=True,check=True,timeout=15)
        checked=json.loads(run.stdout);save_new(folder/'checked.json',checked)
        self.assertEqual(checked['upper_bound'],str(-F(2**40,7)))
        self.assertFalse(checked['solver_or_model_imported'])

    def test_bad_denominator_and_singular_modulus_not_network_verdict(self):
        p=1073741789
        for name,sys in [('denominator',[({0:F(1,p)},F(1,p))]),('singular',[({0:F(p)},F(p))])]:
            out=solve(sys);self.record(name,sys,out)
            self.assertEqual(out['solution'],['1']);self.assertFalse(out['network_UNSAFE'])

    def test_sparse_multiround_scale_differential(self):
        n=1024;x=F(2**40,7)
        system=[({i:F(3),**({i-1:F(1)} if i else {})},(4 if i else 3)*x) for i in range(n)]
        on=solve(system);off=solve(system,reuse=False)
        self.record('sparse_multiround_1024',system,on,{'off':off})
        self.assertEqual(on['solution'],[str(x)]*n);self.assertEqual(on['solution'],off['solution'])
        self.assertGreater(on['stats']['plan_replays'],0)

    def test_original_lp_checks_and_relocated_solver_free_replay(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3],[6]],1),'h':[1,2],'A':csr([[1]],1),'b':[1]}
        s,c,h=hints(lp,[col('x',0),col('E_residual',1),col('A_slack',0)],[zero('E_residual',0)])
        for reuse in (True,False):
            out=propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+20,reuse=reuse)
            self.assertEqual(out['status'],'CANDIDATE_ONLY');self.assertFalse(out['feasibility_certified'])
            self.assertEqual(check(out['bundle'],identity(s))['upper_bound'],'-1/3')
        folder=self.root/'relocated';folder.mkdir();save_new(folder/'bundle.json',out['bundle'])
        shutil.copyfile(ROOT/'lp_sandwich/check.py',folder/'verify.py')
        cmd=[ACT,'-I','-S',str(folder/'verify.py'),str(folder/'bundle.json'),
             '--bundle-sha256',digest((folder/'bundle.json').read_bytes()),'--statement-sha256',identity(s),'--timeout-seconds','10']
        run=subprocess.run(cmd,cwd=folder,capture_output=True,text=True,check=True,timeout=15)
        checked=json.loads(run.stdout);save_new(folder/'checked.json',checked)
        self.assertFalse(checked['solver_or_model_imported']);self.assertEqual(checked['upper_bound'],'-1/3')
        bad=deepcopy(out['bundle']);bad['primal']['x']=['0'];bad['primal']['claimed_objective']='0'
        self.assertEqual(check(bad,identity(s))['primal_status'],'NOT_EXACTLY_FEASIBLE')

    def test_original_lp_inexact_residual_and_binding_remain_rejected(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3]],1),'h':[1],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('E_residual',0)],[{'column':col('x',0),'at':'lower'}])
        out=propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+20)
        self.assertEqual(out['status'],'CANDIDATE_ONLY')
        self.assertEqual(check(out['bundle'],identity(s))['primal_status'],'NOT_EXACTLY_FEASIBLE')
        folder=self.root/'rejected_lp';folder.mkdir();save_new(folder/'bundle.json',out['bundle'])
        save_new(folder/'checked.json',check(out['bundle'],identity(s)))
        bad=deepcopy(h);bad['rows']=[]
        out=propose(lp,s,c,bad,identity(s),identity(bad),deadline=time.monotonic()+20)
        self.assertEqual(out['status'],'ERROR');self.assertIsNone(out['bundle'])

    def test_empty_system_and_unresolved_large_answer(self):
        self.assertEqual(solve([])['solution'],[])
        system=[({0:F(2**2100)},F(1))]
        with patch.dict(POLICY,primes=2):out=solve(system)
        self.record('unresolved_large_denominator',system,out)
        self.assertEqual(out['status'],'UNRESOLVED');self.assertIsNone(out['solution'])


if __name__=='__main__':unittest.main()
