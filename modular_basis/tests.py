"""Analytic mechanism, adverse and synthetic scale controls, never real LPs."""
from copy import deepcopy
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
from portable_proof.runtime import digest
from lp_sandwich.check import check, identity
from lp_sandwich.tests import csr
from sparse_basis.tests import hints, col, zero, stats as old_stats
from sparse_basis.engine import eliminate as old_eliminate, Budget as OldBudget, propose as old_propose
from modular_basis.engine import POLICY, Budget, Limit, Unresolved, statistics, eliminate, solve_field
from modular_basis.propose import propose

ARTIFACT_ROOT=None
OBSERVATIONS=[]


def generate(lp,s,c,h,seconds=30):
    return propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+seconds)


def solve(system,seconds=30):
    b=Budget(time.monotonic()+seconds);st=statistics();start=time.monotonic();answer=None;error=None
    try:answer=eliminate(deepcopy(system),b,st);status='CANDIDATE_ONLY'
    except Limit as e:status,error='LIMIT',str(e)
    except Unresolved as e:status,error='UNRESOLVED_MODULAR_RECONSTRUCTION',str(e)
    except TimeoutError as e:status,error='TIMEOUT',str(e)
    return {'status':status,'error':error,'solution':None if answer is None else list(map(str,answer)),
            'stats':st,'arithmetic':b.arithmetic,'operations':b.operations,'seconds':time.monotonic()-start,
            'policy':dict(POLICY),'network_SAFE':False,'network_UNSAFE':False}


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT
        ARTIFACT_ROOT=Path(tempfile.mkdtemp(prefix='modular_basis_controls_',dir=ROOT/'data/moe/results'))
        cls.root=ARTIFACT_ROOT

    def record(self,name,system,out,extra=None):
        path=self.root/name;path.mkdir()
        save_new(path/'system.json',{'rows':[[[j,str(v)] for j,v in sorted(row.items())] for row,_ in system],
                                    'rhs':[str(rhs) for _,rhs in system]})
        save_new(path/'result.json',out)
        if extra is not None:save_new(path/'reference.json',extra)
        OBSERVATIONS.append({'case':name,**{k:out[k] for k in ('status','arithmetic','stats','seconds')},
                             'reference':extra})

    def record_lp(self,name,lp,s,c,h,out):
        path=self.root/name;path.mkdir()
        save_new(path/'input.json',{'lp':lp,'statement':s,'candidate':c,'hint':h})
        save_new(path/'proposal.json',out)
        if out['bundle'] is not None:
            save_new(path/'bundle.json',out['bundle'])
            save_new(path/'check_reference.json',check(out['bundle'],identity(s)))

    def test_random_rational_differential_and_original_residuals(self):
        rng=random.Random(6221)
        for n in range(2,10):
            for repeat in range(3):
                mat=[[F(rng.randint(-4,4),rng.randint(1,5)) for _ in range(n)] for _ in range(n)]
                for i in range(n):mat[i][i]=1+sum(abs(v) for j,v in enumerate(mat[i]) if j!=i)
                x=[F(rng.randint(-3,3),7) for _ in range(n)]
                system=[({j:v for j,v in enumerate(row) if v},sum(a*b for a,b in zip(row,x))) for row in mat]
                ref=old_eliminate(deepcopy(system),OldBudget(time.monotonic()+10),old_stats())
                out=solve(system,10);self.record(f'random_{n}_{repeat}',system,out,{'old_solution':list(map(str,ref))})
                self.assertEqual(out['solution'],list(map(str,x)))
                self.assertEqual(ref,x)

    def test_deadline_operations_fill_and_storage_caps(self):
        system=[({0:F(1),1:F(1)},F(2)),({0:F(1),2:F(1)},F(2)),({1:F(1),2:F(1)},F(2))]
        self.assertEqual(solve(system,-1)['status'],'TIMEOUT')
        for k,v in [('operations',1),('fill_insertions',0),('live_nnz',2),('heap_entries',1),('equations',1),('input_nnz',1)]:
            with patch.dict(POLICY,{k:v}):out=solve(system)
            self.assertEqual(out['status'],'LIMIT',k);self.assertIsNotNone(out['arithmetic']['first_limit'])
        self.assertEqual(solve(system)['solution'],['1','1','1'])
        with self.assertRaises(ValueError):Budget(time.monotonic()+301)

    def test_synthetic_4096_sparse_scale(self):
        n=4096
        system=[({i:F(3),**({i-1:F(1)} if i else {})},F(4,3) if i else F(1)) for i in range(n)]
        out=solve(system);self.record('synthetic_4096',system,out,{'scope':'structured synthetic only'})
        self.assertEqual(out['solution'],['1/3']*n)
        self.assertEqual(out['stats']['pivots'],n);self.assertEqual(out['stats']['fill_insertions'],0)

    def test_inexact_basis_remains_rejected_by_original_lp(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3]],1),'h':[1],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('E_residual',0)],[{'column':col('x',0),'at':'lower'}])
        out=generate(lp,s,c,h);self.record_lp('inexact_residual',lp,s,c,h,out)
        self.assertEqual(out['status'],'CANDIDATE_ONLY')
        d=check(out['bundle'],identity(s));self.assertEqual(d['primal_status'],'NOT_EXACTLY_FEASIBLE')
        self.assertIsNone(d['upper_bound'])

    def test_redundant_rows_retained_and_fractional_solution(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3],[6]],1),'h':[1,2],'A':csr([[1]],1),'b':[1]}
        s,c,h=hints(lp,[col('x',0),col('E_residual',1),col('A_slack',0)],[zero('E_residual',0)])
        out=generate(lp,s,c,h);self.record_lp('redundant_rows',lp,s,c,h,out)
        self.assertEqual(out['bundle']['primal']['x'],['1/3'])
        self.assertEqual(out['row_residuals']['E_fixed_zero_required'],['0','0'])
        self.assertEqual(check(out['bundle'],identity(s))['primal_status'],'EXACT_FEASIBLE')

    def test_binding_and_missing_obligation_rejected(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3]],1),'h':[1],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('x',0)],[zero('E_residual',0)])
        for field in ('rows','anchors','lp_sha256','coordinates'):
            bad=deepcopy(h);bad[field]=[] if field in ('rows','anchors') else 'wrong'
            out=generate(lp,s,c,bad);self.assertEqual(out['status'],'ERROR',field)
            self.assertIsNone(out['bundle'])
        bad=deepcopy(c);bad['x']=[];self.assertEqual(generate(lp,s,bad,h)['status'],'ERROR')

    def test_fixed_box_empty_basis_and_no_mutation(self):
        lp={'matrix_format':'csr_v1','c':[0.1],'offset':0,'lower':[0.2],'upper':[0.2],
            'E':csr([],1),'h':[],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[],[{'column':col('x',0),'at':'upper'}]);before=deepcopy((lp,s,c,h))
        out=generate(lp,s,c,h);self.record_lp('fixed_box',lp,s,c,h,out)
        self.assertEqual((lp,s,c,h),before)
        self.assertEqual(F(out['bundle']['primal']['claimed_objective']),F(.1)*F(.2))
        self.assertEqual(check(out['bundle'],identity(s))['primal_status'],'EXACT_FEASIBLE')

    def test_checker_relocation_and_tampered_point(self):
        lp={'matrix_format':'csr_v1','c':[-1],'offset':0,'lower':[0],'upper':[1],
            'E':csr([[3]],1),'h':[1],'A':csr([],1),'b':[]}
        s,c,h=hints(lp,[col('x',0)],[zero('E_residual',0)]);out=generate(lp,s,c,h)
        self.record_lp('relocated',lp,s,c,h,out)
        moved=self.root/'moved';moved.mkdir();save_new(moved/'bundle.json',out['bundle'])
        shutil.copyfile(ROOT/'lp_sandwich/check.py',moved/'verify.py')
        cmd=[ACT,'-I','-S',str(moved/'verify.py'),str(moved/'bundle.json'),
             '--bundle-sha256',digest((moved/'bundle.json').read_bytes()),'--statement-sha256',identity(s),
             '--timeout-seconds','10']
        result=subprocess.run(cmd,cwd=moved,capture_output=True,text=True,check=True,timeout=15)
        checked=json.loads(result.stdout);save_new(moved/'check.json',checked)
        self.assertEqual(checked['upper_bound'],'-1/3');self.assertFalse(checked['solver_or_model_imported'])
        bad=deepcopy(out['bundle']);bad['primal']['x']=['0'];bad['primal']['claimed_objective']='0'
        self.assertEqual(check(bad,identity(s))['primal_status'],'NOT_EXACTLY_FEASIBLE')

    def test_review_checks_execution_flags_separately_from_bound_identity(self):
        from modular_basis.review import compare_checked
        ref={'status':'CHECKED_LP_DIAGNOSTIC','upper_bound':'-1'}
        v={**ref,'seconds':.1,'isolated':True,'site_disabled':True,'solver_or_model_imported':False}
        compare_checked(v,ref)
        for field,value in [('isolated',False),('site_disabled',False),('solver_or_model_imported',True),
                            ('seconds',31),('upper_bound','-2')]:
            bad={**v,field:value}
            with self.assertRaises(ValueError):compare_checked(bad,ref)
        bad=deepcopy(v);bad.pop('site_disabled')
        with self.assertRaises(ValueError):compare_checked(bad,ref)


    def test_avoidable_4201_bit_cross_product_and_checked_original_lp(self):
        from primitive_basis.tests import solve as primitive_solve
        p=2**2100+1
        system=[({0:F(p),1:F(1)},F(p+1)),({0:F(1),1:F(p)},F(p+1))]
        old=primitive_solve(system);out=solve(system)
        self.record('large_product_small_answer',system,out,{'primitive':old,'analytic_solution':['1','1']})
        self.assertEqual(old['status'],'LIMIT')
        self.assertEqual(old['arithmetic']['first_limit']['operation'],'row_product')
        self.assertEqual(out['solution'],['1','1'])
        self.assertLessEqual(out['stats']['max_field_product_bits'],60)
        self.assertEqual(out['stats']['primes_used'],1)
        lp={'matrix_format':'csr_v1','c':[-1,0],'offset':0,'lower':[-2,-2],'upper':[2,2],
            'E':csr([[p,1],[1,p]],2),'h':[p+1,p+1],'A':csr([],2),'b':[]}
        s,c,h=hints(lp,[col('x',0),col('x',1)],[zero('E_residual',0),zero('E_residual',1)])
        proposal=generate(lp,s,c,h);self.record_lp('checked_large_product',lp,s,c,h,proposal)
        self.assertEqual(proposal['status'],'CANDIDATE_ONLY')
        self.assertFalse(proposal['feasibility_certified'])
        checked=check(proposal['bundle'],identity(s))
        self.assertEqual(checked['primal_status'],'EXACT_FEASIBLE')
        self.assertEqual(checked['upper_bound'],'-1')
        self.assertIsNone(checked['lower_bound'])

    def test_avoids_global_denominator_lcm(self):
        from primitive_basis.tests import solve as primitive_solve
        system=[({0:F(1,2**2500),1:F(1,3**1600)},F(0)),({0:F(1)},F(0))]
        old=primitive_solve(system);out=solve(system)
        self.record('lcm_avoidance',system,out,{'primitive':old})
        self.assertEqual(old['status'],'LIMIT')
        self.assertEqual(old['arithmetic']['first_limit']['operation'],'lcm')
        self.assertEqual(out['solution'],['0','0'])

    def test_bad_denominator_and_singular_prime_do_not_imply_rational_singular(self):
        p=1073741789
        cases=[('bad_denominator',[({0:F(1,p)},F(1,p))],'bad_denominator_primes'),
               ('singular_prime',[({0:F(p)},F(p))],'singular_primes')]
        for name,system,field in cases:
            out=solve(system);self.record(name,system,out)
            self.assertEqual(out['solution'],['1'])
            self.assertEqual(out['stats'][field],1)
            self.assertEqual(out['stats']['primes_tried'],2)

    def test_premature_alias_rejected_by_exact_equations(self):
        p=1073741789
        system=[({0:F(1)},F(p+1))]
        out=solve(system);self.record('early_alias',system,out)
        self.assertEqual(out['solution'],[str(p+1)])
        self.assertGreater(out['stats']['residual_rejections'],0)
        self.assertGreater(out['stats']['primes_used'],1)

    def test_singular_and_schedule_exhaustion_are_unresolved_not_infeasible(self):
        cases=[('singular',[({0:F(1),1:F(1)},F(1)),({0:F(2),1:F(2)},F(2))]),
               ('inconsistent',[({0:F(1),1:F(1)},F(1)),({0:F(2),1:F(2)},F(3))]),
               ('finite_schedule',[({0:F(1)},F(2**100))])]
        for name,system in cases:
            with patch.dict(POLICY,primes=2):out=solve(system)
            self.record(name,system,out)
            self.assertEqual(out['status'],'UNRESOLVED_MODULAR_RECONSTRUCTION')
            self.assertIsNone(out['solution']);self.assertFalse(out['network_UNSAFE'])
            self.assertEqual(out['stats']['primes_tried'],2)

    def test_large_intrinsic_solution_stays_unresolved_with_full_schedule(self):
        p=2**2100
        system=[({0:F(1),1:F(-p)},F(0)),({1:F(1)},F(p))]
        out=solve(system);self.record('intrinsic_large_answer',system,out,
                                     {'analytic_solution_bits':[4201,2101]})
        self.assertIn(out['status'],('LIMIT','UNRESOLVED_MODULAR_RECONSTRUCTION'))
        self.assertIsNone(out['solution'])
        self.assertLessEqual(out['stats']['primes_used'],128)

    def test_corrupt_modular_candidate_never_bypasses_exact_residual(self):
        system=[({0:F(1)},F(1))]
        with patch.dict(POLICY,primes=2),patch('modular_basis.engine.solve_field',return_value=[0]):
            out=solve(system)
        self.record('corrupt_field',system,out)
        self.assertEqual(out['status'],'UNRESOLVED_MODULAR_RECONSTRUCTION')
        self.assertEqual(out['stats']['residual_rejections'],2)

    def test_crt_bits_and_post_field_deadline(self):
        system=[({0:F(1)},F(2**100))]
        with patch.dict(POLICY,max_bits=100):
            out=solve(system)
        self.record('CRT_bit_cap',system,out)
        self.assertEqual(out['status'],'LIMIT')
        # Input fits: 2**100 is 101 bits, so use a second case to reach CRT.
        system=[({0:F(1)},F(2**99))]
        with patch.dict(POLICY,max_bits=100):out=solve(system)
        self.record('CRT_merge_cap',system,out)
        self.assertEqual(out['status'],'LIMIT')
        self.assertEqual(out['arithmetic']['first_limit']['phase'],'CRT')
        def expired(system,p,b,st):
            answer=solve_field(system,p,b,st);b.deadline=time.monotonic()-1
            return answer
        with patch('modular_basis.engine.solve_field',side_effect=expired):
            out=solve([({0:F(1)},F(1))])
        self.assertEqual(out['status'],'TIMEOUT')
        self.assertIsNone(out['solution'])

    def test_reconstruction_sign_fraction_and_no_large_congruence_product(self):
        from modular_basis.engine import reconstruct
        b=Budget(time.monotonic()+5)
        for x in (F(0),F(1),F(-1),F(123,17),F(-331,13)):
            modulus=1073741789
            residue=(x.numerator*pow(x.denominator,-1,modulus))%modulus
            self.assertEqual(reconstruct(residue,modulus,b),x)
        with patch.dict(POLICY,primes=2):
            out=solve([({0:F(2**2100)},F(1))])
        self.record('small_fraction_not_yet_reconstructable',[({0:F(2**2100)},F(1))],out)
        self.assertEqual(out['status'],'UNRESOLVED_MODULAR_RECONSTRUCTION')

if __name__=='__main__':unittest.main()
