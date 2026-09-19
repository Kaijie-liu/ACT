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
from primitive_basis.engine import POLICY, Budget, Limit, Singular, statistics, eliminate, primitive
from primitive_basis.propose import propose

ARTIFACT_ROOT=None
OBSERVATIONS=[]


def generate(lp,s,c,h,seconds=30):
    return propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+seconds)


def solve(system,seconds=30):
    b=Budget(time.monotonic()+seconds);st=statistics();start=time.monotonic();answer=None;error=None
    try:answer=eliminate(deepcopy(system),b,st);status='CANDIDATE_ONLY'
    except Limit as e:status,error='LIMIT',str(e)
    except Singular as e:status,error='UNRESOLVED_SINGULAR_BASIS',str(e)
    except TimeoutError as e:status,error='TIMEOUT',str(e)
    return {'status':status,'error':error,'solution':None if answer is None else list(map(str,answer)),
            'stats':st,'arithmetic':b.arithmetic,'operations':b.operations,'seconds':time.monotonic()-start,
            'policy':dict(POLICY),'network_SAFE':False,'network_UNSAFE':False}


class Controls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global ARTIFACT_ROOT
        ARTIFACT_ROOT=Path(tempfile.mkdtemp(prefix='primitive_basis_controls_',dir=ROOT/'data/moe/results'))
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

    def test_common_row_scale_4096_same_pivots_small_exact_solution(self):
        p,k=2**3000+1,2**2000
        system=[({0:F(p),1:F(1)},F(p+1)),({0:F(k),1:F(2*k)},F(3*k))]
        st=old_stats()
        with self.assertRaises(Limit):old_eliminate(deepcopy(system),OldBudget(time.monotonic()+30),st)
        out=solve(system)
        self.record('avoidable_row_scale',system,out,{'old_status':'LIMIT','old_stats':st,
            'analytic_solution':['1','1'],'scope':'synthetic demonstration, not the four real LPs'})
        self.assertEqual(out['solution'],['1','1']);self.assertIsNone(out['arithmetic']['first_limit'])
        self.assertEqual([(t['column'],t['row']) for t in out['stats']['pivot_prefix']],[(0,0),(1,1)])
        self.assertLessEqual(out['arithmetic']['max_integer_bits'],4096)
        self.assertGreater(out['stats']['content_divisions'],0)

    def test_positive_mechanism_original_lp_checked_not_solver_assertion(self):
        p,k=2**3000+1,2**2000
        lp={'matrix_format':'csr_v1','c':[-1,0],'offset':0,'lower':[-2,-2],'upper':[2,2],
            'E':csr([[p,1],[k,2*k]],2),'h':[p+1,3*k],'A':csr([],2),'b':[]}
        s,c,h=hints(lp,[col('x',0),col('x',1)],[zero('E_residual',0),zero('E_residual',1)])
        old=old_propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+30)
        out=generate(lp,s,c,h);self.record_lp('checked_row_scale',lp,s,c,h,out)
        save_new(self.root/'checked_row_scale/old_proposal.json',old)
        self.assertEqual(old['status'],'LIMIT');self.assertEqual(out['status'],'CANDIDATE_ONLY')
        self.assertFalse(out['feasibility_certified']);self.assertEqual(out['solver_calls'],0)
        d=check(out['bundle'],identity(s));self.assertEqual(d['primal_status'],'EXACT_FEASIBLE')
        self.assertEqual(d['upper_bound'],'-1');self.assertIsNone(d['lower_bound'])

    def test_intrinsically_large_solution_rejected_with_location(self):
        p=2**2100
        system=[({0:F(1),1:F(-p)},F(0)),({1:F(1)},F(p))]
        out=solve(system);self.record('large_solution',system,out,
             {'analytic_solution_bits':[4201,2101],'scope':'this constructed system only'})
        self.assertEqual(out['status'],'LIMIT');self.assertIsNone(out['solution'])
        event=out['arithmetic']['first_limit'];self.assertEqual(event['phase'],'back_substitution')
        self.assertEqual(event['observed'],4201);self.assertEqual(event['cap'],4096)

    def test_denominator_clearing_can_be_worse_than_rational(self):
        a,b=2**2500,3**1600
        system=[({0:F(1,a),1:F(1,b)},F(0)),({0:F(1)},F(0))]
        old=old_eliminate(deepcopy(system),OldBudget(time.monotonic()+30),old_stats())
        out=solve(system);self.record('lcm_disadvantage',system,out,{'old_solution':list(map(str,old))})
        self.assertEqual(old,[0,0]);self.assertEqual(out['status'],'LIMIT')
        self.assertEqual(out['arithmetic']['first_limit']['operation'],'lcm')
        self.assertGreater(out['arithmetic']['first_limit']['observed'],4096)

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

    def test_primitive_divides_rhs_and_sign_together(self):
        b=Budget(time.monotonic()+5);st=statistics()
        self.assertEqual(primitive({0:-6,1:-12},-9,b,st),({0:2,1:4},3))
        self.assertEqual(primitive({0:6,1:12},0,b,st),({0:1,1:2},0))

    def test_zero_or_inconsistent_selected_basis_not_lp_infeasibility(self):
        for rhs in (2,3):
            system=[({0:F(1),1:F(1)},F(1)),({0:F(2),1:F(2)},F(rhs))]
            out=solve(system);self.record('singular_'+str(rhs),system,out)
            self.assertEqual(out['status'],'UNRESOLVED_SINGULAR_BASIS')
            self.assertFalse(out['network_UNSAFE'])

    def test_raw_products_rejected_before_favorable_cancellation(self):
        system=[({0:F(17),1:F(1)},F(0)),({0:F(1),1:F(17)},F(0))]
        with patch.dict(POLICY,max_bits=8):out=solve(system)
        self.record('raw_product_guard',system,out)
        self.assertEqual(out['status'],'LIMIT');self.assertEqual(out['arithmetic']['first_limit']['operation'],'row_product')
        self.assertEqual(out['arithmetic']['first_limit']['observed'],9)

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
        self.assertEqual(len(out['stats']['pivot_prefix']),64)

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
        from primitive_basis.review import compare_checked
        ref={'status':'CHECKED_LP_DIAGNOSTIC','upper_bound':'-1'}
        v={**ref,'seconds':.1,'isolated':True,'site_disabled':True,'solver_or_model_imported':False}
        compare_checked(v,ref)
        for field,value in [('isolated',False),('site_disabled',False),('solver_or_model_imported',True),
                            ('seconds',31),('upper_bound','-2')]:
            bad={**v,field:value}
            with self.assertRaises(ValueError):compare_checked(bad,ref)
        bad=deepcopy(v);bad.pop('site_disabled')
        with self.assertRaises(ValueError):compare_checked(bad,ref)


if __name__=='__main__':unittest.main()
