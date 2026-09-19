import copy
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from lp_sandwich.check import check,identity,inspect_bounds,rational,strict_json,matrix_rows
from lp_sandwich.propose import propose_to,save_new

ROOT=Path('/data1/Kane/MOE/ACT')
OBSERVATIONS=[]


def csr(rows,n):
    data=[];indices=[];ptr=[0]
    for row in rows:
        for j,v in enumerate(row):
            if v:indices.append(j);data.append(v)
        ptr.append(len(data))
    return {'shape':[len(rows),n],'indptr':ptr,'indices':indices,'data':data}


def fixture(offset=0,x=0):
    lp={'matrix_format':'csr_v1','c':[1],'offset':offset,'lower':[-1],'upper':[1],
        'A':csr([],1),'b':[],'E':csr([],1),'h':[]}
    return bundle(lp,[x])


def bundle(lp,x,y=None,z=None):
    s={'schema':'LP_OBLIGATION_IDENTITY_V1','request_id':'1'*64,'source_sha256':'2'*64,
       'export_sha256':'3'*64,'pair':[0,1],'property_index':0,'property':{'q':[1,-1],'constant':0},
       'lp_sha256':identity(lp),'acceptance_threshold':1e-7}
    p=None if x is None else {'lp_sha256':identity(lp),'statement_sha256':identity(s),
        'x':x,'claimed_objective':str(rational(lp['offset'])+sum((rational(a)*rational(b) for a,b in zip(lp['c'],x)),F(0)))}
    d={'lp_sha256':identity(lp),'inequality_dual':[0]*len(lp['b']) if y is None else y,
       'equality_dual':[0]*len(lp['h']) if z is None else z}
    from act.back_end.solver.sparse_lp_certificate import evaluate
    d['claimed_lower_bound']=str(evaluate(lp,d)[0])
    return {'schema':'LP_SANDWICH_V1','statement':s,'lp':lp,'primal':p,'dual':d}


def checked(b):return check(b,identity(b['statement']))


class Checks(unittest.TestCase):
    def test_multidimensional_sparse_rational_optimum(self):
        lp={'matrix_format':'csr_v1','c':[1,-2,'1/3'],'offset':'2/7',
            'lower':[-1,0,'3/5'],'upper':[2,1,'3/5'],
            'A':csr([[1,0,0],[0,-1,0]],3),'b':[2,0],
            'E':csr([[0,0,5]],3),'h':[3]}
        r=checked(bundle(lp,[-1,1,'3/5'],y=[0,0],z=['1/15']))
        self.assertEqual(F(r['lower_bound']),F(-88,35))
        self.assertEqual(r['lower_bound'],r['upper_bound']);self.assertTrue(r['exact_optimality'])
        self.assertEqual(r['checked_rows'],{'A':2,'E':1});self.assertEqual(r['checked_variables'],3)

    def test_exact_primal_dual_optimum(self):
        b=fixture(x=-1);r=checked(b)
        self.assertEqual(r['upper_bound'],'-1');self.assertEqual(r['lower_bound'],'-1')
        self.assertTrue(r['exact_optimality']);self.assertEqual(r['classification'],'LP_NONPOSITIVE_FEASIBLE_POINT')
        self.assertFalse(r['network_UNSAFE']);self.assertFalse(r['network_SAFE'])

    def test_positive_primal_does_not_prove_positive_minimum(self):
        r=checked(fixture(x=1))
        self.assertEqual(r['upper_bound'],'1');self.assertEqual(r['lower_bound'],'-1')
        self.assertEqual(r['classification'],'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION')

    def test_positive_lower_bound(self):
        r=checked(fixture(offset=2,x=1))
        self.assertEqual(r['lower_bound'],'1');self.assertEqual(r['upper_bound'],'3')
        self.assertEqual(r['classification'],'LP_POSITIVE_LOWER_BOUND');self.assertFalse(r['exact_optimality'])

    def test_same_weak_lower_bound_opposite_lp_optimum(self):
        b=fixture(x=1);lp=copy.deepcopy(b['lp']);lp['A']=csr([[-1]],1);lp['b']=[-1]
        safe=bundle(lp,[1]);r=checked(safe)
        self.assertEqual(r['lower_bound'],'-1');self.assertEqual(r['upper_bound'],'1')
        self.assertFalse(r['exact_optimality'])
        stronger=checked(bundle(lp,[1],y=[-1]));self.assertEqual(stronger['lower_bound'],'1')
        self.assertTrue(stronger['exact_optimality'])
        bad=checked(fixture(x=-1));self.assertEqual(bad['upper_bound'],'-1')

    def test_tiny_equality_error_is_not_feasible(self):
        lp=fixture()['lp'];lp['E']=csr([[1]],1);lp['h']=['1/3']
        r=checked(bundle(lp,[1/3]))
        self.assertEqual(r['primal_status'],'NOT_EXACTLY_FEASIBLE');self.assertIsNone(r['upper_bound'])
        self.assertEqual(r['violation_counts']['equality'],1)
        self.assertEqual(checked(bundle(lp,['1/3']))['primal_status'],'EXACT_FEASIBLE')

    def test_inequality_and_box_violations_never_upper_bounds(self):
        lp=fixture()['lp'];lp['A']=csr([[1]],1);lp['b']=[0]
        for x in ('1/100000000000000000000',2,-2):
            r=checked(bundle(lp,[x]));self.assertIsNone(r['upper_bound'])
            self.assertEqual(r['primal_status'],'NOT_EXACTLY_FEASIBLE')

    def test_missing_primal_or_dual(self):
        b=fixture(x=-1);b['dual']=None
        r=checked(b);self.assertEqual(r['classification'],'LP_NONPOSITIVE_FEASIBLE_POINT')
        self.assertIsNone(r['lower_bound']);b['primal']=None
        self.assertEqual(checked(b)['classification'],'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION')

    def test_zero_and_threshold_are_not_strict_positivity(self):
        b=fixture(x=0);b['dual']=None
        self.assertEqual(checked(b)['classification'],'LP_NONPOSITIVE_FEASIBLE_POINT')
        lp=b['lp'];lp['c']=[0];lp['offset']=1e-7
        self.assertEqual(checked(bundle(lp,[0]))['classification'],'LP_UPPER_AT_OR_BELOW_ACCEPTANCE_THRESHOLD')

    def test_objective_and_dual_overclaims_rejected(self):
        b=fixture(x=-1)
        for which,key,value in [('primal','claimed_objective','-2'),('dual','claimed_lower_bound','0')]:
            bad=copy.deepcopy(b);bad[which][key]=value
            with self.assertRaises(ValueError):checked(bad)

    def test_request_source_property_pair_binding(self):
        b=fixture();expected=identity(b['statement'])
        for key,v in [('request_id','a'*64),('source_sha256','b'*64),('export_sha256','c'*64),
                      ('pair',[1,2]),('property_index',1),('property',{'q':[-1,1],'constant':0})]:
            bad=copy.deepcopy(b);bad['statement'][key]=v
            with self.assertRaises(ValueError):check(bad,expected)

    def test_changed_lp_point_binding_and_incomplete_fields(self):
        b=fixture()
        for key in ('c','lower','upper'):
            bad=copy.deepcopy(b);bad['lp'][key]=[2]
            with self.assertRaises(ValueError):checked(bad)
        bad=copy.deepcopy(b);bad['primal']['statement_sha256']='0'*64
        with self.assertRaises(ValueError):checked(bad)
        bad=copy.deepcopy(b);bad.pop('dual')
        with self.assertRaises(ValueError):checked(bad)

    def test_invalid_sparse_and_numerical_values(self):
        for m in [{'shape':[1,1],'indptr':[0,2],'indices':[0,0],'data':[1,1]},
                  {'shape':[1,1],'indptr':[0,1],'indices':[1],'data':[1]}]:
            with self.assertRaises(ValueError):list(matrix_rows(m,1,lambda:None))
        for v in (True,None,float('nan'),float('inf')):
            with self.assertRaises(ValueError):rational(v)
        for raw in ('{"a":1,"a":2}','{"x":NaN}'):
            with self.assertRaises(ValueError):strict_json(raw)

    def test_dual_sign_residual_and_reference_differential(self):
        from act.back_end.solver.sparse_lp_certificate import check as reference
        lp=fixture()['lp'];lp['A']=csr([[1]],1);lp['b']=[1]
        for y in (0,-1,'-1/7'):
            b=bundle(lp,[0],y=[y]);r=checked(b)
            self.assertEqual(r['lower_bound'],reference(lp,b['dual'])['checked_lower_bound'])
        b=bundle(lp,[0]);b['dual']['inequality_dual']=[1]
        with self.assertRaises(ValueError):checked(b)

    def test_zero_width_and_unrestricted_equality_dual(self):
        lp=fixture()['lp'];lp['lower']=[0];lp['upper']=[0];lp['E']=csr([[1]],1);lp['h']=[0]
        for z in (-7,8):
            r=checked(bundle(lp,[0],z=[z]));self.assertTrue(r['exact_optimality'])

    def test_deadline_inside_check(self):
        b=fixture();count=[0]
        def tick():
            count[0]+=1
            if count[0]>=4:raise TimeoutError('test cutoff')
        with self.assertRaises(TimeoutError):check(b,identity(b['statement']),tick)

    def test_relocated_stdlib_check_hash_mutation_and_timeout(self):
        b=fixture(x=-1)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp);shutil.copyfile(ROOT/'lp_sandwich/check.py',root/'verify.py')
            save_new(root/'bundle.json',b);raw=(root/'bundle.json').read_bytes()
            cmd=[sys.executable,'-I','-S',str(root/'verify.py'),str(root/'bundle.json'),
                 '--bundle-sha256',hashlib.sha256(raw).hexdigest(),'--statement-sha256',identity(b['statement']),
                 '--timeout-seconds','10']
            p=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=15,check=True)
            r=json.loads(p.stdout);self.assertFalse(r['solver_or_model_imported']);self.assertTrue(r['site_disabled'])
            self.assertTrue(r['isolated']);self.assertEqual(r['upper_bound'],'-1')
            late=subprocess.run(cmd[:-1]+['0.0000001'],cwd=root,capture_output=True,text=True,timeout=5)
            self.assertEqual(late.returncode,3);self.assertEqual(json.loads(late.stdout)['status'],'TIMEOUT')
            (root/'bundle.json').write_bytes(raw+b' ')
            bad=subprocess.run(cmd,cwd=root,capture_output=True,text=True,timeout=15)
            self.assertEqual(bad.returncode,2)


class Capture(unittest.TestCase):
    def test_one_shot_native_capture_and_approximate_point(self):
        for fractional in (False,True):
            b=fixture(x=-1)
            if fractional:
                lp=b['lp'];lp['E']=csr([[1]],1);lp['h']=['1/3'];lp['c']=[0];b=bundle(lp,['1/3'])
            with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
                root=Path(tmp)/'attempt';t=propose_to(b['lp'],b['statement'],root,deadline=time.monotonic()+30)
                self.assertEqual(t['status'],'COMPLETED_DIAGNOSTIC');self.assertEqual(t['native_calls'],1)
                native=json.loads((root/'native.json').read_text());r=json.loads((root/'diagnostic.json').read_text())
                self.assertIn('marginals',native['lower']);self.assertTrue(native['success'])
                self.assertEqual(r['primal_status'],'NOT_EXACTLY_FEASIBLE' if fractional else 'EXACT_FEASIBLE')
                if fractional:self.assertIsNone(r['upper_bound'])
                OBSERVATIONS.append({'case':'float_third' if fractional else 'exact_endpoint',
                    'native_status':native['status'],'native_calls':t['native_calls'],'result':r})
                with self.assertRaises(FileExistsError):propose_to(b['lp'],b['statement'],root,deadline=time.monotonic()+30)

    def test_native_infeasible_status_not_promoted_to_a_proof(self):
        b=fixture();lp=b['lp'];lp['A']=csr([[-1]],1);lp['b']=[-2];b=bundle(lp,[0])
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp)/'infeasible';t=propose_to(lp,b['statement'],root,deadline=time.monotonic()+30)
            self.assertEqual(t['native_calls'],1);self.assertEqual(t['status'],'COMPLETED_DIAGNOSTIC')
            r=json.loads((root/'diagnostic.json').read_text())
            self.assertEqual(r['classification'],'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION')
            self.assertFalse(r['network_UNSAFE'])

    def test_capture_preserved_on_checker_failure(self):
        b=fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp)/'failed'
            with patch('lp_sandwich.propose.check',side_effect=TimeoutError('test checker cutoff')):
                t=propose_to(b['lp'],b['statement'],root,deadline=time.monotonic()+30)
            self.assertEqual(t['status'],'TIMEOUT');self.assertTrue((root/'native.json').exists())
            self.assertEqual(t['native_calls'],1);self.assertFalse((root/'diagnostic.json').exists())

    def test_expired_deadline_no_solver(self):
        b=fixture()
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp)/'expired'
            with self.assertRaises(TimeoutError):propose_to(b['lp'],b['statement'],root,deadline=time.monotonic()-1)
            self.assertFalse(root.exists())
