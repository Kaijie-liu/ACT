from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from lp_sandwich.tests import fixture,csr
from lp_sandwich.check import identity,check
from exact_primal.propose import propose,POLICY,Budget,system_solution,Limit
from single_check_portable.execution import ROOT,ACT,save_new
from portable_proof.runtime import digest

OBSERVATIONS=[]

def run(lp,x):
    s=deepcopy(fixture()['statement']);s['lp_sha256']=identity(lp)
    c={'statement_sha256':identity(s),'lp_sha256':identity(lp),'x':x}
    return propose(lp,s,c,identity(s),deadline=time.monotonic()+10)

def verdict(r):return check(r['bundle'],r['statement_sha256'])

class Controls(unittest.TestCase):
    def test_one_third_exact_reconstruction_and_portable_check(self):
        lp=fixture()['lp'];lp['E']=csr([[3]],1);lp['h']=[1];lp['offset']=-1
        old=deepcopy(lp);r=run(lp,[1/3]);self.assertEqual(lp,old)
        self.assertEqual(r['status'],'CANDIDATE_ONLY');self.assertFalse(r['feasibility_certified'])
        self.assertEqual(r['bundle']['primal']['x'],['1/3'])
        d=verdict(r);self.assertEqual(d['upper_bound'],'-2/3')
        self.assertFalse(d['exact_optimality']);self.assertIsNone(d['lower_bound'])
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            base=Path(tmp);src=base/'source';src.mkdir()
            save_new(src/'bundle.json',r['bundle']);shutil.copyfile(ROOT/'lp_sandwich/check.py',src/'verify.py')
            moved=base/'moved';shutil.copytree(src,moved)
            def command():return [ACT,'-I','-S',str(moved/'verify.py'),str(moved/'bundle.json'),
                '--bundle-sha256',digest((moved/'bundle.json').read_bytes()),'--statement-sha256',r['statement_sha256'],
                '--timeout-seconds','10']
            p=subprocess.run(command(),cwd=moved,capture_output=True,text=True,timeout=15,check=True)
            v=json.loads(p.stdout);self.assertEqual(v['upper_bound'],'-2/3')
            self.assertFalse(v['solver_or_model_imported']);self.assertTrue(v['site_disabled'] and v['isolated'])
            bad=deepcopy(r['bundle']);bad['primal']['claimed_objective']='-1'
            (moved/'bundle.json').write_text(json.dumps(bad))
            self.assertNotEqual(subprocess.run(command(),cwd=moved,capture_output=True,timeout=15).returncode,0)
        OBSERVATIONS.append({'case':'3x=1','status':d['classification'],'upper':d['upper_bound'],
            'relocated_isolated':True,'network_UNSAFE':False,'solver_calls':0})

    def test_active_inequality_reconstructs_vertex(self):
        lp=fixture()['lp'];lp['c']=[-1];lp['A']=csr([[3]],1);lp['b']=[1]
        r=run(lp,[1/3]);self.assertEqual(r['bundle']['primal']['x'],['1/3'])
        self.assertEqual(verdict(r)['upper_bound'],'-1/3')
        self.assertEqual(r['selected_rows'],[{'kind':'inequality','index':0}])

    def test_full_checker_rejects_nonselected_constraint(self):
        lp={'matrix_format':'csr_v1','c':[1,1],'offset':-2,'lower':[0,0],'upper':[1,1],
            'E':csr([[1,1]],2),'h':[1],'A':csr([[1,0]],2),'b':['1/5']}
        # LP is feasible, e.g. (1/10,9/10). Equality-only correction with y fixed
        # to the original .1 instead raises x above 1/5: repair is not proof.
        r=run(lp,[.1,.1]);self.assertEqual(r['status'],'CANDIDATE_ONLY')
        d=verdict(r);self.assertEqual(d['primal_status'],'NOT_EXACTLY_FEASIBLE');self.assertIsNone(d['upper_bound'])

    def test_guess_conflict_is_not_lp_infeasibility(self):
        lp=fixture()['lp'];lp['lower']=[0];lp['upper']=['1/1000000000']
        r=run(lp,[0]);self.assertEqual(r['status'],'UNRESOLVED_ACTIVE_SYSTEM')
        self.assertIsNone(r['bundle']);self.assertFalse(r['network_UNSAFE']) # Original box is feasible.

    def test_dependent_rows_and_zero_width_box(self):
        lp=fixture()['lp'];lp['E']=csr([[3],[6],[0]],1);lp['h']=[1,2,0]
        lp['lower']=lp['upper']=['1/3']
        r=run(lp,[1/3]);self.assertEqual(verdict(r)['upper_bound'],'1/3')
        self.assertEqual(r['pivot_columns'],[0])

    def test_rank_deficiency_preserves_free_candidate(self):
        lp={'matrix_format':'csr_v1','c':[1,1],'offset':-2,'lower':[0,0],'upper':[1,1],
            'A':csr([],2),'b':[],'E':csr([[1,1]],2),'h':[1]}
        r=run(lp,[.49,.5]);self.assertEqual(r['bundle']['primal']['x'],['1/2','1/2'])
        self.assertEqual(verdict(r)['upper_bound'],'-1')

    def test_positive_upper_not_safety_proof(self):
        lp=fixture()['lp'];lp['E']=csr([[3]],1);lp['h']=[1]
        d=verdict(run(lp,[1/3]));self.assertEqual(d['upper_bound'],'1/3')
        self.assertEqual(d['classification'],'UNRESOLVED_CANDIDATE_VS_LP_RELAXATION')
        self.assertFalse(d['network_SAFE'])

    def test_binary_float_coefficients_not_decimal_rounded(self):
        lp=fixture()['lp'];lp['E']=csr([[.1]],1);lp['h']=[.03]
        r=run(lp,[.3]);self.assertEqual(F(r['bundle']['primal']['x'][0]),F(.03)/F(.1))
        self.assertNotEqual(F(r['bundle']['primal']['x'][0]),F(3,10))
        self.assertEqual(verdict(r)['primal_status'],'EXACT_FEASIBLE')

    def test_point_and_statement_binding(self):
        f=fixture();lp=f['lp'];s=f['statement'];s['lp_sha256']=identity(lp)
        c={'lp_sha256':identity(lp),'statement_sha256':identity(s),'x':[0]}
        for key in ('lp_sha256','statement_sha256'):
            bad=deepcopy(c);bad[key]='0'*64
            self.assertEqual(propose(lp,s,bad,identity(s),deadline=time.monotonic()+10)['status'],'ERROR')
        bad=deepcopy(s);bad['pair']=[1,2]
        self.assertEqual(propose(lp,bad,c,identity(s),deadline=time.monotonic()+10)['status'],'ERROR')

    def test_limits_deadline_and_no_implicit_retry(self):
        lp=fixture()['lp'];f=fixture();s=f['statement'];s['lp_sha256']=identity(lp)
        c={'lp_sha256':identity(lp),'statement_sha256':identity(s),'x':[0]}
        r=propose(lp,s,c,identity(s),deadline=time.monotonic()-1)
        self.assertEqual(r['status'],'TIMEOUT');self.assertIsNone(r['bundle'])
        with patch.dict(POLICY,max_operations=1):self.assertEqual(run(lp,[0])['status'],'LIMIT')
        bad=deepcopy(lp);bad['c']=[1]*65
        self.assertEqual(run(bad,[0]*65)['status'],'LIMIT')
        bad=deepcopy(lp);bad['c']=[str(2**4100)]
        self.assertEqual(run(bad,[0])['status'],'LIMIT')
        with self.assertRaises(ValueError):propose(lp,s,c,identity(s),deadline=time.monotonic()+301)

    def test_multidimensional_pivot_order(self):
        b=Budget(time.monotonic()+10)
        x,piv=system_solution([({1:F(2)},F(2)),({0:F(3),1:F(1)},F(2))],[F(0),F(0)],b)
        self.assertEqual(x,[F(1,3),F(1)]);self.assertEqual(piv,[0,1])

    def test_coupled_exact_system_against_known_solution(self):
        lp={'matrix_format':'csr_v1','c':[1,-2,3],'offset':-4,'lower':[-2]*3,'upper':[2]*3,
            'A':csr([],3),'b':[],'E':csr([[2,1,0],[0,3,1],[1,0,4]],3),'h':['16/15','67/35','67/21']}
        r=run(lp,[.333,.4001,.714]);self.assertEqual(r['status'],'CANDIDATE_ONLY')
        self.assertEqual(list(map(F,r['bundle']['primal']['x'])),[F(1,3),F(2,5),F(5,7)])
        self.assertEqual(verdict(r)['primal_status'],'EXACT_FEASIBLE')

    def test_no_active_rows_does_not_invent_a_vertex(self):
        r=run(fixture()['lp'],[.2]);self.assertEqual(r['selected_rows'],[])
        self.assertEqual(F(r['bundle']['primal']['x'][0]),F(.2));self.assertEqual(r['pivot_columns'],[])
        self.assertEqual(verdict(r)['primal_status'],'EXACT_FEASIBLE')

if __name__=='__main__':unittest.main()
