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
from exact_primal.propose import Budget,Limit
from exact_basis.propose import col,manifest,propose,solve,LIMITS
from single_check_portable.execution import ROOT,ACT,save_new
from portable_proof.runtime import digest

OBSERVATIONS=[]

def case():
    lp={'matrix_format':'csr_v1','c':[1,0],'offset':-1,'lower':[0,0],'upper':[1,1],
        'E':csr([[1,1]],2),'h':[1],'A':csr([[3,0]],2),'b':[1]}
    s=deepcopy(fixture()['statement']);s['lp_sha256']=identity(lp)
    c={'lp_sha256':identity(lp),'statement_sha256':identity(s),'x':[.333,.667]}
    h=manifest(lp,s,c,[col('x',0),col('x',1)],[{'column':col('slack',0),'at':'zero'}])
    return lp,s,c,h

def run(t):
    lp,s,c,h=t
    return propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+10)

def checked(r):return check(r['bundle'],r['statement_sha256'])

class Controls(unittest.TestCase):
    def test_basis_slack_exact_map_and_relocation(self):
        t=case();old=deepcopy(t);r=run(t)
        self.assertEqual(t,old);self.assertEqual(r['status'],'CANDIDATE_ONLY')
        self.assertEqual(r['bundle']['primal']['x'],['1/3','2/3']);self.assertEqual(r['slacks'],['0'])
        self.assertFalse(r['feasibility_certified']);self.assertEqual(checked(r)['upper_bound'],'-2/3')
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            base=Path(tmp);src=base/'source';src.mkdir()
            save_new(src/'bundle.json',r['bundle']);shutil.copyfile(ROOT/'lp_sandwich/check.py',src/'verify.py')
            moved=base/'moved';shutil.copytree(src,moved)
            def cmd():return [ACT,'-I','-S',str(moved/'verify.py'),str(moved/'bundle.json'),
                '--bundle-sha256',digest((moved/'bundle.json').read_bytes()),
                '--statement-sha256',r['statement_sha256'],'--timeout-seconds','10']
            p=subprocess.run(cmd(),cwd=moved,capture_output=True,text=True,check=True,timeout=15)
            d=json.loads(p.stdout);self.assertEqual(d['upper_bound'],'-2/3')
            self.assertTrue(d['isolated'] and d['site_disabled']);self.assertFalse(d['solver_or_model_imported'])
            bad=deepcopy(r['bundle']);bad['primal']['x'][0]='0'
            (moved/'bundle.json').write_text(json.dumps(bad))
            self.assertNotEqual(subprocess.run(cmd(),cwd=moved,capture_output=True,timeout=15).returncode,0)
        OBSERVATIONS.append({'case':'original E plus A/slack','point':['1/3','2/3'],'upper':'-2/3',
            'moved_isolated':True,'solver_calls':0,'network_UNSAFE':False})

    def test_permuted_original_rows_and_columns(self):
        t=case();a=run(t);t[3]['rows'].reverse();t[3]['basic_columns'].reverse();b=run(t)
        self.assertEqual(a['bundle'],b['bundle']);self.assertEqual(a['slacks'],b['slacks'])

    def test_negative_basic_slack_does_not_pass_original_lp(self):
        lp,s,c,h=case();h['basic_columns']=[col('x',0),col('slack',0)]
        h['anchors']=[{'column':col('x',1),'at':'lower'}]
        r=run((lp,s,c,h));self.assertEqual(r['slacks'],['-2'])
        d=checked(r);self.assertEqual(d['primal_status'],'NOT_EXACTLY_FEASIBLE');self.assertIsNone(d['upper_bound'])

    def test_upper_anchor_and_candidate_anchor(self):
        for at,expected in [('upper',['0','1']),('candidate',[str(1-F(.667)),str(F(.667))])]:
            lp,s,c,h=case();h['basic_columns']=[col('x',0),col('slack',0)]
            h['anchors']=[{'column':col('x',1),'at':at}]
            r=run((lp,s,c,h));self.assertEqual(r['bundle']['primal']['x'],expected)
            self.assertEqual(checked(r)['primal_status'],'EXACT_FEASIBLE')

    def test_missing_duplicate_or_overlapping_coordinates(self):
        for change in ('missing_anchor','duplicate_anchor','duplicate_basic','overlap','bad_index','bool_index'):
            lp,s,c,h=case()
            if change=='missing_anchor':h['anchors']=[]
            elif change=='duplicate_anchor':h['anchors']*=2
            elif change=='duplicate_basic':h['basic_columns']=[col('x',0)]*2
            elif change=='overlap':h['anchors']=[{'column':col('x',0),'at':'lower'}]
            elif change=='bad_index':h['basic_columns'][0]=col('x',999)
            else:h['basic_columns'][0]=col('x',True)
            self.assertEqual(run((lp,s,c,h))['status'],'ERROR',change)

    def test_original_rows_and_transforms_bound(self):
        for change in ('missing','duplicate','hash','scaled','slack_sign','anchor_value'):
            lp,s,c,h=case()
            if change=='missing':h['rows'].pop()
            elif change=='duplicate':h['rows'][1]=deepcopy(h['rows'][0])
            elif change=='hash':h['rows'][0]['row_sha256']='0'*64
            elif change=='scaled':h['coordinates']='PRESOLVED_SCALED'
            elif change=='slack_sign':h['coordinates']='AX_MINUS_SLACK'
            else:h['anchors'][0]['value']=0
            self.assertEqual(run((lp,s,c,h))['status'],'ERROR',change)

    def test_external_hint_candidate_and_statement_binding(self):
        for what in ('hint','point','request','LP'):
            lp,s,c,h=case();es,eh=identity(s),identity(h)
            if what=='hint':h['basic_columns'].reverse()
            elif what=='point':c['x'][0]=0
            elif what=='request':s['request_id']='f'*64
            else:lp['b']=[2]
            r=propose(lp,s,c,h,es,eh,deadline=time.monotonic()+10)
            self.assertEqual(r['status'],'ERROR',what)

    def test_dependent_equalities_return_unresolved_not_infeasible(self):
        lp={'matrix_format':'csr_v1','c':[1,0],'offset':0,'lower':[0,0],'upper':[1,1],
            'E':csr([[1,1],[2,2]],2),'h':[1,2],'A':csr([],2),'b':[]}
        s=deepcopy(fixture()['statement']);s['lp_sha256']=identity(lp)
        c={'lp_sha256':identity(lp),'statement_sha256':identity(s),'x':[.5,.5]}
        h=manifest(lp,s,c,[col('x',0),col('x',1)],[])
        r=run((lp,s,c,h));self.assertEqual(r['status'],'UNRESOLVED_SINGULAR_BASIS')
        self.assertIsNone(r['bundle']);self.assertFalse(r['network_UNSAFE'])

    def test_fixed_variable_bounds_are_exact_anchors(self):
        lp,s,c,h=case();lp['lower'][1]=lp['upper'][1]='2/3';s['lp_sha256']=identity(lp)
        c.update(lp_sha256=identity(lp),statement_sha256=identity(s))
        h=manifest(lp,s,c,[col('x',0),col('slack',0)],[{'column':col('x',1),'at':'lower'}])
        r=run((lp,s,c,h));self.assertEqual(r['bundle']['primal']['x'],['1/3','2/3'])
        self.assertEqual(checked(r)['primal_status'],'EXACT_FEASIBLE')

    def test_sparse_fill_in_is_measured_and_capped(self):
        rows=[({0:F(1),1:F(1),2:F(1)},F(3)),({0:F(1),1:F(2)},F(3)),({2:F(1)},F(1))]
        stats={'peak_elimination_nnz':0,'fill_in_insertions':0}
        self.assertEqual(solve(rows,Budget(time.monotonic()+10),stats),[F(1)]*3)
        self.assertGreater(stats['fill_in_insertions'],0)
        with patch.dict(LIMITS,live_elimination_nnz=2):
            with self.assertRaises(Limit):solve(rows,Budget(time.monotonic()+10),{'peak_elimination_nnz':0,'fill_in_insertions':0})

    def test_size_and_deadline_limits_no_candidate(self):
        with patch.dict(LIMITS,equations=1):self.assertEqual(run(case())['status'],'LIMIT')
        lp,s,c,h=case();r=propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()-1)
        self.assertEqual(r['status'],'TIMEOUT');self.assertIsNone(r['bundle'])
        with self.assertRaises(ValueError):propose(lp,s,c,h,identity(s),identity(h),deadline=time.monotonic()+301)

    def test_box_only_empty_basis(self):
        lp=fixture()['lp'];s=fixture()['statement'];s['lp_sha256']=identity(lp)
        c={'lp_sha256':identity(lp),'statement_sha256':identity(s),'x':[0]}
        h=manifest(lp,s,c,[],[{'column':col('x',0),'at':'lower'}])
        r=run((lp,s,c,h));self.assertEqual(checked(r)['upper_bound'],'-1')

if __name__=='__main__':unittest.main()
