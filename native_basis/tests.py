from copy import deepcopy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from lp_sandwich.check import identity,check
from lp_sandwich.tests import csr,fixture
from exact_basis.propose import propose
from exact_basis.tests import case
from native_basis.adapter import capture,map_capture
from single_check_portable.execution import ROOT,ACT,save_new,read
from portable_proof.runtime import digest

OBSERVATIONS=[]

def statement(lp):
    s=deepcopy(fixture()['statement']);s['lp_sha256']=identity(lp);return s

def construct(lp,s,r):
    m=map_capture(lp,s,r,identity(r))
    if m['status']!='MAPPED_HINT_ONLY':return m,None
    p=propose(lp,s,m['candidate'],m['hint'],identity(s),identity(m['hint']),deadline=time.monotonic()+10)
    return m,p

class Controls(unittest.TestCase):
    def test_actual_native_active_row_and_isolated_check(self):
        lp,s,_,_=case();lp['c']=[-1,0];s=statement(lp)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            base=Path(tmp);r=capture(lp,s,base/'native',deadline=time.monotonic()+10)
            self.assertEqual(r,read(base/'native/capture.json'))
            m,p=construct(lp,s,r);self.assertEqual(m['status'],'MAPPED_HINT_ONLY',r)
            self.assertEqual(p['bundle']['primal']['x'],['1/3','2/3'])
            self.assertEqual(check(p['bundle'],identity(s))['upper_bound'],'-4/3')
            out=base/'moved';out.mkdir();save_new(out/'bundle.json',p['bundle'])
            shutil.copyfile(ROOT/'lp_sandwich/check.py',out/'verify.py')
            result=subprocess.run([ACT,'-I','-S',str(out/'verify.py'),str(out/'bundle.json'),
                '--bundle-sha256',digest((out/'bundle.json').read_bytes()),'--statement-sha256',identity(s),
                '--timeout-seconds','10'],cwd=out,capture_output=True,text=True,check=True,timeout=15)
            d=json.loads(result.stdout);self.assertEqual(d['upper_bound'],'-4/3')
            self.assertFalse(d['solver_or_model_imported']);self.assertFalse(d['network_UNSAFE'])
            OBSERVATIONS.append({'case':'active <= row','upper':'-4/3','column_status':r['column_status'],
                'row_status':r['row_status'],'isolated':True,'real_LP':False})
            with self.assertRaises(FileExistsError):capture(lp,s,base/'native',deadline=time.monotonic()+10)
            for change in ('hash','presolve','scale','matrix','dimension'):
                bad=deepcopy(r);expected=identity(bad)
                if change=='hash':bad['column_values'][0]=0
                elif change=='presolve':bad['options']['presolve']='on';expected=identity(bad)
                elif change=='scale':bad['options']['simplex_scale_strategy']=2;expected=identity(bad)
                elif change=='matrix':bad['readback_after']['offset']=7;expected=identity(bad)
                else:bad['column_status'].pop();expected=identity(bad)
                with self.assertRaises(ValueError):map_capture(lp,s,bad,expected)

    def test_actual_inactive_slack_and_bound_anchor(self):
        lp,s,_,_=case()
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            r=capture(lp,s,Path(tmp)/'native',deadline=time.monotonic()+10)
            m,p=construct(lp,s,r);self.assertEqual(m['status'],'MAPPED_HINT_ONLY',r)
            self.assertEqual(check(p['bundle'],identity(s))['primal_status'],'EXACT_FEASIBLE')
            self.assertIn({'kind':'slack','index':0},m['hint']['basic_columns'])
            self.assertEqual(p['slacks'],['1'])
            OBSERVATIONS.append({'case':'inactive <= row','point':p['bundle']['primal']['x'],
                'column_status':r['column_status'],'row_status':r['row_status']})

    def test_actual_fixed_column(self):
        lp,s,_,_=case();lp['lower'][1]=lp['upper'][1]=1;lp['c']=[-1,0];s=statement(lp)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            r=capture(lp,s,Path(tmp)/'native',deadline=time.monotonic()+10)
            m,p=construct(lp,s,r);self.assertEqual(m['status'],'MAPPED_HINT_ONLY',r)
            self.assertEqual(p['bundle']['primal']['x'],['0','1'])
            self.assertEqual(check(p['bundle'],identity(s))['primal_status'],'EXACT_FEASIBLE')

    def test_actual_column_and_row_permutations(self):
        # Same geometry with structural columns exchanged and two A rows reversed.
        lp={'matrix_format':'csr_v1','c':[0,-1],'offset':-1,'lower':[0,0],'upper':[1,1],
            'E':csr([[1,1]],2),'h':[1],'A':csr([[1,0],[0,3]],2),'b':[1,1]}
        s=statement(lp)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            r=capture(lp,s,Path(tmp)/'native',deadline=time.monotonic()+10)
            m,p=construct(lp,s,r);self.assertEqual(m['status'],'MAPPED_HINT_ONLY',r)
            self.assertEqual(p['bundle']['primal']['x'],['2/3','1/3'])
            self.assertEqual(check(p['bundle'],identity(s))['upper_bound'],'-4/3')

    def test_redundant_equalities_preserve_unsupported_mapping(self):
        lp={'matrix_format':'csr_v1','c':[-1,0],'offset':0,'lower':[0,0],'upper':[1,1],
            'E':csr([[1,1],[2,2]],2),'h':[1,2],'A':csr([],2),'b':[]};s=statement(lp)
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp)/'native';r=capture(lp,s,root,deadline=time.monotonic()+10)
            m,p=construct(lp,s,r);self.assertEqual(m['status'],'UNSUPPORTED_MAPPING',r)
            self.assertIsNone(p);self.assertTrue((root/'capture.json').exists())
            OBSERVATIONS.append({'case':'redundant equalities','status':m['status'],'reason':m['reason'],
                'column_status':r['column_status'],'row_status':r['row_status']})

    def test_unknown_native_status_not_guessed(self):
        lp,s,_,_=case()
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            r=capture(lp,s,Path(tmp)/'native',deadline=time.monotonic()+10)
            for field,index,value in [('column_status',0,'kZero'),('row_status',1,'kLower'),('row_status',0,'kBasic')]:
                bad=deepcopy(r);bad[field][index]=value
                self.assertEqual(map_capture(lp,s,bad,identity(bad))['status'],'UNSUPPORTED_MAPPING')
            bad=deepcopy(r);bad['basis_valid']=False
            self.assertEqual(map_capture(lp,s,bad,identity(bad))['status'],'UNSUPPORTED_MAPPING')

    def test_expired_and_oversize_no_native_capture(self):
        lp,s,_,_=case()
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            root=Path(tmp)/'native'
            with self.assertRaises(TimeoutError):capture(lp,s,root,deadline=time.monotonic()-1)
            self.assertFalse(root.exists())
            lp['c']=[0]*65
            with self.assertRaises(ValueError):capture(lp,s,root,deadline=time.monotonic()+10)
            self.assertFalse(root.exists())

if __name__=='__main__':unittest.main()
