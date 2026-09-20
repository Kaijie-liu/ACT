"""Controls before any real LP call: positivity, completeness, exactness, deadlines."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from act.back_end.solver.lp_certificate import identity,check as old_check
from full_bounds.check import program,check_one,aggregate
from full_bounds.worker import prepare,propose,seal,publish
from full_bounds.run import supervise
from full_source.tests import fixture
from full_source.build import build
from full_source.obligations import materialize
from router_source.capture import sha
from router_source.checker import compact
from checked_gate.candidate_run import execute


def lp():
    empty={'shape':[0,1],'data':[],'indices':[],'indptr':[0]}
    return {'matrix_format':'csr_v1','c':['1'],'offset':'2','lower':['-1'],'upper':['1'],
            'A':copy.deepcopy(empty),'b':[],'E':copy.deepcopy(empty),'h':[]}


class BoundControls(unittest.TestCase):
    def test_exact_residual_matches_existing_checker(self):
        p=lp();p['E']={'shape':[1,1],'data':['3/2'],'indices':[0],'indptr':[0,1]};p['h']=['1/3']
        c={'lp_sha256':identity(p),'inequality_dual':[],'equality_dual':[.7]}
        result=check_one(p,c);v=F(result['checked_lower_bound'])
        self.assertEqual(v,2+F(.7)*F(1,3)-abs(1-F(.7)*F(3,2)))
        self.assertEqual(v,F(old_check(p,{**c,'claimed_lower_bound':str(v)})['checked_lower_bound']))
        self.assertEqual(v,F(result['dual_constant'])+F(result['residual_box_correction']))

    def test_negative_zero_and_mutated_duals(self):
        p=lp();c={'lp_sha256':identity(p),'inequality_dual':[],'equality_dual':[]}
        self.assertEqual(check_one(p,c)['status'],'CHECKED_POSITIVE_BOUND')
        for offset in ('1','0'):
            p['offset']=offset;c['lp_sha256']=identity(p)
            self.assertEqual(check_one(p,c)['status'],'CHECKED_NONPOSITIVE_BOUND')
        bad=copy.deepcopy(c);bad['lp_sha256']='0'*64
        with self.assertRaises(ValueError):check_one(p,bad)
        bad=copy.deepcopy(c);bad['claimed_lower_bound']='999'
        with self.assertRaises(ValueError):check_one(p,bad)
        p['A']={'shape':[1,1],'data':['1'],'indices':[0],'indptr':[0,1]};p['b']=['1']
        c={'lp_sha256':identity(p),'inequality_dual':[1.],'equality_dual':[]}
        with self.assertRaises(ValueError):check_one(p,c)
        c['inequality_dual']=[float('nan')]
        with self.assertRaises(ValueError):check_one(p,c)

    def test_supervisor_partial_and_exception(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            for mode in ('limit','exception','malformed'):
                dst=root/mode;dst.mkdir()
                def cmd(phase):
                    if phase=='propose' and mode=='exception':raise ValueError('injected exception')
                    script='print("partial")'
                    if phase=='propose' and mode=='limit':script='import time; print("partial",flush=True); time.sleep(1)'
                    return [sys.executable,'-I','-S','-c',script]
                result=supervise(dst,cmd,dict(os.environ),budget=2,propose_cap=.05)
                self.assertIn(result['status'],('ERROR','TIMEOUT'));self.assertFalse(result['complete_declared_real_output_proof'])
                self.assertTrue((dst/'publication.json').exists())
                if mode=='limit':self.assertEqual(result['stages'][1]['state'],'TIMEOUT')

    def test_atomic_publication_no_overwrite(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p=Path(tmp)/'candidate.json';publish(p,{'v':1});self.assertEqual(json.loads(p.read_text()),{'v':1})
            with self.assertRaises(FileExistsError):publish(p,{'v':2})
            self.assertEqual(json.loads(p.read_text()),{'v':1})
            self.assertEqual(json.loads(p.with_name(p.name+'.partial').read_text()),{'v':2})

    def test_full_moved_positive_and_semantic_rejections(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);prefix,doc=fixture(root);source=root/'full';source.mkdir()
            build(source,prefix,sha(prefix/'manifest.json'),doc)
            dest=root/'bound';dest.mkdir();prepare(dest,source/'relocated',sha(source/'relocated/manifest.json'))
            propose(dest,native_seconds=2.);seal(dest)
            moved=root/'moved';shutil.copytree(dest/'relocated',moved)
            # All source directories here are temporary test-owned fixtures.
            for path in (root/'old',root/'pb',source,dest):shutil.rmtree(path)
            def run(where,name,seconds=15):
                return execute([sys.executable,'-I','-S',str(where/'verify_bounds.py'),'--manifest-hash',sha(where/'manifest.json')],
                    root/(name+'.log'),time.monotonic()+seconds,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            row=run(moved,'good');self.assertEqual(row['state'],'COMPLETED',(root/'good.log').read_text())
            result=json.loads((root/'good.log').read_bytes());self.assertTrue(result['complete_declared_real_output_proof'])
            self.assertFalse(result['deployed_floating_point_proof']);self.assertEqual(result['positive_bounds'],1)
            base=json.loads((moved/'source/lp_base.json').read_bytes());obs=json.loads((moved/'source/obligations.json').read_bytes())
            self.assertEqual(program(base,obs['rows'][0]),materialize(base,obs['rows'][0]))
            for mode in ('missing','duplicate','wrong_lp','wrong_request','wrong_property','sign','partial'):
                target=root/mode;shutil.copytree(moved,target);m=json.loads((target/'manifest.json').read_bytes())
                if mode=='missing':m['outcomes']=[]
                elif mode=='duplicate':m['outcomes']*=2
                elif mode=='partial':m['outcomes'][0]['file']=None;m['files'].pop('candidate_1.json')
                else:
                    name='candidate_1.json';r=json.loads((target/name).read_bytes())
                    if mode=='wrong_lp':r['candidate']['lp_sha256']='0'*64
                    if mode=='wrong_request':r['request_id']='0'*64
                    if mode=='wrong_property':r['competitor']=0
                    if mode=='sign':r['candidate']['inequality_dual'][0]=1.
                    (target/name).write_bytes(compact(r));m['files'][name]=sha(target/name)
                (target/'manifest.json').write_bytes(compact(m));checked=run(target,mode)
                if mode=='partial':
                    self.assertEqual(checked['state'],'COMPLETED');partial=json.loads((root/(mode+'.log')).read_bytes())
                    self.assertEqual(partial['status'],'UNKNOWN_MISSING_BOUND_EVIDENCE');self.assertEqual(partial['missing_bounds'],1)
                else:
                    with self.subTest(mode=mode):self.assertEqual(checked['state'],'ERROR')
            expired=run(moved,'expired',-.1);self.assertEqual(expired['state'],'TIMEOUT');self.assertFalse(expired['started'])


if __name__=='__main__':unittest.main()
