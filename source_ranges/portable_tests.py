"""Moved single-step controls; original directories and producers unavailable."""
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

from source_ranges.tests import mixed_control,constrained,interval_fact,CONTEXT
from source_ranges.produce import affine
from source_enclosure.format import compact,sparse
from router_source.capture import ROOT,sha
from router_source.build import save
from checked_gate.candidate_run import execute

CODE={'source_ranges/check.py':'range_check.py','source_ranges/verify.py':'verify_range.py',
      'source_enclosure/format.py':'proof_format.py','upstream_source/checker.py':'local_check.py',
      'act/back_end/solver/lp_certificate.py':'act/back_end/solver/lp_certificate.py',
      'act/back_end/solver/sparse_lp_certificate.py':'act/back_end/solver/sparse_lp_certificate.py'}


def build(root,kind):
    root.mkdir()
    if kind=='relu':
        s,t,p=mixed_control();step={'kind':'relu','context':CONTEXT,'tag':'mixed'}
    else:
        s=constrained();fact=interval_fact(s,{0:1},0,F(1,4),F(3,4))
        t,p=affine(s,[{0:1}],[0],[fact],CONTEXT,'shift')
        step={'kind':'affine','context':CONTEXT,'tag':'shift','operator':sparse([{0:F(1)}],1),'bias':['0']}
    for name,v in [('source.json',s),('target.json',t),('proof.json',p)]:save(root/name,v)
    for a,b in CODE.items():
        dest=root/b;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(ROOT/a,dest)
    files=['source.json','target.json','proof.json']+list(CODE.values())
    save(root/'manifest.json',{'schema':'SCOPED_RANGE_STEP_BUNDLE_V1','step':step,'files':{n:sha(root/n) for n in files}})


class PortableControls(unittest.TestCase):
    def test_moved_source_steps_mutations_and_deadline(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as temp:
            root=Path(temp)
            def run(path,name,limit=15):
                return execute([sys.executable,'-I','-S',str(path/'verify_range.py'),'--manifest-hash',sha(path/'manifest.json')],
                    root/(name+'.log'),time.monotonic()+limit,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            for kind in ('affine','relu'):
                original=root/(kind+'_original');build(original,kind)
                moved=root/kind;shutil.copytree(original,moved);shutil.rmtree(original) # test-owned only
                r=run(moved,kind);self.assertEqual(r['state'],'COMPLETED',(root/(kind+'.log')).read_text())
                result=json.loads((root/(kind+'.log')).read_bytes())
                self.assertFalse(result['complete_network_certificate'])
                if kind=='relu':self.assertEqual(result['new_binary_factors'],1)
            for mode in ('missing','context','old_source','bound','dual','rhs'):
                target=root/mode;shutil.copytree(root/'affine',target)
                m=json.loads((target/'manifest.json').read_bytes())
                name='target.json' if mode=='rhs' else 'proof.json';p=json.loads((target/name).read_bytes())
                if mode=='missing':p['facts']=[]
                if mode=='context':p['facts'][0]['query']['context']['scope']='other-pair'
                if mode=='old_source':p['facts'][0]['query']['source_sha256']='old'
                if mode=='bound':p['facts'][0]['range'][0]='1/2'
                if mode=='dual':p['facts'][0]['lower']['inequality_dual']=[1,0]
                if mode=='rhs':p['hz']['b'][0]='0'
                (target/name).write_bytes(compact(p));m['files'][name]=sha(target/name)
                (target/'manifest.json').write_bytes(compact(m))
                with self.subTest(mode=mode):self.assertEqual(run(target,mode)['state'],'ERROR')
            expired=run(root/'affine','expired',-.1)
            self.assertEqual(expired['state'],'TIMEOUT');self.assertFalse(expired['started'])


if __name__=='__main__':unittest.main()
