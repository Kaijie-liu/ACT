"""Fresh relocated composition, semantic tampering, deadline/partial controls."""
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest

from router_source.capture import sha
from router_source.build import save
from router_source.tests import fixture,encode
from router_source.checker import tensor,compact
from router_source.propose import propose
from upstream_source.tests import source
from source_enclosure.build import build
from checked_gate.candidate_run import execute


def make_parent(root):
    old=root/'old';old.mkdir();d=fixture([[0.],[0.],[0.]],[3.,2.,0.]);items=[]
    for i in (0,1):
        weight=encode([.1],[1,1,1,1]);bias=encode([-.05],[1])
        wn,bn=f'experts.{i}.0.weight',f'experts.{i}.0.bias'
        d['state_inventory'] += [{'name':wn,**tensor(weight)[0]},{'name':bn,**tensor(bias)[0]}]
        items.append({'expert':i,'layer_index':0,'weight_name':wn,'bias_name':bn,'weight':weight,'bias':bias,
            'topology_inspected':['Conv2d','ReLU'],'output_file':f'expert{i}_conv0.json',
            'graph':{'stride':[1,1],'padding':[0,0],'dilation':[1,1],'groups':1,'padding_mode':'zeros','training':False}})
        save(old/f'expert{i}_conv0.json',source([.01]*4,[{j:.05} for j in range(4)],4))
    d['state_inventory'].sort(key=lambda v:v['name']);h=hashlib.sha256()
    for v in d['state_inventory']:h.update(v['name'].encode());h.update(v['sha256'].encode())
    d['request']['model_state'].update(sha256=h.hexdigest(),tensor_count=6,parameter_count=10)
    save(old/'router_source.json',d);save(old/'router_proof.json',propose(d));save(old/'experts.json',items)
    save(old/'input_hz.json',source([.5]*4,[{j:.5} for j in range(4)],4))
    save(old/'router_hz.json',source([3,2,0],[{}, {}, {}],4,guards=[{},{}],rhs=[3,2]))
    save(old/'manifest.json',{'request':d['request'],'pair':[0,1],'files':{p.name:sha(p) for p in old.iterdir()}})
    return old


def rebind(root,file,obj):
    (root/file).write_bytes(compact(obj));m=json.loads((root/'manifest.json').read_bytes())
    m['files'][file]=sha(root/file);(root/'manifest.json').write_bytes(compact(m))


class PortableControls(unittest.TestCase):
    def test_relocated_complete_trace_and_mutations(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);old=make_parent(root);built=root/'built';built.mkdir();build(built,old,sha(old/'manifest.json'))
            moved=root/'moved';shutil.copytree(built/'relocated',moved)
            shutil.rmtree(old);shutil.rmtree(built) # Only test-owned sources, deliberately unavailable.
            def run(directory,log,seconds=10):
                return execute([sys.executable,'-I','-S',str(directory/'verify_prefix.py'),
                    '--manifest-hash',sha(directory/'manifest.json')],root/log,time.monotonic()+seconds,
                    dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            result=run(moved,'checked.log');self.assertEqual(result['state'],'COMPLETED',(root/'checked.log').read_text())
            report=json.loads((root/'checked.log').read_bytes());self.assertEqual(report['checked_steps'],7)
            self.assertFalse(report['complete_strict_network_certificate'])
            for mode in ('missing_step','error_bound','relu_range','join_map','old_certificate','parameter'):
                dst=root/mode;shutil.copytree(moved,dst)
                if mode=='parameter':
                    file='experts.json';obj=json.loads((dst/file).read_bytes());obj[0]['weight_name']='experts.1.0.weight'
                else:
                    file='trace.json';obj=json.loads((dst/file).read_bytes())
                    if mode=='missing_step':obj['states'].pop('expert0_relu')
                    if mode=='error_bound':obj['certificates']['expert0_affine']['error_bounds'][0]='0'
                    if mode=='relu_range':obj['certificates']['expert0_relu']['ranges'][0]=['0','1']
                    if mode=='join_map':obj['certificates']['join']['maps']['right_c'][-1]=0
                    if mode=='old_certificate':obj['certificates']['expert0_affine']['source']='0'*64
                rebind(dst,file,obj)
                with self.subTest(mode=mode):self.assertEqual(run(dst,mode+'.log')['state'],'ERROR')
            # No partial/late prefix is accepted after the parent's budget.
            result=run(moved,'expired.log',seconds=-.1)
            self.assertEqual(result['state'],'TIMEOUT');self.assertFalse(result['started'])


if __name__=='__main__':unittest.main()
