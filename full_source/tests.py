"""Analytic extension, graph semantics, transport and full-inventory controls."""
import copy
from fractions import Fraction as F
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from router_source.tests import encode as tensor_bytes
from router_source.checker import tensor,compact
from router_source.propose import propose
from router_source.capture import sha
from router_source.build import save
from source_enclosure.portable_tests import make_parent
from source_enclosure.build import build as prefix_build
from source_enclosure.format import unpack,identity
from source_enclosure.produce import box,relu,join
from checked_gate.candidate_run import execute
from full_source.lift import affine
from full_source.check_lift import check
from full_source.graph import operator,KINDS
from full_source.delta import encode,restore
from full_source.obligations import build as output_build,materialize
from full_source.check_obligations import check as output_check
from full_source.build import build


def fixture(root):
    old=make_parent(root);source=json.loads((old/'router_source.json').read_bytes());source['request']['clean_prediction']=0
    pe=json.loads((old/'experts.json').read_bytes());experts=[]
    for p in pe:
        i=p['expert'];p['topology_inspected']=KINDS;layers=[]
        for j,kind in enumerate(KINDS):
            r={'index':j,'kind':kind,'training':False}
            if kind=='Conv2d':
                r.update(graph=copy.deepcopy(p['graph']),weight=tensor_bytes([.1],[1,1,1,1]),bias=tensor_bytes([-.05],[1]))
                if j==0:r.update(weight=p['weight'],bias=p['bias'])
            if kind=='Linear':
                r.update(weight=tensor_bytes([1.,-1.] if j==6 else [1.,0.,0.,1.],[2,1] if j==6 else [2,2]),
                         bias=tensor_bytes([1.,0.],[2]))
            if kind in ('Conv2d','Linear'):
                for role in ('weight','bias'):
                    name=f'experts.{i}.{j}.{role}';r[role+'_name']=name
                    if j:source['state_inventory'].append({'name':name,**tensor(r[role])[0]})
            if kind=='ReLU':r['inplace']=False
            if kind=='AvgPool2d':r.update(kernel=[2,2],stride=[2,2],padding=[0,0],ceil_mode=False,count_include_pad=True,divisor_override=None)
            if kind=='Flatten':r['dimensions']=[1,-1]
            layers.append(r)
        experts.append({'expert':i,'layers':layers})
    source['state_inventory'].sort(key=lambda v:v['name']);h=hashlib.sha256();count=0
    import math
    for v in source['state_inventory']:h.update(v['name'].encode());h.update(v['sha256'].encode());count+=math.prod(v['shape'])
    source['request']['model_state'].update(sha256=h.hexdigest(),tensor_count=len(source['state_inventory']),parameter_count=count)
    # Mutating only these test-owned synthetic files, before any prefix construction.
    (old/'router_source.json').write_bytes(compact(source));(old/'router_proof.json').write_bytes(compact(propose(source)))
    (old/'experts.json').write_bytes(compact(pe))
    m=json.loads((old/'manifest.json').read_bytes());m['request']=source['request']
    m['files']={p.name:sha(p) for p in old.iterdir() if p.name!='manifest.json'};(old/'manifest.json').write_bytes(compact(m))
    pb=root/'pb';pb.mkdir();prefix_build(pb,old,sha(old/'manifest.json'))
    return pb/'relocated',{'schema':'DECLARED_FULL_EXPERTS_V1','request':source['request'],'pair':[0,1],'experts':experts}


def rebind(root,name,obj):
    (root/name).write_bytes(compact(obj));m=json.loads((root/'manifest.json').read_bytes());m['files'][name]=sha(root/name)
    (root/'manifest.json').write_bytes(compact(m))


class LiftControls(unittest.TestCase):
    def test_supervisor_partial_deadline_and_exception(self):
        from full_source.run import supervise
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp)
            for mode in ('deadline','exception','partial'):
                dest=root/mode;dest.mkdir()
                def command(phase):
                    if mode=='exception':raise ValueError('injected callback exception')
                    code='import time; print("partial", flush=True); time.sleep(2)' if mode=='deadline' else 'print("partial")'
                    return [sys.executable,'-I','-S','-c',code]
                result=supervise(dest,command,dict(os.environ),budget=.2 if mode=='deadline' else 2)
                self.assertIn(result['status'],('ERROR','TIMEOUT'));self.assertIsNone(result['check'])
                self.assertFalse(result['complete_output_positive_proof'])
                self.assertTrue((dest/'terminal.json').is_file());self.assertTrue((dest/'publication.json').is_file())

    def test_exact_extension_and_constant_rows(self):
        s=box(['-2','1'],['3','1']);op=[{0:F(2),1:F(-1)},{}];bias=[F(1,3),F(7)]
        t,p=affine(s,op,bias,'a');self.assertEqual(check(s,t,op,bias,p,'a')['new_factors'],1)
        source,_,_=unpack(s);target,_,_=unpack(t)
        for x in [F(-1),F(0),F(1)]:
            assignment=[x,F(0),x]
            self.assertEqual(sum(v*assignment[k] for k,v in target['Ac'][-1].items()),target['b'][-1])
            self.assertEqual(target['c'][0]+sum(v*assignment[k] for k,v in target['Gc'][0].items()),
                             2*(source['c'][0]+source['Gc'][0][0]*x)-1+F(1,3))

    def test_binary_source_and_graph_tampering(self):
        s,_=relu(box([-1],[1]),'r');t,p=affine(s,[{0:F(-2)}],[F(0)],'a')
        check(s,t,[{0:F(-2)}],[F(0)],p,'a')
        mutations=[lambda t:t['hz']['b'].__setitem__(-1,'1'),
                   lambda t:t['hz']['Gc']['data'].__setitem__(0,'17'),
                   lambda t:t['continuous_ids'].__setitem__(-1,t['continuous_ids'][0])]
        for mut in mutations:
            bad=copy.deepcopy(t);mut(bad)
            with self.assertRaises(ValueError):check(s,bad,[{0:F(-2)}],[F(0)],p,'a')
        with patch('full_source.lift.affine',side_effect=AssertionError('producer called')):
            check(s,t,[{0:F(-2)}],[F(0)],p,'a')

    def test_delta_roundtrip_and_binding(self):
        s=box([-1],[2]);t,p=affine(s,[{0:F(2)}],[F(1)],'a');d=encode(s,t)
        self.assertEqual(restore(s,d),t)
        d['source']='0'*64
        with self.assertRaises(ValueError):restore(s,d)

    def test_pool_flatten_linear_exact_semantics(self):
        layer={'kind':'AvgPool2d','training':False,'kernel':[2,2],'stride':[2,2],'padding':[0,0],
               'ceil_mode':False,'count_include_pad':True,'divisor_override':None}
        shape,op,b=operator([1,2,2,2],layer)
        self.assertEqual(shape,[1,2,1,1]);self.assertEqual(op,[{i:F(1,4) for i in range(4)},{i:F(1,4) for i in range(4,8)}])
        self.assertEqual(operator(shape,{'kind':'Flatten','training':False,'dimensions':[1,-1]})[0],[1,2])
        layer['divisor_override']=3
        with self.assertRaises(ValueError):operator([1,2,2,2],layer)

    def test_all_obligations_and_exact_construction_differential(self):
        from act.back_end.solver.rational_mccormick import build as reference
        from act.back_end.solver.check_rational_mccormick import check_construction
        s=box([-1],[1]);a,_=affine(s,[{0:F(2)},{0:F(1)}],[F(1),F(0)],'a')
        b,_=affine(s,[{0:F(-1)},{0:F(2)}],[F(2),F(0)],'b');state,_=join(s,a,b)
        base,obs=output_build(state,[0,1],2,0);output_check(state,base,obs,[0,1],2,0)
        r=obs['rows'][0];lp=materialize(base,r)
        ref=reference(state['hz'],[1,-1],0,[0,1],r['difference'])
        from upstream_source.checker import csr
        for k in ('A','E'):self.assertEqual(csr(lp[k]),csr(ref['lp'][k]))
        for k in ('c','offset','b','h','lower','upper'):
            self.assertEqual(list(map(F,lp[k])) if type(lp[k])is list else F(lp[k]),
                             list(map(F,ref['lp'][k])) if type(ref['lp'][k])is list else F(ref['lp'][k]))
        check_construction(ref,source_hash=identity(state['hz']),q=[1,-1],offset=0,gate=[0,1],difference=r['difference'])
        for key in ('rows','source_sha256'):
            bad=copy.deepcopy(obs);bad[key]=[] if key=='rows' else '0'*64
            with self.assertRaises(ValueError):output_check(state,base,bad,[0,1],2,0)
        for action in ('sign','certificate','range'):
            bad=copy.deepcopy(obs)
            if action=='sign':bad['rows'][0]['A_extra']['data'][0]='17'
            if action=='certificate':bad['rows'][0]['lower_bound_certificate']={'old':'positive'}
            if action=='range':bad['rows'][0]['difference']=['0','0']
            with self.assertRaises(ValueError):output_check(state,base,bad,[0,1],2,0)

    def test_relocated_full_graph_and_semantic_mutations(self):
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            root=Path(tmp);prefix,doc=fixture(root);built=root/'full';built.mkdir()
            build(built,prefix,sha(prefix/'manifest.json'),doc)
            moved=root/'moved';shutil.copytree(built/'relocated',moved)
            shutil.rmtree(root/'old');shutil.rmtree(root/'pb');shutil.rmtree(built)
            def run(path,name,seconds=15):
                return execute([sys.executable,'-I','-S',str(path/'verify_full.py'),'--manifest-hash',sha(path/'manifest.json')],
                    root/(name+'.log'),time.monotonic()+seconds,dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
            good=run(moved,'good');self.assertEqual(good['state'],'COMPLETED',(root/'good.log').read_text())
            result=json.loads((root/'good.log').read_bytes());self.assertEqual(result['remaining_steps_checked'],16)
            self.assertEqual(result['outputs']['obligations'],1);self.assertFalse(result['complete_output_positive_proof'])
            for mode in ('layer','pool','parameter','shape','property','old_dual','wrong_frame'):
                dst=root/mode;shutil.copytree(moved,dst)
                name='trace.json' if mode in ('layer','shape','wrong_frame') else 'full_experts.json' if mode in ('pool','parameter') else 'obligations.json'
                obj=json.loads((dst/name).read_bytes())
                if mode=='layer':obj['steps'].pop()
                if mode=='shape':obj['steps'][0]['output_shape']=[1,1]
                if mode=='pool':obj['experts'][0]['layers'][4]['divisor_override']=3
                if mode=='parameter':obj['experts'][0]['layers'][6]['weight_name']='experts.1.6.weight'
                if mode=='property':obj['rows']=[]
                if mode=='old_dual':obj['rows'][0]['lower_bound_certificate']={'old':'positive'}
                if mode=='wrong_frame':obj['joint_proof']['maps']['right_c'][-1]=0
                rebind(dst,name,obj)
                with self.subTest(mode=mode):self.assertEqual(run(dst,mode)['state'],'ERROR')
            result=run(moved,'expired',-.1);self.assertEqual(result['state'],'TIMEOUT');self.assertFalse(result['started'])


if __name__=='__main__':unittest.main()
