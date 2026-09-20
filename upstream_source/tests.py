"""Exact source controls and native local-transfer differentials, no real jobs."""
import copy
from fractions import Fraction as F
import json
from pathlib import Path
import tempfile
import unittest
import shutil
import subprocess
import sys

from upstream_source.checker import input_cover,affine_error,conv_operator,pair_guards,csr


def matrix(rows,n):
    data=[];indices=[];ptr=[0]
    for row in rows:
        for i,v in sorted(row.items()):indices.append(i);data.append(v)
        ptr.append(len(data))
    return {'shape':[len(rows),n],'data':data,'indices':indices,'indptr':ptr}


def source(centers,rows,n,guards=(),rhs=(),nb=0):
    count=len(centers)
    return {'c':centers,'Gc':matrix(rows,n),'Gb':matrix([{}]*count,nb),
        'Ac':matrix([],n),'Ab':matrix([],nb),'b':[],
        'Auc':matrix(guards,n),'Aub':matrix([{}]*len(rhs),nb),'ub':list(rhs),
        'frame_id':17,'exact':True}


class ExactControls(unittest.TestCase):
    def test_box_cover_and_missing_small_radius(self):
        s=source([0],[{0:1}],1)
        self.assertEqual(input_cover([-1],[1],s)['inward_coordinates'],0)
        s['Gc']['data'][0]=.5
        self.assertEqual(input_cover([-1],[1],s)['maximum_inward_gap'],'1/2')
        point=source([0],[{}],0)
        self.assertEqual(input_cover([0],[1e-13],point)['inward_coordinates'],1)
        with self.assertRaises(ValueError):input_cover([1],[-1],point)
        correlated=source([0,0],[{0:1},{0:1}],1)
        with self.assertRaises(ValueError):input_cover([-1,-1],[1,1],correlated)

    def test_exact_affine_and_rounding_gap_not_pass(self):
        s=source([.5],[{0:.25}],1)
        t=source([2],[{0:.5}],1)
        self.assertTrue(affine_error(s,t,[{0:2}],[1])['target_containment_established'])
        s=source([.1],[{0:.1}],1);t=source([.1*.1],[{0:.1*.1}],1)
        r=affine_error(s,t,[{0:.1}],[0])
        self.assertFalse(r['target_containment_established'])
        self.assertGreater(F(r['maximum_same_factor_error']),0)
        t['frame_id']=18
        with self.assertRaises(ValueError):affine_error(s,t,[{0:.1}],[0])

    def test_binary_factors_constraints_and_wrong_columns(self):
        s=source([0],[{0:.5}],2,guards=[{1:1}],rhs=[1],nb=1)
        s['Gb']=matrix([{0:2}],1);t=copy.deepcopy(s)
        self.assertTrue(affine_error(s,t,[{0:1}],[0])['target_containment_established'])
        t['Gb']['data'][0]=1
        self.assertEqual(affine_error(s,t,[{0:1}],[0])['maximum_same_factor_error'],'1')
        t=copy.deepcopy(s);t['ub']=[0]
        with self.assertRaises(ValueError):affine_error(s,t,[{0:1}],[0])
        t=copy.deepcopy(s);t['Gc']['indices']=[1]
        self.assertEqual(affine_error(s,t,[{0:1}],[0])['maximum_same_factor_error'],'1')

    def test_actual_guards_ties_prefix_and_nonredundant(self):
        r=source([2,1,0],[{0:.1},{0:.1},{0:.1}],1,guards=[{},{}],rhs=[2,1])
        j=source([0,0],[{},{}],2,guards=[{},{}],rhs=[2,1],nb=1)
        out=pair_guards(r,j,[0,1]);self.assertTrue(out['all_factor_assignments_retained'])
        r['ub']=[0,0];j['ub']=[0,0] # ties are included, not strictly excluded
        self.assertTrue(pair_guards(r,j,[0,1])['all_factor_assignments_retained'])
        r['Auc']=matrix([{0:1},{}],1);j['Auc']=matrix([{0:1},{}],2)
        self.assertFalse(pair_guards(r,j,[0,1])['all_factor_assignments_retained'])
        j['Auc']['indices'][0]=1
        with self.assertRaises(ValueError):pair_guards(r,j,[0,1])
        with self.assertRaises(ValueError):pair_guards(r,j,[1,0])

    def test_sparse_malformed_rejected(self):
        m=matrix([{0:1}],1)
        for edit in ('duplicate','badptr','nonfinite','column'):
            x=copy.deepcopy(m)
            if edit=='duplicate':x.update(data=[1,2],indices=[0,0],indptr=[0,2])
            if edit=='badptr':x['indptr']=[1,1]
            if edit=='nonfinite':x['data']=[float('nan')]
            if edit=='column':x['indices']=[1]
            with self.subTest(edit=edit),self.assertRaises(ValueError):csr(x)

    def test_conv_semantics_group_padding_and_dilation(self):
        g={'stride':[1,1],'padding':[0,0],'dilation':[1,1],'groups':2,
           'padding_mode':'zeros','training':False}
        op,b=conv_operator([1,2,2,2],g,[2,3],[2,1,1,1],[1,-1])
        self.assertEqual(op,[{i:F(2 if i<4 else 3)} for i in range(8)])
        self.assertEqual(b,[1]*4+[-1]*4)
        g.update(groups=1,padding=[1,1],stride=[2,2],dilation=[2,2])
        op,b=conv_operator([1,1,3,3],g,[1,2,3,4],[1,1,2,2],[0])
        self.assertEqual(op,[{4:F(4)},{4:F(3)},{4:F(2)},{4:F(1)}])


class NativeControls(unittest.TestCase):
    def test_native_box_small_radius_is_detected(self):
        import torch
        from act.back_end.core import Bounds
        from act.back_end.solver.solver_hz import sparse_hz_from_bounds
        from act.back_end.solver.hz_lp_export import snapshot
        h=snapshot(sparse_hz_from_bounds(Bounds(torch.tensor([0.],dtype=torch.float64),
                                               torch.tensor([1e-13],dtype=torch.float64)),frame_id=17))
        self.assertEqual(input_cover([0.],[1e-13],h)['inward_coordinates'],1)

    def test_capture_parameter_bound_small_model(self):
        from dataclasses import asdict
        import torch
        from act.back_end.moe.conv_factory import ConvOutputMoEConfig,build_conv_output_moe
        from act.pipeline.moe.staged_verifier import _model_state_identity,_tensor_identity
        from router_source.capture import capture as router_capture,sha
        from router_source.checker import tensor
        from upstream_source.build import capture
        config=ConvOutputMoEConfig(input_shape=(1,8,8),num_classes=2,num_experts=3,
                                  channels=(1,1),hidden=2,router_pool=2)
        model=build_conv_output_moe(config)
        with torch.no_grad():
            model.router[2].weight.zero_();model.router[2].bias.copy_(torch.tensor([3.,2.,0.]))
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as tmp:
            p=Path(tmp);ck=p/'m.pt';inp=p/'x.pt'
            torch.save({'format':'act-output-conv-moe-v1','factory_config':asdict(config),'state_dict':model.state_dict()},ck)
            model=model.double().eval();c=torch.full((1,1,8,8),.5,dtype=torch.float64)
            values={'center':c,'lower':c-.125,'upper':c+.125};torch.save(values,inp)
            request={'experts':3,'classes':2,'top_k':2,'tie_policy':'ANY_LEGAL_TOPK',
                     'model_state':_model_state_identity(model),**{k:_tensor_identity(v) for k,v in values.items()}}
            job={'parent_request':{'subject':{'checkpoint':str(ck),'checkpoint_sha256':sha(ck)},
                                   'tensors':{'path':str(inp),'sha256':sha(inp)}}}
            doc=router_capture(job,{'request':request});entry,experts=capture(job,doc,[0,1])
            self.assertEqual(len(experts),2)
            self.assertEqual(input_cover([.375]*64,[.625]*64,entry)['inward_coordinates'],0)
            for item in experts:
                wi,w=tensor(item['weight']);_,b=tensor(item['bias'])
                op,b=conv_operator([1,1,8,8],item['graph'],w,wi['shape'],b)
                result=affine_error(entry,item['output'],op,b)
                # Float32 source weights and dyadic test inputs give exact first transfer.
                self.assertTrue(result['target_containment_established'])
            bad=copy.deepcopy(doc);bad['request']['model_state']['sha256']='0'*64
            with self.assertRaises(ValueError):capture(job,bad,[0,1])
            self._portable_controls(p,doc,entry,experts)

    def _portable_controls(self,p,doc,entry,experts):
        from router_source.capture import ROOT,sha
        from router_source.checker import compact
        from router_source.propose import propose
        from router_source.build import save
        directory=p/'moved';directory.mkdir()
        save(directory/'router_source.json',doc);save(directory/'router_proof.json',propose(doc))
        r=source([3,2,0],[{}, {}, {}],1,guards=[{},{}],rhs=[3,2])
        j=source([0]*4,[{}]*4,2,guards=[{},{}],rhs=[3,2])
        save(directory/'router_hz.json',r);save(directory/'joint_hz.json',j)
        save(directory/'input_hz.json',entry)
        bindings=[]
        for original in experts:
            item=copy.deepcopy(original);out=item.pop('output');name=f"expert{item['expert']}_conv0.json"
            save(directory/name,out);item['output_file']=name;bindings.append(item)
        save(directory/'experts.json',bindings)
        for src,dst in [('router_source/checker.py','router_check.py'),('upstream_source/checker.py','local_check.py'),
                        ('upstream_source/verify.py','verify_upstream.py')]:shutil.copyfile(ROOT/src,directory/dst)
        manifest={'schema':'UPSTREAM_LOCAL_AUDIT_V1','request':doc['request'],'pair':[0,1],
                  'files':{v.name:sha(v) for v in directory.iterdir()},
                  'historical_source_sha256':{n:sha(directory/n) for n in ('router_hz.json','joint_hz.json')}}
        save(directory/'manifest.json',manifest)
        def call(path):
            return subprocess.run([sys.executable,'-I','-S',str(path/'verify_upstream.py'),
                '--manifest-hash',sha(path/'manifest.json')],capture_output=True,text=True,timeout=10,cwd=path)
        result=call(directory);self.assertEqual(result.returncode,0,result.stderr)
        self.assertFalse(json.loads(result.stdout)['complete_strict_network_certificate'])
        # Rebind transport hashes, retaining the anchored request/state. These
        # must fail semantic checking, not merely checksum comparison.
        for name in ('missing_expert','wrong_expert_parameter','wrong_frame'):
            target=p/name;shutil.copytree(directory,target);m=copy.deepcopy(manifest)
            file='input_hz.json' if name=='wrong_frame' else 'experts.json'
            obj=json.loads((target/file).read_bytes())
            if name=='missing_expert':obj.pop()
            if name=='wrong_expert_parameter':obj[0]['weight_name']='experts.2.0.weight'
            if name=='wrong_frame':obj['frame_id']+=1
            (target/file).write_bytes(compact(obj));m['files'][file]=sha(target/file)
            (target/'manifest.json').write_bytes(compact(m))
            with self.subTest(mutation=name):self.assertNotEqual(call(target).returncode,0)


if __name__=='__main__':unittest.main()
