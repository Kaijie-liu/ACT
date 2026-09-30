"""Finite representation controls, not timing/real-model or hard-budget runs."""
import base64
from copy import deepcopy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, save, sha
from scoped_source.factored_source import Source, pack, SOURCE_TOTAL_LIMIT, TENSOR_ELEMENTS
from scoped_source.factored_io import load, write, read, decode, digest, referenced, HEADER_LIMIT, BLOCK_LIMIT, MEMBER_LIMIT, TOTAL_LIMIT
from scoped_source.factored_build import build
from scoped_source.factored_check import check
from scoped_source.factored_ir import assemble
from scoped_source.endpoint_source_controls import cases, weighted_source
from scoped_source.endpoint_source_build import build as legacy_build, source_request, mc_lp
from scoped_source.endpoint_source_check import check as legacy_check
from scoped_source.sparse_ir import index
from scoped_source.sparse_controls import source as synthetic
from source_enclosure.format import identity, compact

PROTOCOL='configs/h2_factored_controls_20260930.json'
PROTOCOL_SHA='d3120813b9613c9e4430728a77416bb3e8bb7351ca445e0f2cc87ed34997822a'
FILES=('scoped_source/factored_io.py','scoped_source/factored_source.py','scoped_source/factored_ir.py',
       'scoped_source/factored_build.py','scoped_source/factored_check.py','scoped_source/factored_tests.py',
       'scripts/check_h2_factored.py',PROTOCOL,'configs/h2_source_controls_20260930.json',
       'scoped_source/endpoint_source_build.py','scoped_source/endpoint_source_check.py',
       'scoped_source/endpoint_source_controls.py','scoped_source/endpoint_check.py','scoped_source/endpoint_build.py',
       'scoped_source/sparse_ir.py','scoped_source/graph.py','scoped_source/sparse_check.py',
       'scoped_source/sparse_controls.py','scoped_source/endpoint_controls.py','router_source/checker.py',
       'upstream_source/checker.py','source_enclosure/format.py','scoped_proof/io.py',
       'act/back_end/solver/lp_certificate.py','act/back_end/solver/sparse_lp_certificate.py')
SAME=('status','required','positive','missing','lp_bounds_checked','source_blocks_checked','duties','scopes','origins')


def protocol():
    cfg=load(ROOT,PROTOCOL)
    if identity(cfg)!=PROTOCOL_SHA: raise ValueError('frozen factor protocol')
    if (cfg['maximum_source_chunk_bytes'],cfg['maximum_header_bytes'],cfg['maximum_member_bytes'],
        cfg['maximum_source_bytes'],cfg['maximum_package_bytes'],cfg['maximum_tensor_elements']) != \
       (BLOCK_LIMIT,HEADER_LIMIT,MEMBER_LIMIT,SOURCE_TOTAL_LIMIT,TOTAL_LIMIT,TENSOR_ELEMENTS):
        raise ValueError('resource contract changed')
    return cfg


def verify(root):
    m=load(root,'manifest.json')
    return check(root,expected_source_manifest=m['source_manifest_sha256'],expected_proof_manifest=identity(m),
                 expected_mode=m['mode'],deadline=time.monotonic()+300)


class FactoredControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg=protocol(); keep=os.environ.get('H2_FACTORED_ROOT')
        if keep: cls.root=Path(keep); cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp=tempfile.TemporaryDirectory(prefix='h2-factored-',dir=ROOT/'data/moe/tmp'); cls.root=Path(cls.tmp.name)
        bindings={}
        for name in FILES:
            out=cls.root/'implementation'/name; out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,out); bindings[name]=sha(out)
        save(cls.root/'implementation.json',bindings); save(cls.root/'protocol.json',cfg)
        from act.back_end.solver.lp_certificate import propose
        cls.propose=staticmethod(propose); cls.examples={}; records=[]
        source_cases=cases()
        if [v[0] for v in source_cases]!=cfg['cases']: raise ValueError('fixed case inventory')
        # Save all declarations before any proposal.
        for name,doc,reuse in source_cases: save(cls.root/(name+'-source.json'),doc)
        for name,doc,reuse in source_cases:
            cls.examples[name]=(doc,reuse,{})
            for mode in cfg['arms']:
                root=cls.root/(name+'-'+mode); root.mkdir()
                s=pack(doc,root/'source',lambda:None,cfg['control_chunk_bytes'])
                p=build(root,expected_source_manifest=s,mode=mode,reuse_keys=reuse,deadline=time.monotonic()+300,proposer=propose)
                old=legacy_build(doc,expected_source_sha256=identity(doc),mode=mode,reuse_keys=reuse,
                                 deadline=time.monotonic()+300,proposer=propose)
                save(cls.root/(name+'-'+mode+'-legacy.json'),old)
                expected=legacy_check(doc,old,expected_source_sha256=identity(doc),expected_mode=mode,deadline=time.monotonic()+300)
                actual=verify(root)
                if {k:actual[k] for k in SAME}!={k:expected[k] for k in SAME}: raise ValueError('legacy/factored semantic difference')
                cls.examples[name][2][mode]=(root,old,actual)
                records.append({'case':name,'mode':mode,'source_manifest_sha256':s,'proof_manifest_sha256':p,
                    'legacy_source_sha256':identity(doc),'legacy_package_sha256':identity(old),'result':actual,
                    'legacy_package_bytes':len(compact(old))})
        save(cls.root/'results.json',{'schema':'H2_FACTORED_CONTROL_RESULTS_V1','records':records,
            'real_requests':0,'hard_budget_supervision':False,'performance_claim':False})

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def clone(self,case='weighted_sign',mode='endpoints'):
        source=self.examples[case][2][mode][0]
        path=Path(tempfile.mkdtemp(prefix='mutation-',dir=self.root)); shutil.copytree(source,path,dirs_exist_ok=True)
        return path

    def rewrite(self,path,name,value):
        raw=compact(value); (path/name).write_bytes(raw)
        return {'file':name,'bytes':len(raw),'sha256':digest(raw)}

    def mutate_pair(self,path,change,index=0):
        m=load(path,'manifest.json'); ref=m['pairs'][index]; pair=load(path,ref['file']); change(pair)
        m['pairs'][index]={**self.rewrite(path,ref['file'],pair),'pair':ref['pair']}
        self.rewrite(path,'manifest.json',m)

    def test_whole_results_and_single_pair_assembly_counts(self):
        expected={'weighted_sign':(3,2,3),'tied_partial_reuse':(18,18,18),'unsafe_tied':(0,0,6),'unresolved_sign':(6,6,6)}
        for name,(doc,reuse,arms) in self.examples.items():
            for i,(mode,(path,old,result)) in enumerate(arms.items()):
                self.assertEqual((result['positive'],result['required']),(expected[name][i],expected[name][2]))
                self.assertEqual(result['pair_bases_reconstructed'],doc['request']['experts']*(doc['request']['experts']-1)//2)
                self.assertFalse(result['hard_budget_supervision']); self.assertFalse(result['deployed_float_SAFE'])
                self.assertEqual(load(path,'manifest.json')['proposal_errors'],[])

    def test_streamed_source_matches_every_old_node_and_identity(self):
        for doc,reuse,arms in self.examples.values():
            path=arms['endpoints'][0]; m=load(path,'manifest.json')
            view=Source(path/'source',m['source_manifest_sha256'],lambda:None)
            r,nodes,outputs=index(doc,identity(doc),lambda:None)
            self.assertEqual(dict(view.nodes()),nodes); self.assertEqual(view.outputs,outputs); self.assertEqual(view.request,r)
            self.assertNotEqual(view.identity,identity(doc))
            with self.assertRaises(ValueError): list(view.nodes())

    def test_every_expanded_base_scope_and_objective_matches(self):
        for doc,reuse,arms in self.examples.values():
            bank,request,scopes=source_request(doc,identity(doc),lambda:None)
            for mode,(path,old,result) in arms.items():
                m=load(path,'manifest.json'); refs={r['name']:r for r in m['blocks']}
                self.assertEqual({n:load(path,r['file']) for n,r in refs.items()},bank)
                grouped={tuple(p['pair']):load(path,p['file']) for p in m['pairs']}
                for duty in request['duties']:
                    pair=grouped[tuple(duty['pair'])]; context=pair['context']
                    base=assemble(path,refs,context['blocks'],context['guards'],lambda:None)
                    self.assertEqual(base,duty['base']); self.assertEqual(context['gate'],duty['gate'])
                    row=next(v for v in pair['duties'] if v['competitor']==duty['competitor'])
                    if mode=='mccormick': self.assertEqual(row['lp_sha256'],identity(mc_lp(duty)))
                    else:
                        for e in row['endpoints']:
                            t=F(e['weight']); lp={**base,'c':[str(t*F(a)+(1-t)*F(b)) for a,b in zip(duty['a']['c'],duty['b']['c'])],
                                'offset':str(t*F(duty['a']['offset'])+(1-t)*F(duty['b']['offset']))}
                            self.assertEqual(identity(lp),e['lp_sha256'])

    def test_no_full_request_builder_or_checker_used(self):
        doc=weighted_source(); path=self.root/'no-legacy-expansion'; path.mkdir()
        source_sha=pack(doc,path/'source',lambda:None)
        with (patch('scoped_source.endpoint_source_build.source_request',side_effect=AssertionError('full request')),
              patch('scoped_source.endpoint_source_check.reconstruct',side_effect=AssertionError('full request'))):
            build(path,expected_source_manifest=source_sha,mode='endpoints',deadline=time.monotonic()+300,proposer=self.propose)
            self.assertEqual(verify(path)['positive'],3)

    def test_missing_duplicate_reordered_pair_or_property_rejected(self):
        changes=[lambda p:p['duties'].clear(),lambda p:p['duties'].append(deepcopy(p['duties'][0])),
                 lambda p:p['duties'][0].update(competitor=0),lambda p:p['context']['blocks'].reverse(),
                 lambda p:p['context']['guards'].clear(),lambda p:p['context'].update(gate=['0','1'])]
        for change in changes:
            path=self.clone(); self.mutate_pair(path,change)
            with self.assertRaises(ValueError): verify(path)
        for change in (lambda m:m['pairs'].pop(),lambda m:m['pairs'].reverse(),lambda m:m['blocks'].pop()):
            path=self.clone(); m=load(path,'manifest.json'); change(m); self.rewrite(path,'manifest.json',m)
            with self.assertRaises(ValueError): verify(path)

    def test_missing_certificate_is_unknown_including_empty_guard(self):
        for index in (0,1):
            path=self.clone(); self.mutate_pair(path,lambda p:p['duties'][0]['endpoints'][0].update(certificate=None),index)
            r=verify(path); self.assertEqual(r['status'],'UNKNOWN_MISSING_EVIDENCE'); self.assertEqual(r['missing'],1)
        for change in (lambda p:p['duties'][0]['endpoints'].pop(),lambda p:p['duties'][0]['endpoints'].reverse()):
            path=self.clone(); self.mutate_pair(path,change)
            with self.assertRaises(ValueError): verify(path)

    def test_private_expert_maps_and_inward_source_ranges_rejected(self):
        for change in (lambda b:b['bounds'].__setitem__(0,b['bounds'][1]),
                       lambda b:b['rows'][0].update(rhs='99')):
            path=self.clone(); m=load(path,'manifest.json'); ref=m['blocks'][1]
            b=load(path,ref['file']); change(b); m['blocks'][1]={**self.rewrite(path,ref['file'],b),'name':ref['name']}
            self.rewrite(path,'manifest.json',m)
            with self.assertRaises(ValueError): verify(path)
        path=self.clone(); self.mutate_pair(path,lambda p:p['context']['blocks'].__setitem__(1,p['context']['blocks'][0]))
        with self.assertRaises(ValueError): verify(path)

    def test_partial_reuse_is_not_pair_reuse_or_mc_lp_bound(self):
        path=self.clone('tied_partial_reuse','mccormick'); m=load(path,'manifest.json')
        self.assertEqual(sum(o=='SOURCE_BOX_REUSE' for o in verify(path)['origins']),3)
        p=load(path,m['pairs'][0]['file']); self.assertEqual(p['duties'][0]['origin'],'SOURCE_BOX_REUSE')
        self.mutate_pair(path,lambda p:p['duties'][0].update(certificate={'kind':'SOURCE_BOX_FACT','lower_bound':'999'}))
        with self.assertRaises(ValueError): verify(path)
        path=self.clone('tied_partial_reuse','mccormick')
        self.mutate_pair(path,lambda p:p['duties'][0].update(lp_sha256='0'*64))
        with self.assertRaises(ValueError): verify(path)

    def test_dual_stale_target_null_hash_and_cost_metadata_rejected(self):
        for change in (lambda p:p['duties'][0]['endpoints'][0].update(lp_sha256='0'*64),
                       lambda p:p['duties'][0]['endpoints'][0]['certificate'].update(claimed_lower_bound='999')):
            path=self.clone(); self.mutate_pair(path,change)
            with self.assertRaises(ValueError): verify(path)
        for field,value in (('sha256',None),('bytes',1)):
            path=self.clone(); m=load(path,'manifest.json'); m['blocks'][0][field]=value; self.rewrite(path,'manifest.json',m)
            with self.assertRaises(ValueError): verify(path)

    def test_proposer_cannot_pollute_shared_bases_or_retained_certificate(self):
        path=self.root/'pollution'; path.mkdir(); s=pack(weighted_source(),path/'source',lambda:None)
        held=[]
        def candidate(lp,**kw):
            result=self.propose(lp,**kw)
            if held: held[-1]['claimed_lower_bound']='999'
            held.append(result); lp['E']['data'][:]=['999']*len(lp['E']['data']); return result
        build(path,expected_source_manifest=s,mode='endpoints',deadline=time.monotonic()+300,proposer=candidate)
        self.assertEqual(verify(path)['positive'],3)

    def test_chunk_gap_duplicate_swap_or_unbound_digest_rejected(self):
        for change in (lambda t:t['chunks'][0].update(offset=8),lambda t:t['chunks'].append(t['chunks'][0]),
                       lambda t:t['chunks'][0].update(sha256=None),lambda t:t.update(byte_order='big'),
                       lambda t:t.update(shape=[2])):
            path=self.clone(); m=load(path/'source','manifest.json'); change(m['tensors'][0]); self.rewrite(path/'source','manifest.json',m)
            with self.assertRaises(ValueError):
                v=Source(path/'source',identity(m),lambda:None); list(v.nodes())
        path=self.clone('unresolved_sign'); m=load(path/'source','manifest.json')
        t=next(t for t in m['tensors'] if len(t['chunks'])>1); t['chunks'].reverse(); self.rewrite(path/'source','manifest.json',m)
        with self.assertRaises(ValueError): Source(path/'source',identity(m),lambda:None)

    def test_manifest_inventory_stays_fixed_through_check(self):
        original=Source.nodes
        path=self.clone()
        def late_source(view):
            yield from original(view)
            (view.root/'late.bin').write_bytes(b'not in manifest')
        with patch.object(Source,'nodes',late_source),self.assertRaises(ValueError): verify(path)
        from scoped_source.sparse_check import check_bound
        path=self.clone()
        def late_bound(lp,cert):
            answer=check_bound(lp,cert)
            (path/'source'/'late.bin').write_bytes(b'not in manifest')
            return answer
        with patch('scoped_source.factored_check.check_bound',side_effect=late_bound),self.assertRaises(ValueError): verify(path)

    def test_graph_request_inventory_and_parameter_bindings(self):
        changes=[lambda d:d['networks'].reverse(),lambda d:d['request'].update(label=99),
                 lambda d:d['state_inventory'].pop(),lambda d:d['networks'][0]['layers'][0].update(index=2),
                 lambda d:d['networks'][0]['layers'][0].update(weight_name='experts.0.0.weight'),
                 lambda d:d['request'].update(top_k=True),lambda d:d['request'].update(radius='-1')]
        for change in changes:
            path=self.clone(); m=load(path/'source','manifest.json'); change(m['declaration']); self.rewrite(path/'source','manifest.json',m)
            with self.assertRaises((ValueError,KeyError)):
                v=Source(path/'source',identity(m),lambda:None); list(v.nodes())

    def test_corrupt_source_bytes_even_with_rehashed_chunk_rejected(self):
        import struct
        for raw in (struct.pack('<d',float('nan')),struct.pack('<d',-0.0)):
            path=self.clone(); m=load(path/'source','manifest.json'); t=m['tensors'][0]; ref=t['chunks'][0]
            (path/'source'/ref['file']).write_bytes(raw); ref['sha256']=digest(raw); self.rewrite(path/'source','manifest.json',m)
            with self.assertRaises(ValueError): Source(path/'source',identity(m),lambda:None)

    def test_rehashed_weight_still_bound_to_full_state(self):
        path=self.clone(); m=load(path/'source','manifest.json')
        ref=next(t for t in m['tensors'] if t['name'].endswith('.weight'))['chunks'][0]
        member=path/'source'/ref['file']; raw=bytearray(member.read_bytes()); raw[0]^=1
        member.write_bytes(raw); ref['sha256']=digest(raw); self.rewrite(path/'source','manifest.json',m)
        with self.assertRaisesRegex(ValueError,'tensor/state identity'):
            view=Source(path/'source',identity(m),lambda:None); list(view.nodes())

    def test_unused_but_valid_inventory_parameter_rejected(self):
        import hashlib
        import math
        from router_source.checker import tensor
        doc=weighted_source(); extra=deepcopy(doc['state_inventory'][0]); extra['name']='unused.weight'
        payload=deepcopy(doc['networks'][0]['layers'][0]['weight']); ident,_=tensor(payload)
        extra.update(ident); doc['state_inventory'].append(extra); doc['state_inventory'].sort(key=lambda v:v['name'])
        h=hashlib.sha256()
        for v in doc['state_inventory']: h.update(v['name'].encode()); h.update(v['sha256'].encode())
        doc['request']['model_state']={'sha256':h.hexdigest(),'tensor_count':len(doc['state_inventory']),
            'parameter_count':sum(math.prod(v['shape']) for v in doc['state_inventory'])}
        path=self.root/'unused-parameter'; s=pack(doc,path,lambda:None)
        m=load(path,'manifest.json'); raw=base64.b64decode(payload['bytes']); i=len(m['tensors'])
        chunk=write(path,f't/{i:05d}-00000.bin',raw,binary=True)
        m['tensors'].append({'name':extra['name'],'dtype':payload['dtype'],'shape':payload['shape'],
            'byte_order':'little','chunks':[{**chunk,'offset':0}]})
        self.rewrite(path,'manifest.json',m)
        view=Source(path,identity(m),lambda:None)
        with self.assertRaisesRegex(ValueError,'unused source tensor'): list(view.nodes())

    def test_input_shape_dtype_flatten_and_relu_modes(self):
        doc=synthetic(experts=3,classes=4,width=2)
        path=self.root/'dimension-control'; path.mkdir(); s=pack(doc,path/'source',lambda:None,8)
        view=Source(path/'source',s,lambda:None); self.assertEqual(dict(view.nodes()),index(doc,identity(doc),lambda:None)[1])
        for change in (lambda m:m['tensors'][0].update(dtype='torch.float32'),
                       lambda m:m['declaration']['networks'][0]['layers'][0].update(dimensions=[0,-1]),
                       lambda m:m['declaration']['networks'][1]['layers'][1].update(inplace=True)):
            copied=Path(tempfile.mkdtemp(prefix='shape-',dir=self.root)); shutil.copytree(path/'source',copied,dirs_exist_ok=True)
            m=load(copied,'manifest.json'); change(m); self.rewrite(copied,'manifest.json',m)
            with self.assertRaises(ValueError): v=Source(copied,identity(m),lambda:None); list(v.nodes())

    def test_real_flatten_operator_and_unused_topology_parameters(self):
        doc=synthetic(experts=3,classes=4,width=2)
        for graph in doc['networks']:
            graph['layers'].append({'index':len(graph['layers']),'kind':'Flatten','training':False,'dimensions':[1,-1]})
        path=self.root/'flatten'; s=pack(doc,path,lambda:None)
        view=Source(path,s,lambda:None); _,nodes,outputs=index(doc,identity(doc),lambda:None)
        self.assertEqual(dict(view.nodes()),nodes); self.assertEqual(view.outputs,outputs)
        m=load(path,'manifest.json'); m['declaration']['networks'][0]['layers'][-1]['dimensions']=[0,-1]
        self.rewrite(path,'manifest.json',m)
        with self.assertRaisesRegex(ValueError,'flatten contract'):
            view=Source(path,identity(m),lambda:None); list(view.nodes())
        doc['networks'][0]['layers'][-1]['dimensions']=[0,-1]
        with self.assertRaises(ValueError): index(doc,identity(doc),lambda:None)
        doc=synthetic(experts=3,classes=4,width=2)
        path=self.root/'unused-topology'; pack(doc,path,lambda:None); m=load(path,'manifest.json')
        m['declaration']['networks'][1]['layers'][0]={'index':0,'kind':'Flatten','training':False,'dimensions':[1,-1]}
        self.rewrite(path,'manifest.json',m); view=Source(path,identity(m),lambda:None)
        with self.assertRaisesRegex(ValueError,'unused source tensor'): list(view.nodes())

    def test_archive_reconstructs_source_and_lp_not_only_results(self):
        from scripts.check_h2_factored import equivalent
        for doc,reuse,arms in self.examples.values():
            for mode,(path,old,result) in arms.items():
                m=load(path,'manifest.json'); args={'source_sha':m['source_manifest_sha256'],
                    'proof_sha':identity(m),'legacy_source_sha':identity(doc),'mode':mode}
                compared=equivalent(doc,old,path,**args)
                self.assertEqual(compared['properties'],result['required'])
                changed=deepcopy(doc); changed['request']['margin']='1/10'
                with self.assertRaisesRegex(ValueError,'differential source'):
                    equivalent(changed,old,path,**{**args,'legacy_source_sha':identity(changed)})
                changed=deepcopy(old); changed['reuse_requested']=[[[0,1],1]] if not old['reuse_requested'] else []
                with self.assertRaisesRegex(ValueError,'differential reuse'):
                    equivalent(doc,changed,path,**args)

    def test_paths_symlinks_duplicate_json_and_total_limits(self):
        path=self.clone()
        with self.assertRaises(ValueError): read(path,'../manifest.json')
        with self.assertRaises(ValueError): decode(b'{"a":0,"a":1}')
        ref=load(path,'manifest.json')['blocks'][0]; (path/ref['file']).unlink(); (path/ref['file']).symlink_to(path/'manifest.json')
        with self.assertRaises(ValueError): verify(path)
        path=self.clone(); (path/'extra.json').write_text('{}')
        with self.assertRaises(ValueError): verify(path)
        from scoped_source.factored_io import inventory
        with self.assertRaises(ValueError): inventory(path,[str(p.relative_to(path)) for p in path.rglob('*') if p.is_file()],1)

    def test_cutoff_at_source_pair_and_final_check_never_accepts(self):
        path=self.examples['weighted_sign'][2]['endpoints'][0]; m=load(path,'manifest.json')
        total=[0]
        def count(): total[0]+=1
        with patch('scoped_source.factored_check.clock',return_value=count):
            check(path,expected_source_manifest=m['source_manifest_sha256'],expected_proof_manifest=identity(m),
                  expected_mode='endpoints',deadline=time.monotonic()+300)
        for stop in (1,15,total[0]//2,total[0]):
            ticks=[0]
            def tick():
                ticks[0]+=1
                if ticks[0]>=stop: raise TimeoutError('control cutoff')
            with patch('scoped_source.factored_check.clock',return_value=tick),self.assertRaises(TimeoutError):
                check(path,expected_source_manifest=m['source_manifest_sha256'],expected_proof_manifest=identity(m),
                      expected_mode='endpoints',deadline=time.monotonic()+300)
        path=self.root/'partial'; path.mkdir(); s=pack(weighted_source(),path/'source',lambda:None)
        def expired(): raise TimeoutError('before generation')
        with patch('scoped_source.factored_build.clock',return_value=expired),self.assertRaises(TimeoutError):
            build(path,expected_source_manifest=s,mode='endpoints',deadline=time.monotonic()+300)
        self.assertFalse((path/'manifest.json').exists())

    def test_checker_is_solver_and_constructor_free(self):
        root=self.examples['weighted_sign'][2]['endpoints'][0]
        code='''import sys,time,json
sys.path.insert(0,sys.argv[1])
def guard(event,args):
 if event=='import' and (args[0].split('.')[0] in ('torch','numpy','scipy','highspy','act') or args[0] in ('scoped_source.factored_build','scoped_source.endpoint_source_build','scoped_source.endpoint_build')): raise ImportError(args[0])
sys.addaudithook(guard)
from scoped_source.factored_check import check
from scoped_source.factored_io import load
from source_enclosure.format import identity
m=load(sys.argv[2],'manifest.json')
print(json.dumps(check(sys.argv[2],expected_source_manifest=m['source_manifest_sha256'],expected_proof_manifest=identity(m),expected_mode='endpoints',deadline=time.monotonic()+30)))
'''
        result=subprocess.run([sys.executable,'-B','-I','-S','-c',code,str(ROOT),str(root)],text=True,capture_output=True,timeout=35)
        self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(json.loads(result.stdout)['positive'],3)

    def test_protocol_and_external_anchors(self):
        self.assertEqual(protocol()['real_requests'],0)
        path=self.examples['weighted_sign'][2]['endpoints'][0]; m=load(path,'manifest.json')
        for key,val in (('expected_source_manifest','0'*64),('expected_proof_manifest','0'*64),('expected_mode','mccormick')):
            args={'expected_source_manifest':m['source_manifest_sha256'],'expected_proof_manifest':identity(m),'expected_mode':'endpoints','deadline':time.monotonic()+300}
            args[key]=val
            with self.assertRaises(ValueError): check(path,**args)


if __name__=='__main__': unittest.main()
