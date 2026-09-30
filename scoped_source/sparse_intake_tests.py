"""Supported object capture controls; no real checkpoints, inputs or training."""
import copy
from fractions import Fraction as F
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch

from scoped_proof.io import ROOT, PYTHON, load, save, sha
from scoped_source.sparse_intake import model_fixture, prepare, capture_bound, validate_fixture
from scoped_source.sparse_ir import index
from scoped_source.sparse_check import check
from scoped_source.sparse_supervised import supervise, audit, bind_producer
from source_enclosure.format import identity

OPTIONS = {'seed':0,'experts':3,'classes':3,'width':4,'hidden':[4,4]}


def observed(root,spec,budget=30):
    result = supervise(root,spec,budget=budget)
    save(root/'caller_observation.json',result)
    audit(root,result)
    return result


class H1IntakeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keep = os.environ.get('H1_INTAKE_ROOT')
        if keep:
            cls.root = Path(keep); cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp = tempfile.TemporaryDirectory(prefix='h1-intake-',dir=ROOT/'data/moe/tmp')
            cls.root = Path(cls.tmp.name)
        # Snapshot the complete producer/checker dependency inventory, not results.
        names = list((ROOT/'scoped_source').glob('sparse_*.py')) + [ROOT/n for n in (
            'scoped_source/capture.py','scoped_source/graph.py','scoped_proof/io.py',
            'scoped_proof/supervisor.py','source_enclosure/format.py','router_source/checker.py',
            'upstream_source/checker.py','act/back_end/moe/model.py','act/back_end/moe/schema.py',
            'act/back_end/solver/lp_certificate.py','act/back_end/solver/sparse_lp_certificate.py')]
        bindings = {}
        for path in names:
            relative = str(path.relative_to(ROOT)); out = cls.root/'implementation'/relative
            out.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(path,out); bindings[relative] = sha(out)
        save(cls.root/'implementation.json',bindings)
        cls.spec, cls.doc = prepare(OPTIONS)
        save(cls.root/'pre_execution_spec.json',cls.spec)
        cls.runs = {}
        for mode in ('dependency','full'):
            spec = {**cls.spec,'mode':mode}
            cls.runs[mode] = observed(cls.root/mode,spec)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def take(self,model,center,request,expected=None):
        return capture_bound(model,center,request,expected_source_sha256=expected or self.spec['source_sha256'],
            deadline=time.monotonic()+30)

    def test_whole_flow_all_obligations_and_costs(self):
        for mode,result in self.runs.items():
            self.assertIn(result['status'],('CHECKED_DECLARED_SOURCE_POSITIVE','UNKNOWN_NONPOSITIVE','UNKNOWN_MISSING_EVIDENCE'))
            root = self.root/mode; row = load(root/'accepted.json')
            self.assertEqual(row['required'],6)
            self.assertEqual(row['missing'],sum(v['lower_bound'] is None for v in row['result']['obligations']))
            if row['missing']: self.assertFalse(result['complete_declared_source_proof'])
            self.assertFalse(row['result']['deployed_float_SAFE'])
            events = [json.loads(line) for line in (root/'produce_events.jsonl').read_text().splitlines()]
            names = {v['operation'] for v in events if v['event']=='EXIT'}
            self.assertTrue({'model_creation_imports','capture_and_validate','construct_and_propose',
                'serialize_and_bundle'} <= names)
            terminal = load(root/'terminal.json')
            self.assertGreater(result['seconds'],terminal['stage_seconds'])
            self.assertTrue(audit(root,result)['external_completion_observed'])

    def test_dense_hidden_dependencies_not_pruned(self):
        _,nodes,outputs = index(self.doc,self.spec['source_sha256'],lambda:None)
        proof = load(self.root/'dependency/bundle/proof.json')
        active = {v for row in proof['obligations'] for v in row['variables']}
        endpoints = {key for values in outputs.values() for key in values}
        hidden_and_input = set(nodes)-endpoints
        self.assertTrue(hidden_and_input <= active)
        self.assertEqual(set(proof['bank']),hidden_and_input)
        # All dense Linear weights are genuinely nonzero, distinct by network.
        from router_source.checker import tensor
        weights = [tensor(layer['weight'])[1] for net in self.doc['networks']
            for layer in net['layers'] if layer['kind']=='Linear']
        self.assertTrue(all(all(v for v in w) for w in weights))
        self.assertEqual(len({tuple(w) for w in weights}),len(weights))

    def test_full_vs_dependency_complete_status_not_forced_positive(self):
        a = load(self.root/'dependency/accepted.json')['result']
        b = load(self.root/'full/accepted.json')['result']
        self.assertEqual(a['required'],b['required'])
        self.assertEqual([(r['pair'],r['competitor']) for r in a['obligations']],
                         [(r['pair'],r['competitor']) for r in b['obligations']])
        # Different numerical candidate duals need not produce bit-identical bounds.
        for mode,checked in [('dependency',a),('full',b)]:
            proof=load(self.root/mode/'bundle/proof.json')
            for evidence,result in zip(proof['obligations'],checked['obligations']):
                self.assertEqual(evidence['certificate'] is None,result['lower_bound'] is None)
                if result['lower_bound'] is None: self.assertFalse(result['positive'])

    def test_capture_never_calls_forward(self):
        import torch
        m,x,r = model_fixture(OPTIONS)
        with patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('forward called')):
            self.assertEqual(self.take(m,x,r),self.doc)

    def test_overrides_hooks_and_count_mismatch_rejected(self):
        import torch
        for kind in ('route','forward_with_routing','state_dict','_save_to_state_dict',
                     '_call_impl','state_hook','pre_state_hook','forward_hook','count','global_hook'):
            with self.subTest(kind=kind):
                m,x,r = model_fixture(OPTIONS); handle = None
                if kind in ('route','forward_with_routing','state_dict','_save_to_state_dict','_call_impl'):
                    setattr(m,kind,lambda *a,**k:None)
                elif kind=='state_hook': handle = m._register_state_dict_hook(lambda *a:None)
                elif kind=='pre_state_hook': handle = m.register_state_dict_pre_hook(lambda *a:None)
                elif kind=='forward_hook': handle = m.experts[0].register_forward_hook(lambda *a:None)
                elif kind=='count': m.experts.append(copy.deepcopy(m.experts[0]))
                elif kind=='global_hook': handle = torch.nn.modules.module.register_module_forward_hook(lambda *a:None)
                try:
                    with self.assertRaises(ValueError): self.take(m,x,r)
                finally:
                    if handle is not None: handle.remove()

    def test_parameter_input_and_property_binding_and_aliases(self):
        import torch
        m,x,r = model_fixture(OPTIONS); doc = self.take(m,x,r); old = identity(doc)
        with torch.no_grad():
            next(m.parameters()).add_(1); x.add_(.1)
        self.assertEqual(identity(doc),old)  # byte snapshot, not a tensor alias
        with self.assertRaises(ValueError): self.take(m,x,r)
        for field,value in [('label',1),('radius','1/4'),('margin','0'),('clip',['-2','2'])]:
            m,x,r = model_fixture(OPTIONS); r[field]=value
            with self.assertRaises(ValueError): self.take(m,x,r)

    def test_tensor_serialization_overrides_rejected(self):
        import torch
        for kind in ('parameter','center','buffer'):
            m,x,r=model_fixture(OPTIONS)
            if kind=='parameter': target=next(m.parameters())
            elif kind=='center': target=x
            else:
                target=torch.zeros(1,dtype=torch.float64); m.register_buffer('extra',target)
            target.detach=lambda:torch.zeros_like(target)
            with self.assertRaises(ValueError): self.take(m,x,r)

    def test_non_square_dimensions_and_zero_radius_capture(self):
        for opts in ({**OPTIONS,'experts':2,'classes':4,'width':3,'hidden':[5,2]},
                     {**OPTIONS,'experts':4,'classes':2,'width':2,'hidden':[3]}):
            spec,doc = prepare(opts); m,x,r=model_fixture(opts)
            self.assertEqual(self.take(m,x,r,spec['source_sha256']),doc)
            r['radius']='0'
            from scoped_source.capture import capture
            singleton = capture(m,x,deadline=time.monotonic()+30,**r)
            self.take(m,x,r,identity(singleton))

    def test_exact_declared_graph_vs_native_component_probes(self):
        import torch
        m,x,r = model_fixture(OPTIONS)
        _,nodes,outputs = index(self.doc,self.spec['source_sha256'],lambda:None)
        for point in ([F(0)]*4,[F(-1,8),F(1,16),F(1,8),F(-1,16)]):
            values={}
            for name,n in nodes.items():
                if n['kind']=='input': v=point[int(name.split('/')[1])]
                elif n['kind']=='affine': v=n['bias']+sum((c*values[k] for k,c in n['terms'].items()),F(0))
                else: v=max(F(0),values[n['parent']])
                values[name]=v
            batch=torch.tensor([[list(map(float,point))]],dtype=torch.float64)
            with torch.no_grad():
                native=[m.router(batch)]+[net(batch) for net in m.experts]
            for name,got in zip(outputs,native):
                expected=torch.tensor([[float(values[k]) for k in outputs[name]]],dtype=torch.float64)
                torch.testing.assert_close(got,expected,rtol=1e-12,atol=1e-12)
        # Finite probes are an implementation control, never a universal FP proof.

    def test_old_source_proof_and_missing_obligation_rejected(self):
        proof=load(self.root/'dependency/bundle/proof.json')
        spec,newdoc=prepare({**OPTIONS,'seed':1})
        with self.assertRaises(ValueError):
            check(newdoc,proof,expected_source_sha256=spec['source_sha256'],deadline=time.monotonic()+30)
        proof['obligations'].pop()
        with self.assertRaises(ValueError):
            check(self.doc,proof,expected_source_sha256=self.spec['source_sha256'],deadline=time.monotonic()+30)

    def test_frozen_source_mutation_and_exception_supervised(self):
        for fault in ('mutate_model','capture_exception'):
            result=observed(self.root/fault,{**self.spec,'control':fault})
            self.assertEqual(result['status'],'ERROR')
            self.assertFalse((self.root/fault/'generation.json').exists())

    def test_capture_deadline_partial_not_positive(self):
        root=self.root/'capture_delay'
        result=observed(root,{**self.spec,'control':'capture_delay'},budget=8)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertTrue((root/'capture_partial.json').exists())
        self.assertFalse(result['complete_declared_source_proof'])
        m,x,r=model_fixture(OPTIONS)
        with self.assertRaises(TimeoutError):
            capture_bound(m,x,r,expected_source_sha256=self.spec['source_sha256'],deadline=time.monotonic()-1)

    def test_implementation_receipt_and_schema_tampering(self):
        root=self.root/'dependency'; inv=load(root/'invocation.json'); inv['producer_sources']['scoped_source/capture.py']='0'*64
        with self.assertRaises(ValueError): bind_producer(self.spec,inv)
        from scoped_source.sparse_supervised import validate_spec
        for field,value in [('fixture',{**OPTIONS,'checkpoint':'/not/permitted'}),('schema','REAL_REQUEST')]:
            with self.assertRaises(ValueError): validate_spec({**self.spec,field:value})

    def test_relocated_capture_proof_no_model_or_repository(self):
        root=self.root/'moved'; shutil.copytree(self.root/'dependency/bundle',root)
        built=load(self.root/'dependency/built.json')
        result=subprocess.run([PYTHON,'-B','-I','-S',str(root/'verify.py'),
            '--manifest-sha',built['sha256'],'--source-sha',self.spec['source_sha256']],
            cwd=root,env=dict(os.environ,PYTHONPATH='/absent'),capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        checked=json.loads(result.stdout)
        self.assertEqual(checked['result']['required'],6)
        self.assertFalse(checked['solver_imported']);self.assertFalse(checked['producer_imported'])


if __name__=='__main__': unittest.main()
