"""Fixed live-object H2 intake controls; no trained model or real inputs."""
from copy import deepcopy
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
from scoped_source.endpoint_intake import specification, model_fixture, expected_source, CAPTURE_SHA256
from scoped_source.endpoint_supervised import supervise, audit, producer_sources, validate_spec, receive, POSITIVE
from scoped_source.endpoint_portable import CODE
from scoped_source.sparse_intake import capture_bound
from scoped_source.endpoint_source_controls import weighted_source, values_at
from scoped_source.endpoint_source_check import check
from scoped_source.sparse_ir import index
from source_enclosure.format import identity


def observed(root, spec, budget=30):
    result = supervise(root,spec,budget=budget)
    save(root/'caller_observation.json',result)
    audit(root,result)
    return result


class H2IntakeControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keep = os.environ.get('H2_INTAKE_ROOT')
        if keep:
            cls.root = Path(keep); cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp = tempfile.TemporaryDirectory(prefix='h2-intake-',dir=ROOT/'data/moe/tmp')
            cls.root = Path(cls.tmp.name)
        cls.doc = expected_source(); cls.spec = specification()
        names = set(CODE) | set(producer_sources(cls.spec)) | {
            'scoped_source/endpoint_verify.py','scoped_source/endpoint_intake_tests.py',
            'scripts/archive_h2_intake.py'}
        bindings = {}
        for name in sorted(names):
            target = cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name] = sha(target)
        save(cls.root/'implementation.json',bindings)
        # These are pre-execution expectations, not cached intake or LP evidence.
        save(cls.root/'expected_source.json',cls.doc)
        save(cls.root/'pre_execution_spec.json',cls.spec)
        cls.runs = {mode:observed(cls.root/mode,specification(mode)) for mode in ('endpoints','mccormick')}

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def take(self,m,x,r):
        return capture_bound(m,x,r,expected_source_sha256=CAPTURE_SHA256,deadline=time.monotonic()+30)

    def test_complete_calls_same_source_base_gate_and_all_output(self):
        packages = []
        for mode,run in self.runs.items():
            path = self.root/mode; positive = mode == 'endpoints'
            self.assertEqual(run['status'],POSITIVE if positive else 'UNKNOWN_NONPOSITIVE')
            self.assertEqual(audit(path,run)['positive_execution_accepted'],positive)
            self.assertFalse(audit(path)['positive_execution_accepted'])
            source = load(path/'bundle/source.json'); self.assertEqual(source,self.doc)
            accepted = load(path/'accepted.json'); result = accepted['result']
            self.assertEqual((result['positive'],result['required']), (3 if positive else 2,3))
            self.assertFalse(result['deployed_float_SAFE']); self.assertFalse(result['hard_budget_supervision'])
            self.assertEqual(accepted['model_capture_receipt_content_sha256'],identity(load(path/'model_intake.json')))
            proof = load(path/'bundle/proof.json'); packages.append(proof)
            self.assertEqual(check(source,proof,expected_source_sha256=CAPTURE_SHA256,
                                   expected_mode=mode,deadline=time.monotonic()+30),result)
            events = [json.loads(s) for s in (path/'produce_events.jsonl').read_text().splitlines()]
            names = {e['operation'] for e in events if e['event']=='EXIT'}
            self.assertTrue({'model_creation_imports','capture_and_validate','capture_receipt_publication',
                'source_construct_and_propose','serialize_and_bundle'} <= names)
            self.assertNotIn('source_creation_imports',names)
            self.assertGreater(run['seconds'],load(path/'terminal.json')['stage_seconds'])
        self.assertEqual(packages[0]['request'],packages[1]['request'])
        self.assertEqual(packages[0]['reuse_requested'],packages[1]['reuse_requested'])

    def test_exact_capture_matches_predeclared_stored_coefficients(self):
        m,x,r = model_fixture(); doc = self.take(m,x,r)
        self.assertEqual(doc,self.doc); self.assertEqual(identity(doc),CAPTURE_SHA256)
        plain = deepcopy(doc); plain.pop('trust')
        self.assertEqual(plain,weighted_source()); self.assertNotEqual(identity(plain),CAPTURE_SHA256)

    def test_creation_preserves_rng_and_experts_are_private(self):
        import torch
        before = torch.random.get_rng_state().clone(); m,x,r = model_fixture()
        self.assertTrue(torch.equal(before,torch.random.get_rng_state()))
        params = [p for expert in m.experts for p in expert.parameters()]
        modules = [module for expert in m.experts for module in expert.modules()]
        self.assertEqual(len({id(v) for v in modules}),len(modules))
        self.assertEqual(len({p.data_ptr() for p in params}),len(params))
        self.assertTrue(all(p.device.type=='cpu' and p.dtype==torch.float64 for p in m.parameters()))
        self.assertFalse(m.training)

    def test_capture_does_not_execute_forward(self):
        import torch
        m,x,r = model_fixture()
        with patch.object(torch.nn.Module,'_call_impl',side_effect=AssertionError('unexpected native forward')):
            self.assertEqual(self.take(m,x,r),self.doc)

    def test_hooks_overrides_and_tensor_alias_snapshot(self):
        import torch
        for kind in ('forward','route','state_dict','state_hook','global_hook','tensor_override'):
            m,x,r = model_fixture(); handle = None
            if kind in ('forward','route','state_dict'): setattr(m,kind,lambda *a,**k:None)
            elif kind=='state_hook': handle = m._register_state_dict_hook(lambda *a:None)
            elif kind=='global_hook': handle = torch.nn.modules.module.register_module_forward_hook(lambda *a:None)
            else: x.detach = lambda:torch.zeros_like(x)
            try:
                with self.assertRaises(ValueError): self.take(m,x,r)
            finally:
                if handle is not None: handle.remove()
        m,x,r=model_fixture(); captured=self.take(m,x,r)
        with torch.no_grad(): next(m.parameters()).add_(1); x.add_(.25)
        self.assertEqual(captured,self.doc)
        with self.assertRaises(ValueError): self.take(m,x,r)

    def test_changed_domain_property_dtype_or_operator_rejected(self):
        import torch
        for key,value in (('radius','1/2'),('label',1),('margin','1/10'),('clip',['0','1'])):
            m,x,r=model_fixture(); r[key]=value
            with self.assertRaises(ValueError): self.take(m,x,r)
        for kind in ('float32','training','BN'):
            m,x,r=model_fixture()
            if kind=='float32': m.float(); x=x.float()
            elif kind=='training': m.train()
            else: m.experts[0][1]=torch.nn.BatchNorm1d(3).double().eval()
            with self.assertRaises(ValueError): self.take(m,x,r)

    def test_native_component_and_selected_weight_probes(self):
        # Finite differential only; this is NOT an all-domain float proof.
        import torch
        m,_,_=model_fixture(); _,_,outputs=index(self.doc,CAPTURE_SHA256,lambda:None)
        for point in (F(-1),F(-1,2),F(0),F(1)):
            values=values_at(self.doc,[point]); x=torch.tensor([[float(point)]],dtype=torch.float64)
            with torch.no_grad():
                components=[m.router(x)]+[e(x) for e in m.experts]
                got,route=m.forward_with_routing(x)
                chosen=[int(i) for i in route.indices[0]]
                expected=sum(route.weights[0,j]*components[1+i] for j,i in enumerate(chosen))
            torch.testing.assert_close(got,expected,rtol=0,atol=0)
            for name,actual in zip(outputs,components):
                exact=torch.tensor([[float(values[k]) for k in outputs[name]]],dtype=torch.float64)
                torch.testing.assert_close(actual,exact,rtol=1e-12,atol=1e-12)
            router=[values[k] for k in outputs['router']]
            self.assertTrue(all(router[i]>=router[j] for i in chosen for j in range(3) if j not in chosen))

    def test_capture_exception_mutation_and_cutoff_are_terminal(self):
        for fault in ('mutate_model','mutate_input','capture_exception','capture_delay'):
            path=self.root/fault; result=observed(path,specification(control=fault),8 if fault=='capture_delay' else 30)
            self.assertEqual(result['status'],'TIMEOUT' if fault=='capture_delay' else 'ERROR')
            self.assertFalse((path/'generation.json').exists()); self.assertFalse((path/'accepted.json').exists())
            if fault=='capture_delay': self.assertTrue((path/'capture_partial.json').exists())
        m,x,r=model_fixture()
        with self.assertRaises(TimeoutError):
            capture_bound(m,x,r,expected_source_sha256=CAPTURE_SHA256,deadline=time.monotonic()-1)

    def test_missing_and_malformed_full_evidence_not_accepted(self):
        for fault,status in (('missing_certificate','UNKNOWN_MISSING_EVIDENCE'),('omit_property','ERROR')):
            result=observed(self.root/fault,specification(control=fault))
            self.assertEqual(result['status'],status); self.assertFalse(result['complete_declared_source_proof'])
        result=observed(self.root/'rewrite_check_stdout',specification('mccormick','rewrite_check_stdout'))
        self.assertEqual(result['status'],'ERROR')

    def test_source_and_scope_admission_is_not_arbitrary_factory(self):
        for key,value in (('case','tied_partial_reuse'),('reuse',[[[0,1],1]]),('source_sha256','0'*64),
                          ('protocol_sha256','0'*64),('mode','full'),('checkpoint','unregistered')):
            with self.assertRaises(ValueError): validate_spec({**self.spec,key:value})
        with self.assertRaises(FileExistsError): supervise(self.root/'endpoints',self.spec)
        from scoped_source.endpoint_supervised import bind_producer
        inv=load(self.root/'endpoints/invocation.json'); inv['producer_sources']['scoped_source/capture.py']='0'*64
        with self.assertRaises(ValueError): bind_producer(self.spec,inv)

    def test_capture_receipt_must_match_checked_bundle(self):
        for field,value in (('invocation','foreign'),('source_sha256','0'*64),('request_sha256','0'*64),
                            ('model_state',{}),('native_float_proof',True)):
            path=self.root/('receipt-'+field); shutil.copytree(self.root/'endpoints',path)
            captured=load(path/'model_intake.json'); captured[field]=value
            (path/'model_intake.json').write_text(json.dumps(captured))
            inv=load(path/'invocation.json'); terminal=load(path/'terminal.json')
            with self.assertRaisesRegex(ValueError,'captured model/source receipt'):
                receive(path,self.spec,inv['invocation'],terminal['invocation_sha256'],terminal['checker_stdout_sha256'])

    def test_old_declaration_or_missing_output_cannot_supply_new_proof(self):
        proof=load(self.root/'endpoints/bundle/proof.json')
        old=weighted_source()
        with self.assertRaises(ValueError):
            check(old,proof,expected_source_sha256=identity(old),expected_mode='endpoints',deadline=time.monotonic()+30)
        proof['proof']['duties'].pop()
        with self.assertRaises(ValueError):
            check(self.doc,proof,expected_source_sha256=CAPTURE_SHA256,expected_mode='endpoints',deadline=time.monotonic()+30)

    def test_relocated_checker_needs_no_object_or_repository(self):
        source=self.root/'endpoints/bundle'; moved=self.root/'moved'; shutil.copytree(source,moved)
        hidden=self.root/'endpoints/bundle-hidden'; source.rename(hidden)
        built=load(self.root/'endpoints/built.json')
        try:
            result=subprocess.run([PYTHON,'-B','-I','-S',str(moved/'verify.py'),
                '--manifest-sha',built['sha256'],'--source-sha',CAPTURE_SHA256,'--mode','endpoints'],
                cwd=moved,env=dict(os.environ,PYTHONPATH='/absent'),text=True,capture_output=True,timeout=15)
        finally: hidden.rename(source)
        self.assertEqual(result.returncode,0,result.stderr)
        checked=json.loads(result.stdout)
        self.assertEqual(checked['result']['status'],POSITIVE)
        self.assertFalse(checked['producer_imported']); self.assertFalse(checked['solver_imported'])


if __name__=='__main__': unittest.main()
