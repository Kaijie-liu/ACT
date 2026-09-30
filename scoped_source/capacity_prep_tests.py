"""Fixed preparation controls, no full-size model and no LP solving."""
from copy import deepcopy
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import unittest
from unittest.mock import patch
from scoped_proof.io import ROOT,PYTHON,load,save,sha
from scoped_proof.supervisor import group_rss
from scoped_source.capacity_intake import sources,protocol,expected_request,PROTOCOL_SHA
from scoped_source.capacity_prepare import prepare,receive,validate_spec
from scoped_source.capacity_prep_audit import audit
from source_enclosure.format import identity


class CapacityPreparationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['H2_CAPACITY_CONTROL_ROOT'])
        cls.root.mkdir(parents=True,exist_ok=False)
        cls.bindings=sources()
        for name,digest in cls.bindings.items():
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target)
            if sha(target)!=digest: raise ValueError('snapshot changed')
        save(cls.root/'implementation.json',cls.bindings)
        cls.calls={}
        cases=[('normal','',30,None),('exception','exception',30,None),('partial','partial',30,None),
               ('wrong-recipe','wrong_recipe',30,None),('wrong-source','wrong_source',30,None),
               ('delay','delay',6,None),('late','late',6,None),('receive-delay','receive_delay',6,None),
               ('resource','',6,1),('prelaunch','',1e-9,None)]
        for name,control,budget,rss in cases:
            path=cls.root/name
            call=prepare(path,control=control,budget=budget,**({} if rss is None else {'rss_limit':rss}))
            save(path/'caller_observation.json',call); cls.calls[name]=(path,call)
        path=cls.root/'final-publication'
        def late_save(where,value):
            result=save(where,value)
            if where.name=='finish.json':
                inv=load(path/'invocation.json'); time.sleep(max(0.,inv['deadline']-time.monotonic())+.03)
            return result
        with patch('scoped_source.capacity_prepare.save',side_effect=late_save):
            call=prepare(path,budget=6)
        save(path/'caller_observation.json',call); cls.calls['final-publication']=(path,call)
        cls.good,cls.observed=cls.calls['normal']
        if cls.observed['status']!='IDENTITY_PREPARED': raise AssertionError(cls.observed)

    def copied(self,name):
        path=self.root/name; shutil.copytree(self.good,path); return path

    def check_candidate(self,path,candidate=None):
        if candidate is not None:
            (path/'prepared.json').write_text(json.dumps(candidate,sort_keys=True))
        return receive(path,load(path/'spec.json'),load(path/'invocation.json'),
                       time.monotonic()+10,sha(path/'prepared.json'))

    def test_frozen_recipe_and_tiny_no_proof(self):
        cfg=protocol(); self.assertEqual(identity(cfg),PROTOCOL_SHA)
        self.assertEqual((cfg['num_experts'],cfg['num_classes'],cfg['required_properties']),(8,10,252))
        r=audit(self.good,self.observed); c=r['source_identity']['candidate']
        self.assertTrue(r['identity_prepared']); self.assertFalse(r['real_capacity_admitted'])
        self.assertEqual(c['proof_status'],'NOT_A_PROOF'); self.assertEqual(c['solves'],0)
        self.assertEqual(r['source_identity']['source_check']['tensor_count'],15)

    def test_fixed_call_roster_and_whole_cost(self):
        expected={'normal':'IDENTITY_PREPARED','exception':'ERROR','partial':'ERROR','wrong-recipe':'ERROR',
            'wrong-source':'ERROR','delay':'TIMEOUT','late':'TIMEOUT','receive-delay':'TIMEOUT',
            'resource':'RESOURCE_LIMIT','prelaunch':'TIMEOUT','final-publication':'TIMEOUT'}
        self.assertEqual(set(expected),set(self.calls)); rows=[]
        for name,(path,call) in self.calls.items():
            with self.subTest(name=name):
                self.assertEqual(call['status'],expected[name]); checked=audit(path,call)
                self.assertEqual(checked['status'],expected[name]); self.assertFalse(checked['real_capacity_admitted'])
                self.assertGreaterEqual(call['seconds'],checked['stage_seconds']); rows.append(checked)
        save(self.root/'audited_calls.json',rows)

    def test_deadlines_identical_to_supervisor_and_cleanup(self):
        for path,call in self.calls.values():
            inv=load(path/'invocation.json'); terminal=load(path/'terminal.json')
            for stage in terminal['stages']:
                if stage['pid'] is None: continue
                expected=inv['produce_deadline'] if stage['phase']=='prepare' else inv['work_deadline']
                self.assertEqual(stage['deadline_monotonic'],expected)
                self.assertEqual(group_rss(stage['pid'])[1],[])
                self.assertTrue(stage['cleanup_included'])

    def test_partial_late_and_reception_delay_never_anchor(self):
        for name,file in [('partial','unaccepted_partial.json'),('late','prepared.json'),('receive-delay','prepared.json')]:
            path,call=self.calls[name]
            self.assertTrue((path/file).is_file()); self.assertIsNone(load(path/'terminal.json')['accepted'])
            self.assertFalse(audit(path,call)['identity_prepared'])

    def test_missing_observation_and_cost_pollution(self):
        with self.assertRaises(ValueError): audit(self.good,None)
        for changes in ({'seconds':-1},{'seconds':float('nan')},{'seconds':1000},{'invocation':'other'},
                        {'finish_sha256':'0'*64},{'real_capacity_admitted':True}):
            with self.subTest(changes=changes),self.assertRaises(ValueError):
                audit(self.good,{**self.observed,**changes})

    def test_null_or_invalid_hash_cannot_disable_binding(self):
        for index,bad in enumerate((None,'',True,'G'*64)):
            with self.assertRaises(ValueError): audit(self.good,{**self.observed,'finish_sha256':bad})
            for field in ('terminal_sha256','invocation_sha256','receipt_sha256','candidate_sha256'):
                path=self.copied('null-'+field+'-'+str(index)); finish=load(path/'finish.json')
                if field=='terminal_sha256': finish[field]=bad
                else:
                    terminal=load(path/'terminal.json'); terminal[field]=bad
                    (path/'terminal.json').write_text(json.dumps(terminal)); finish['terminal_sha256']=sha(path/'terminal.json')
                (path/'finish.json').write_text(json.dumps(finish))
                observed={**self.observed,'root':str(path),'finish_sha256':sha(path/'finish.json')}
                with self.subTest(field=field,bad=bad),self.assertRaises(ValueError): audit(path,observed)

    def test_late_source_inventory_or_manifest_pollution(self):
        import base64
        original=base64.b64encode
        for fault in ('extra','manifest'):
            path=self.copied('late-source-'+fault); changed=[]
            def encode(raw):
                result=original(raw)
                if not changed:
                    changed.append(True)
                    if fault=='extra': (path/'source/late').write_text('unexpected')
                    else:
                        m=load(path/'source/manifest.json'); m['declaration']['trust']='changed late'
                        (path/'source/manifest.json').write_text(json.dumps(m))
                return result
            with patch('scoped_source.capacity_sourcecheck.base64.b64encode',side_effect=encode):
                with self.assertRaises(ValueError): self.check_candidate(path)

    def test_publication_timeout_overrides_success(self):
        path,call=self.calls['final-publication']
        self.assertTrue((path/'publication_timeout.json').exists())
        self.assertFalse(audit(path,call)['identity_prepared'])
        with self.assertRaises(ValueError): audit(path,{**call,'status':'IDENTITY_PREPARED'})

    def test_parent_candidate_anchor_is_not_self_report(self):
        path=self.copied('changed-candidate'); c=load(path/'prepared.json'); c['torch_version']='replaced'
        (path/'prepared.json').write_text(json.dumps(c))
        with self.assertRaises(ValueError):
            receive(path,load(path/'spec.json'),load(path/'invocation.json'),time.monotonic()+10,
                    load(path/'terminal.json')['candidate_sha256'])

    def test_wrong_domain_invocation_or_guarantee_rejected(self):
        original=load(self.good/'prepared.json')
        for name in ('invocation','radius','proof','dataset','source'):
            path=self.copied('candidate-'+name); value=deepcopy(original)
            if name=='invocation': value['invocation']='alien'
            if name=='radius': value['request']['radius']='1/255'; value['request_sha256']=identity(value['request'])
            if name=='proof': value['proof_status']='SAFE'
            if name=='dataset': value['dataset_loaded']=True
            if name=='source': value['declared_source_sha256']='0'*64
            with self.subTest(name=name),self.assertRaises(ValueError): self.check_candidate(path,value)

    def test_rehashed_bad_chunk_structure_or_graph_rejected(self):
        for fault in ('offset','duplicate','missing','shape','state','operator','bytes'):
            path=self.copied('chunk-'+fault); m=load(path/'source/manifest.json'); c=load(path/'prepared.json')
            t=m['tensors'][0]; chunk=t['chunks'][0]
            if fault=='offset': chunk['offset']=8
            if fault=='duplicate': m['tensors'].append(deepcopy(t))
            if fault=='missing': t['chunks']=[]
            if fault=='shape': t['shape']=[1,2]
            if fault=='state': m['declaration']['state_inventory'][0]['sha256']='0'*64
            if fault=='operator': m['declaration']['networks'][0]['layers'][0]['kind']='Conv2d'
            if fault=='bytes':
                target=path/'source'/chunk['file']; target.write_bytes(b'\x00'*len(target.read_bytes()))
                # center is already zero; mutate the first expert/router tensor instead.
                chunk=m['tensors'][1]['chunks'][0]; target=path/'source'/chunk['file']
                target.write_bytes(b'\x00'*len(target.read_bytes())); chunk['sha256']=sha(target)
            (path/'source/manifest.json').write_text(json.dumps(m)); c['source_manifest_sha256']=identity(m)
            with self.subTest(fault=fault),self.assertRaises(ValueError): self.check_candidate(path,c)

    def test_relocated_source_same_identity(self):
        path=self.copied('relocated'); checked=self.check_candidate(path)
        self.assertEqual(checked,load(self.good/'received.json'))

    def test_checker_is_stdlib_and_does_not_import_solver(self):
        code=('import sys; from pathlib import Path; from scoped_proof.io import load; '
              'from scoped_source.capacity_prep_audit import audit; p=Path(sys.argv[1]); '
              'audit(p,load(p/"caller_observation.json")); '
              'assert not any(x in sys.modules for x in ("torch","numpy","scipy","highspy")); print("PASS")')
        result=subprocess.run([PYTHON,'-B','-S','-c',code,str(self.good)],cwd=ROOT,
                              text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr); self.assertEqual(result.stdout.strip(),'PASS')

    def test_checker_deadline_and_extra_inventory(self):
        with self.assertRaises(TimeoutError):
            receive(self.good,load(self.good/'spec.json'),load(self.good/'invocation.json'),0.,sha(self.good/'prepared.json'))
        path=self.copied('extra-file'); (path/'source/extra').write_text('not declared')
        with self.assertRaises(ValueError): self.check_candidate(path)

    def test_fixed_full_request_center_hash_without_large_model(self):
        import hashlib,struct
        from source_enclosure.format import compact
        cfg=protocol(); h=hashlib.sha256(b'torch.float64'+compact([1,3,32,32])+struct.pack('<d',.5)*3072)
        request={k:cfg[k] for k in ('label','top_k','radius','margin','clip')}
        request.update(experts=8,classes=10,training=False,gate='SELECTED_SOFTMAX',tie_policy='ANY_LEGAL_TOPK',
            center={'shape':[1,3,32,32],'dtype':'torch.float64','sha256':h.hexdigest()},
            model_state={'tensor_count':52,'parameter_count':6961368,'sha256':'0'*64})
        expected_request('full_size',request)
        request['center']['sha256']='0'*64
        with self.assertRaises(ValueError): expected_request('full_size',request)

    def test_limits_and_no_full_size_fault_modes(self):
        for changes in ({'budget':301},{'rss_limit':2**31+1},{'budget':True},{'rss_limit':True}):
            with self.assertRaises(ValueError): prepare(self.root/'unused',**changes)
        with self.assertRaises(ValueError):
            validate_spec({'case':'full_size','control':'partial','recipe_sha256':PROTOCOL_SHA})
        with self.assertRaises(FileExistsError): prepare(self.good)
        with self.assertRaises(ValueError):
            audit(self.good,self.observed,expected_sources={'unrelated':'0'*64})


if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(CapacityPreparationControls))
    path=Path(os.environ['H2_CAPACITY_CONTROL_ROOT'])
    save(path/'test_outcome.json',{'tests':result.testsRun,'success':result.wasSuccessful(),
        'failures':[str(t) for t,_ in result.failures],'errors':[str(t) for t,_ in result.errors],
        'skipped':[str(t) for t,_ in result.skipped],'full_size_models':0,'solves':0})
    raise SystemExit(0 if result.wasSuccessful() else 1)
