"""Fixed row-wise H2 portable/budget controls. Retain with HR_CONTROL_ROOT."""
from copy import deepcopy
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from scoped_proof.io import ROOT,PYTHON,load,save,sha
from scoped_proof.supervisor import group_rss
from scoped_source.rowwise_supervised import specification,supervise,audit,receive,POSITIVE,producer_sources,trusted_checker_sources
from scoped_source.rowwise_check import check
from scoped_source.factored_check import check as legacy_check
from source_enclosure.format import identity

BASE=ROOT/'data/moe/tmp'


def observe(root,spec,**kwargs):
    value=supervise(root,spec,**kwargs); save(root/'caller_observation.json',value); return value


class RowwiseSupervisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get('HR_CONTROL_ROOT'):
            cls.root=Path(os.environ['HR_CONTROL_ROOT']); cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp=tempfile.TemporaryDirectory(prefix='hf-supervision-',dir=BASE); cls.root=Path(cls.tmp.name)
        names=set(producer_sources(specification(intake='model')))|{
            'scoped_source/rowwise_supervision_tests.py','scripts/archive_h2_rowwise_supervision.py'}
        bindings={}
        for name in sorted(names):
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name]=sha(target)
        save(cls.root/'implementation.json',bindings)
        cls.normal={}
        for intake,case in [('declared',v) for v in ('weighted_sign','tied_partial_reuse','unsafe_tied','unresolved_sign')]+[('model','weighted_sign')]:
            for mode in ('endpoints','mccormick'):
                path=cls.root/(intake+'-'+case+'-'+mode); spec=specification(case,mode,intake)
                result=observe(path,spec,budget=30); cls.normal[(case,intake,mode)]=(path,result)
        cls.success,cls.observed=cls.normal[('weighted_sign','declared','endpoints')]
        if cls.observed['status']!=POSITIVE: raise AssertionError('initial control failed: '+str(cls.observed))
        cls.built=load(cls.success/'built.json')

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def verify(self,root,built=None,**changes):
        b={**(built or self.built),**changes}
        return subprocess.run([PYTHON,'-B','-I','-S',str(root/'verify.py'),
            '--manifest-sha',b['sha256'],'--source-sha',b['source_manifest_sha256'],
            '--proof-sha',b['proof_manifest_sha256'],'--mode',b['mode']],
            cwd=self.root,env=dict(os.environ,PYTHONPATH='/nonexistent'),text=True,capture_output=True,timeout=10)

    def copy(self,name,full=False,parent=None):
        root=self.root/name; shutil.copytree(parent or (self.success if full else self.success/'bundle'),root); return root

    def rebind(self,root):
        """Rehash newly altered control artifact, not authorize checker code."""
        m=load(root/'manifest.json')
        for name,ref in m['files'].items(): ref.update(sha256=sha(root/name),bytes=(root/name).stat().st_size)
        proof=load(root/'proof/manifest.json'); m['proof_manifest_sha256']=identity(proof)
        (root/'manifest.json').write_text(json.dumps(m,sort_keys=True))
        return {**self.built,'sha256':sha(root/'manifest.json'),'proof_manifest_sha256':identity(proof)}

    def test_normal_all_sources_and_intake_both_arms(self):
        for (case,intake,mode),(path,result) in self.normal.items():
            with self.subTest(case=case,intake=intake,mode=mode):
                expected='UNKNOWN_NONPOSITIVE' if case=='unsafe_tied' or (case,mode)==('weighted_sign','mccormick') else POSITIVE
                self.assertEqual(result['status'],expected); self.assertEqual(audit(path,result)['execution_status'],expected)
                self.assertFalse(audit(path)['positive_execution_accepted'])
                accepted=load(path/'accepted.json'); b=load(path/'built.json')
                from scoped_source.rowwise_bound import check_bound
                deadlines=[]; deadline=time.monotonic()+10
                def observe_bound(lp,cert,*,deadline):
                    deadlines.append(deadline)
                    return check_bound(lp,cert,deadline=deadline)
                with patch('scoped_source.rowwise_check.check_bound',side_effect=observe_bound):
                    core=check(path/'bundle/proof',expected_source_manifest=b['source_manifest_sha256'],
                        expected_proof_manifest=b['proof_manifest_sha256'],expected_mode=mode,deadline=deadline)
                self.assertEqual(core,accepted['result']); self.assertFalse(core['deployed_float_SAFE'])
                self.assertEqual(len(deadlines),core['lp_bounds_checked'])
                self.assertTrue(all(value==deadline for value in deadlines))
                legacy=legacy_check(path/'bundle/proof',expected_source_manifest=b['source_manifest_sha256'],
                    expected_proof_manifest=b['proof_manifest_sha256'],expected_mode=mode,deadline=time.monotonic()+10)
                self.assertEqual(core,legacy)
                if case=='tied_partial_reuse':
                    self.assertEqual((core['positive'],core['required']),(18,18))
                    self.assertEqual(core['lp_bounds_checked'],18 if mode=='endpoints' else 15)
                    self.assertEqual(core['origins'].count('SOURCE_BOX_REUSE'),3)
                events=[json.loads(l) for l in (path/'produce_events.jsonl').read_text().splitlines()]
                operations={e['operation'] for e in events}
                for name in ('source_chunk_construction_publication','proposal_imports','source_construct_and_propose','serialize_and_bundle'):
                    self.assertIn(name,operations)
                if intake=='model':
                    for name in ('model_creation_imports','capture_and_validate','capture_receipt_publication'): self.assertIn(name,operations)
                terminal=load(path/'terminal.json')
                self.assertGreater(result['seconds'],terminal['stage_seconds'])
                self.assertEqual(terminal['checker_stdout_sha256'],sha(path/'check.stdout'))

    def test_relocation_hides_old_bundle_and_isolation(self):
        root=self.copy('relocated'); original=self.success/'bundle'; hidden=self.success/'hidden'
        original.rename(hidden)
        try: result=self.verify(root)
        finally: hidden.rename(original)
        self.assertEqual(result.returncode,0,result.stderr); checked=json.loads(result.stdout)
        self.assertTrue(checked['isolated']); self.assertTrue(checked['site_disabled'])
        self.assertFalse(checked['solver_imported']); self.assertFalse(checked['producer_executed'])
        self.assertEqual(checked['result'],load(self.success/'accepted.json')['result'])
        for key,value in [('source_manifest_sha256','0'*64),('proof_manifest_sha256','0'*64),('mode','mccormick')]:
            self.assertNotEqual(self.verify(root,**{key:value}).returncode,0)

    def test_kernel_wiring_preserves_source_checker_and_absolute_deadline(self):
        from scoped_source.rowwise_verify import CODE
        from scoped_source.rowwise_supervised import protocol
        self.assertIn('scoped_source/rowwise_bound.py',CODE)
        self.assertNotIn('scoped_source/rowwise_native.py',CODE)
        self.assertIn('scoped_source/sparse_check.py',CODE)  # transitive old imports
        trees=[ast.parse((ROOT/name).read_text()) for name in (
            'scoped_source/factored_check.py','scoped_source/rowwise_check.py')]
        functions=[next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='check') for tree in trees]
        calls=[n for n in ast.walk(functions[1]) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='check_bound']
        self.assertEqual(len(calls),2)
        for call in calls:
            self.assertEqual(len(call.keywords),1)
            self.assertEqual(call.keywords[0].arg,'deadline')
            self.assertEqual(ast.dump(call.keywords[0].value),"Name(id='deadline', ctx=Load())")
            call.keywords=[]  # normalize only the intentional checker-call change
        self.assertEqual(ast.dump(functions[0]),ast.dump(functions[1]))
        worker=ast.parse((ROOT/'scoped_source/rowwise_worker.py').read_text())
        self.assertTrue(any(isinstance(n,ast.ImportFrom) and n.module=='scoped_source.rowwise_native' for n in ast.walk(worker)))
        calls=[n for n in ast.walk(worker) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='native']
        self.assertEqual(len(calls),1)
        self.assertEqual([(k.arg,ast.dump(k.value)) for k in calls[0].keywords],
                         [('deadline',"Name(id='deadline', ctx=Load())")])
        self.assertEqual(protocol()['maximum_budget_seconds'],300)
        self.assertEqual(protocol()['source_chunk_bytes'],64)
        save(self.root/'kernel_wiring.json',{'source_checker_body_equal_except_deadline':True,
            'dual_call_sites':2,'portable_native_import':False,'worker_absolute_deadline':True})

    def test_member_chunks_code_inventory_and_paths(self):
        names=['proof/source/t/00000-00000.bin','proof/b/000000.json','proof/p/000000.json',
               'code/scoped_source/rowwise_check.py','code/scoped_source/rowwise_bound.py','verify.py']
        for i,name in enumerate(names):
            root=self.copy('bytes-'+str(i))
            with (root/name).open('ab') as f: f.write(b'corrupt')
            self.assertNotEqual(self.verify(root).returncode,0)
        for kind in ('extra','missing','symlink','path','null','duplicate'):
            root=self.copy('inventory-'+kind); m=load(root/'manifest.json')
            if kind=='extra': (root/'extra.json').write_text('{}')
            elif kind=='missing': (root/'proof/p/000000.json').unlink()
            elif kind=='symlink':
                (root/'proof/p/000000.json').unlink(); (root/'proof/p/000000.json').symlink_to(self.success/'bundle/proof/p/000000.json')
            elif kind=='path': m['files']['../escape']={'bytes':2,'sha256':'0'*64}
            elif kind=='null': m['files']['proof/p/000000.json']['sha256']=None
            if kind in ('path','null'): (root/'manifest.json').write_text(json.dumps(m))
            if kind=='duplicate':
                raw=(root/'manifest.json').read_text(); (root/'manifest.json').write_text('{"mode":"endpoints",'+raw[1:])
            self.assertNotEqual(self.verify(root,sha256=sha(root/'manifest.json')).returncode,0)

    def test_rebound_math_mutations(self):
        for kind in ('gate','block','property','pair','endpoint','dual','reuse'):
            root=self.copy('math-'+kind); m=load(root/'proof/manifest.json'); ref=m['pairs'][0]
            part=load(root/'proof'/ref['file'])
            if kind=='gate': part['context']['gate']=['2','3']
            if kind=='property': part['duties'].pop()
            if kind=='pair': m['pairs'].pop()
            if kind=='endpoint': part['duties'][0]['endpoints'].pop()
            if kind=='dual': part['duties'][0]['endpoints'][0]['certificate']['claimed_lower_bound']='1000000000'
            if kind=='reuse': m['reuse_requested']=[[[0,1],99]]
            if kind=='block':
                blockref=m['blocks'][0]; block=load(root/'proof'/blockref['file']); block['bounds']=['0','0']
                (root/'proof'/blockref['file']).write_text(json.dumps(block))
                blockref.update(sha256=sha(root/'proof'/blockref['file']),bytes=(root/'proof'/blockref['file']).stat().st_size)
            (root/'proof'/ref['file']).write_text(json.dumps(part))
            ref.update(sha256=sha(root/'proof'/ref['file']),bytes=(root/'proof'/ref['file']).stat().st_size)
            (root/'proof/manifest.json').write_text(json.dumps(m))
            self.assertNotEqual(self.verify(root,self.rebind(root)).returncode,0,kind)

    def test_failures_and_partial_evidence(self):
        expected={'missing_certificate':'UNKNOWN_MISSING_EVIDENCE','missing_both':'UNKNOWN_MISSING_EVIDENCE',
            'proposal_exception':'UNKNOWN_MISSING_EVIDENCE'}
        for fault in ('missing_certificate','missing_both','missing_endpoint','omit_property','omit_pair','wrong_mode',
                      'proposal_exception','partial_output','exception_after_bundle','wrong_invocation'):
            root=self.root/('fault-'+fault); result=observe(root,specification(control=fault),budget=30)
            self.assertEqual(result['status'],expected.get(fault,'ERROR'),fault)
            self.assertFalse(audit(root,result)['positive_execution_accepted'])

    def test_deadlines_partial_and_full_cost(self):
        for fault in ('produce_delay','chunk_delay','construct_delay','proposal_delay','serialization_delay',
                      'check_delay','receive_delay','late_publish'):
            root=self.root/('deadline-'+fault); result=observe(root,specification(control=fault),budget=6)
            self.assertEqual(result['status'],'TIMEOUT',fault); self.assertFalse(audit(root,result)['positive_execution_accepted'])
            self.assertFalse((root/'accepted.json').exists()); self.assertGreater(result['seconds'],0)
            expected='unaccepted_partial.json' if fault=='late_publish' else fault.removesuffix('_delay')+'_partial.json'
            self.assertTrue((root/expected).exists(),fault+' must reach controlled phase')

    def test_model_capture_faults(self):
        for fault in ('mutate_model','mutate_input','capture_exception','capture_delay'):
            root=self.root/('model-'+fault); result=observe(root,specification(intake='model',control=fault),budget=6 if fault=='capture_delay' else 30)
            self.assertEqual(result['status'],'TIMEOUT' if fault=='capture_delay' else 'ERROR',fault)
            self.assertFalse(audit(root,result)['positive_execution_accepted'])
            self.assertFalse((root/'accepted.json').exists())

    def test_checker_context_rebinding_and_stdout_replacement(self):
        root=self.root/'rebound-checker'; result=observe(root,specification(control='rebind_checker_context'),budget=30)
        self.assertEqual(result['status'],'ERROR'); self.assertFalse((root/'check.stdout').exists())
        with self.assertRaisesRegex(ValueError,'invocation identity'): audit(root,result)
        root=self.root/'replaced-stdout'; result=observe(root,specification(mode='mccormick',control='rewrite_check_stdout'),budget=30)
        self.assertEqual(result['status'],'ERROR'); audit(root,result)
        self.assertNotEqual(load(root/'terminal.json')['checker_stdout_sha256'],sha(root/'check.stdout'))
        self.assertFalse((root/'accepted.json').exists())

    def test_receiver_aggregation_and_mandatory_hashes(self):
        for kind in ('minimum','positive','missing','count','origins','coverage'):
            root=self.copy('receive-'+kind,full=True); value=load(root/'check.stdout'); result=value['result']
            if kind=='minimum': result['duties'][0]['lower_bound']='100'
            if kind=='positive': result['duties'][0]['positive']=False
            if kind=='missing': result['missing']+=1
            if kind=='count': result['lp_bounds_checked']+=1
            if kind=='origins': result['origins'][0]='UNTRUSTED'
            if kind=='coverage': result['duties'].pop()
            (root/'check.stdout').write_text(json.dumps(value))
            with self.assertRaises(ValueError): receive(root,load(root/'spec.json'),load(root/'invocation.json')['invocation'],sha(root/'invocation.json'),sha(root/'check.stdout'))
        for digest in (None,'',0,'g'*64,'0'*63):
            call={**self.observed,'finish_sha256':digest}
            with self.assertRaises(ValueError): audit(self.success,call)
            with self.assertRaises(ValueError): receive(self.success,specification(),self.observed['invocation'],sha(self.success/'invocation.json'),digest)

    def test_owned_process_cleanup_and_rss(self):
        sentinel=subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            for fault in ('descendant','memory'):
                root=self.root/('resource-'+fault)
                result=observe(root,specification(control=fault),budget=6,rss_limit=64*2**20 if fault=='memory' else 2*2**30)
                self.assertEqual(result['status'],'RESOURCE_LIMIT' if fault=='memory' else 'ERROR'); audit(root,result)
                self.assertIsNone(sentinel.poll())
                for s in load(root/'terminal.json')['stages']:
                    if s['pid'] is not None: self.assertEqual(group_rss(s['pid'])[1],[])
        finally: sentinel.terminate(); sentinel.wait()

    def test_capture_receipt_binding(self):
        parent,_=self.normal[('weighted_sign','model','endpoints')]
        for key in ('source_sha256','request_sha256','model_state','center','capture_producer_sources'):
            root=self.copy('capture-binding-'+key,full=True,parent=parent)
            value=load(root/'model_intake.json'); value[key]=None
            (root/'model_intake.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'capture receipt'):
                receive(root,load(root/'spec.json'),load(root/'invocation.json')['invocation'],sha(root/'invocation.json'),sha(root/'check.stdout'))

    def test_mc_facts_not_counted_as_lp(self):
        parent,_=self.normal[('tied_partial_reuse','declared','mccormick')]
        for kind in ('origin','count'):
            root=self.copy('mc-'+kind,full=True,parent=parent); checked=load(root/'check.stdout'); result=checked['result']
            if kind=='origin': result['origins'][result['origins'].index('SOURCE_BOX_REUSE')]='PROPOSED'
            else: result['lp_bounds_checked']+=3
            (root/'check.stdout').write_text(json.dumps(checked))
            with self.assertRaisesRegex(ValueError,'aggregation'):
                receive(root,load(root/'spec.json'),load(root/'invocation.json')['invocation'],sha(root/'invocation.json'),sha(root/'check.stdout'))

    def test_portable_read_policy(self):
        root=self.copy('read-policy')
        for expression in (f"open({str(ROOT/'AGENTS.md')!r}).read()",'import torch','import scoped_source.rowwise_native',
                           f"open({str(root/'new.txt')!r},'w')",'import subprocess; subprocess.run(["true"])'):
            code=f"import runpy,sys; ns=runpy.run_path({str(root/'verify.py')!r}); sys.addaudithook(ns['guard']); "+expression
            out=subprocess.run([PYTHON,'-B','-I','-S','-c',code],cwd=self.root,capture_output=True,text=True,timeout=10)
            self.assertNotEqual(out.returncode,0,expression)
            self.assertRegex(out.stderr,'PermissionError|ImportError')

    def test_midcheck_outer_change_rejected(self):
        from scoped_source.rowwise_verify import verify
        root=self.copy('midcheck-inventory')
        def changed(*args,**kwargs):
            result=check(*args,**kwargs); (root/'late-extra.json').write_text('{}'); return result
        with patch('scoped_source.rowwise_check.check',side_effect=changed):
            with self.assertRaisesRegex(ValueError,'inventory'):
                verify(root,self.built['sha256'],self.built['source_manifest_sha256'],
                       self.built['proof_manifest_sha256'],'endpoints',time.monotonic()+10)

    def test_final_publication_and_prelaunch_deadline(self):
        def slow(path,value):
            result=save(path,value)
            if path.name=='finish.json': time.sleep(6)
            return result
        root=self.root/'final-publication'
        with patch('scoped_source.rowwise_supervised.save',side_effect=slow):
            result=observe(root,specification(),budget=5)
        self.assertEqual(result['status'],'TIMEOUT'); self.assertTrue((root/'publication_timeout.json').exists())
        terminal=load(root/'terminal.json')
        self.assertEqual(terminal['status_before_publication'],POSITIVE)
        self.assertEqual([s['status'] for s in terminal['stages']],['COMPLETED']*3)
        self.assertTrue(load(root/'receipt.json')['complete_declared_source_proof'])
        self.assertFalse(audit(root,result)['positive_execution_accepted']); self.assertGreater(result['seconds'],5)
        root=self.root/'prelaunch'; result=observe(root,specification(),budget=1e-9)
        self.assertEqual(result['status'],'TIMEOUT'); audit(root,result)
        self.assertIsNone(load(root/'terminal.json')['stages'][0]['pid'])

    def test_archive_requires_complete_registered_calls(self):
        from scripts.archive_h2_rowwise_supervision import declared_calls,roster
        self.assertEqual(len(roster()),38)
        root=self.root/'incomplete-archive'; root.mkdir()
        with self.assertRaisesRegex(ValueError,'missing registered call artifact'):
            declared_calls(root)
        shutil.copytree(self.success,root/self.success.name)
        with self.assertRaisesRegex(ValueError,'registered call identity/outcome'):
            declared_calls(root)

    def test_cost_chain_and_receipt_tampering(self):
        for name,key in [('terminal','stage_seconds'),('receipt','budget'),('invocation','deadline')]:
            root=self.copy('cost-'+name,full=True); value=load(root/(name+'.json')); value[key]=-1
            (root/(name+'.json')).write_text(json.dumps(value))
            with self.assertRaises(ValueError): audit(root)
        with self.assertRaises(ValueError): audit(self.success,{**self.observed,'seconds':0})
        with self.assertRaisesRegex(ValueError,'caller proof claim'):
            audit(self.success,{**self.observed,'complete_declared_source_proof':False})
        root=self.copy('late-marker',full=True); save(root/'publication_timeout.json',{'status':'TIMEOUT'})
        self.assertEqual(audit(root)['execution_status'],'TIMEOUT')


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(RowwiseSupervisionControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    root=os.environ.get('HR_CONTROL_ROOT')
    if root and Path(root).exists():
        save(Path(root)/'test_outcome.json',{'tests':result.testsRun,'success':result.wasSuccessful(),
            'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],
            'skipped':[(str(t),s) for t,s in result.skipped],'real_requests':0})
    raise SystemExit(not result.wasSuccessful())
