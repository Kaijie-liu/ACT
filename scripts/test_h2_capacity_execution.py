"""Controls for separately versioned full-path capacity, never real requests."""
import ast
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
from source_enclosure.format import identity
from scripts.h2_capacity_supervised import specification,validate_spec,supervise,audit,receive,POSITIVE,producer_sources
from scripts.archive_h2_capacity_execution import roster,declared_calls,TEST_NAMES,check_control_bindings


def observe(root,spec,**kwargs):
    result=supervise(root,spec,**kwargs); save(root/'caller_observation.json',result); return result


class StripObserver(ast.NodeTransformer):
    def visit_Expr(self,node):
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) and node.value.func.id=='_observe': return None
        return self.generic_visit(node)
    def visit_Assign(self,node):
        if any(isinstance(t,ast.Name) and t.id=='_observe' for t in node.targets): return None
        return self.generic_visit(node)
    def visit_FunctionDef(self,node):
        if node.name in ('build','propose') and node.args.kwonlyargs and node.args.kwonlyargs[-1].arg=='observer':
            node.args.kwonlyargs.pop(); node.args.kw_defaults.pop()
        return self.generic_visit(node)


class CapacityControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['H2_CAPACITY_CONTROL_ROOT']); cls.root.mkdir(parents=True,exist_ok=False)
        names=set(producer_sources(specification()))|{'scripts/test_h2_capacity_execution.py','scripts/archive_h2_capacity_execution.py'}
        bindings={}
        for name in sorted(names):
            target=cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name]=sha(target)
        save(cls.root/'implementation.json',bindings); cls.calls={}
        for mode in ('endpoints','mccormick'):
            cls.run_call('normal-'+mode)
        cls.success=cls.root/'normal-endpoints'; cls.observed=cls.calls['normal-endpoints']
        if cls.observed['status']!=POSITIVE: raise AssertionError('initial tiny capacity call: '+str(cls.observed))
        cls.built=load(cls.success/'built.json')

    @classmethod
    def run_call(cls,name):
        spec,status,budget,rss=roster()[name]
        value=observe(cls.root/name,spec,budget=budget,rss_limit=rss); cls.calls[name]=value
        return value

    def copy(self,name,full=True,parent=None):
        dest=self.root/name; shutil.copytree(parent or (self.success if full else self.success/'bundle'),dest); return dest

    def verify(self,root,**changes):
        b={**self.built,**changes}
        return subprocess.run([PYTHON,'-B','-I','-S',str(root/'verify.py'),'--manifest-sha',b['sha256'],
            '--source-sha',b['source_manifest_sha256'],'--proof-sha',b['proof_manifest_sha256'],
            '--mode',b['mode']],cwd=self.root,env=dict(os.environ,PYTHONPATH='/nonexistent'),
            capture_output=True,text=True,timeout=15)

    def test_normal_both_arms_and_charged_capture(self):
        for mode,positive in (('endpoints',3),('mccormick',2)):
            path=self.root/('normal-'+mode); value=self.calls[path.name]
            checked=audit(path,value); accepted=load(path/'accepted.json')
            self.assertTrue(checked['pipeline_complete']); self.assertTrue(checked['all_obligations_checked'])
            self.assertEqual((accepted['positive'],accepted['required']),(positive,3))
            self.assertFalse(checked['real_model_admitted']); self.assertFalse(audit(path)['positive_execution_accepted'])
            events=[json.loads(l) for l in (path/'produce_events.jsonl').read_text().splitlines()]
            operations={e['operation'] for e in events}
            for op in ('model_creation_imports','capture_and_validate','source_chunk_construction_publication',
                       'source_begin','pair_assembled','property_begin','native_call_boundary',
                       'candidate_independent_bound_complete','serialize_and_bundle'):
                self.assertIn(op,operations)
            self.assertGreater(value['seconds'],load(path/'terminal.json')['stage_seconds'])

    def test_observation_only_ast_and_aggregation(self):
        for old,new,name in [('scoped_source/factored_build.py','scripts/h2_capacity_build.py','build'),
                             ('scoped_source/rowwise_native.py','scripts/h2_capacity_native.py','propose')]:
            left=ast.parse((ROOT/old).read_text()); right=StripObserver().visit(ast.parse((ROOT/new).read_text()))
            fn=lambda tree: next(v for v in tree.body if isinstance(v,ast.FunctionDef) and v.name==name)
            self.assertEqual(ast.dump(fn(left)),ast.dump(fn(right)))
        def aggregation(path):
            raw=(ROOT/path).read_text(); start=raw.index('    missing=checked_lps=0'); end=raw.index('    accepted=',start)
            return ast.dump(ast.parse('def aggregate():\n'+raw[start:end]))
        self.assertEqual(aggregation('scoped_source/rowwise_supervised.py'),aggregation('scripts/h2_capacity_supervised.py'))

    def test_old_new_builder_differential_all_tiny_sources(self):
        from scoped_source.endpoint_source_controls import cases
        from scoped_source.factored_source import pack
        from scoped_source.factored_build import build as old
        from scripts.h2_capacity_build import build as new
        for case,doc,reuse in cases():
            for mode in ('endpoints','mccormick'):
                results=[]
                for label,build in (('old',old),('new',new)):
                    root=self.root/(case+'-'+mode+'-'+label); root.mkdir(); end=time.monotonic()+30
                    source=pack(doc,root/'source',lambda:None,2**20)
                    digest=build(root,expected_source_manifest=source,mode=mode,reuse_keys=reuse,deadline=end)
                    results.append((digest,{str(p.relative_to(root)):sha(p) for p in root.rglob('*') if p.is_file()}))
                self.assertEqual(results[0],results[1],(case,mode))
        save(self.root/'differential.json',{'cases':4,'arms':2,'matrices_and_manifests_identical':True,'new_solves':0})

    def test_faults_partial_and_missing_evidence(self):
        for name,(spec,status,_,_) in roster().items():
            if not name.startswith('fault-'): continue
            result=self.run_call(name); self.assertEqual(result['status'],status,name)
            checked=audit(self.root/name,result); self.assertFalse(checked['positive_execution_accepted'])
            if spec['control']=='missing_stdout':
                t=load(self.root/name/'terminal.json')
                self.assertEqual([s['phase'] for s in t['stages']],['produce','check'])
                self.assertIn('output_reception_error',t['stages'][1]); self.assertGreater(t['stages'][1]['seconds'],0)
            if status=='UNKNOWN_MISSING_EVIDENCE':
                self.assertTrue(checked['pipeline_complete']); self.assertFalse(checked['all_obligations_checked'])

    def test_deadlines_reached_and_no_late_acceptance(self):
        for name,(spec,status,_,_) in roster().items():
            if not name.startswith('deadline-'): continue
            result=self.run_call(name); self.assertEqual(result['status'],status,name); path=self.root/name
            checked=audit(path,result); self.assertFalse(checked['pipeline_complete'])
            partial='unaccepted_partial.json' if spec['control']=='late_publish' else spec['control'].removesuffix('_delay')+'_partial.json'
            self.assertTrue((path/partial).exists(),name+' must reach target stage')
            self.assertFalse((path/'accepted.json').exists())

    def test_owned_cleanup_and_memory(self):
        sentinel=subprocess.Popen([PYTHON,'-B','-c','import time; time.sleep(30)'],start_new_session=True)
        try:
            for name in ('resource-descendant','resource-memory'):
                result=self.run_call(name); self.assertEqual(result['status'],roster()[name][1]); audit(self.root/name,result)
                self.assertIsNone(sentinel.poll())
                for s in load(self.root/name/'terminal.json')['stages']:
                    if s['pid'] is not None: self.assertEqual(group_rss(s['pid'])[1],[])
        finally: sentinel.terminate(); sentinel.wait()

    def test_checker_and_stdout_rebinding(self):
        result=self.run_call('rebound-checker'); self.assertEqual(result['status'],'ERROR')
        with self.assertRaisesRegex(ValueError,'invocation identity'): audit(self.root/'rebound-checker',result)
        result=self.run_call('replaced-stdout'); self.assertEqual(result['status'],'ERROR'); audit(self.root/'replaced-stdout',result)
        self.assertFalse((self.root/'replaced-stdout/accepted.json').exists())

    def test_relocation_and_inventory(self):
        root=self.copy('relocated',full=False); original=self.success/'bundle'; hidden=self.success/'hidden'
        original.rename(hidden)
        try: result=self.verify(root)
        finally: hidden.rename(original)
        self.assertEqual(result.returncode,0,result.stderr); value=json.loads(result.stdout)
        self.assertTrue(value['isolated'] and value['site_disabled']); self.assertFalse(value['solver_imported'])
        self.assertEqual(value['result'],load(self.success/'accepted.json')['result'])
        for key in ('source_manifest_sha256','proof_manifest_sha256'):
            self.assertNotEqual(self.verify(root,**{key:'0'*64}).returncode,0)
        for kind in ('extra','missing','code','null'):
            target=self.copy('portable-'+kind,full=False)
            if kind=='extra': save(target/'extra.json',{})
            elif kind=='missing': (target/'proof/p/000000.json').unlink()
            elif kind=='code': (target/'code/scoped_source/rowwise_check.py').write_text('pass')
            else:
                value=load(target/'manifest.json'); value['files']['proof/p/000000.json']['sha256']=None
                (target/'manifest.json').write_text(json.dumps(value))
            self.assertNotEqual(self.verify(target,sha256=sha(target/'manifest.json')).returncode,0)

    def test_receiver_aggregation_and_capture_binding(self):
        for kind in ('coverage','minimum','positive','missing','count','origins'):
            root=self.copy('receiver-'+kind); checked=load(root/'check.stdout'); r=checked['result']
            if kind=='coverage': r['duties'].pop()
            elif kind=='minimum': r['duties'][0]['lower_bound']='999'
            elif kind=='positive': r['duties'][0]['positive']=False
            elif kind=='missing': r['missing']+=1
            elif kind=='count': r['lp_bounds_checked']+=1
            else: r['origins'][0]='UNTRUSTED'
            (root/'check.stdout').write_text(json.dumps(checked))
            with self.assertRaises(ValueError): receive(root,load(root/'spec.json'),self.observed['invocation'],sha(root/'invocation.json'),sha(root/'check.stdout'))
        for key in ('request','declared_source_sha256','source_manifest_sha256','producer_sources_sha256','invocation'):
            root=self.copy('capture-'+key); value=load(root/'model_intake.json'); value[key]=None
            (root/'model_intake.json').write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'capture receipt'):
                receive(root,load(root/'spec.json'),self.observed['invocation'],sha(root/'invocation.json'),sha(root/'check.stdout'))

    def test_fixed_admission_and_mandatory_hashes(self):
        for key,value in [('case','real'),('source_manifest_sha256','0'*64),('reuse',[[[0,1],1]]),('request',{})]:
            spec=specification(); spec[key]=value
            with self.assertRaises(ValueError): validate_spec(spec)
        for bad in (None,'','g'*64,'0'*63):
            with self.assertRaises(ValueError): audit(self.success,{**self.observed,'finish_sha256':bad})
            with self.assertRaises(ValueError): receive(self.success,specification(),self.observed['invocation'],sha(self.success/'invocation.json'),bad)
        for budget,rss in ((301,2*2**30),(300,3*2**30),(20,2*2**30),(300,2**30)):
            with self.assertRaises(ValueError): supervise(self.root/'must-not-create',specification('full_size'),budget=budget,rss_limit=rss)
        self.assertFalse((self.root/'must-not-create').exists())
        with self.assertRaises(ValueError): specification('full_size',control='capture_delay')

    def test_final_publication_and_prelaunch(self):
        def slow(path,value):
            result=save(path,value)
            if path.name=='finish.json': time.sleep(11)
            return result
        with patch('scripts.h2_capacity_supervised.save',side_effect=slow): result=self.run_call('final-publication')
        self.assertEqual(result['status'],'TIMEOUT'); path=self.root/'final-publication'
        self.assertEqual(load(path/'terminal.json')['status_before_publication'],POSITIVE)
        self.assertFalse(audit(path,result)['positive_execution_accepted'])
        result=self.run_call('prelaunch'); self.assertEqual(result['status'],'TIMEOUT'); audit(self.root/'prelaunch',result)
        self.assertIsNone(load(self.root/'prelaunch/terminal.json')['stages'][0]['pid'])

    def test_cost_and_hash_chain_tampering(self):
        for name,key in (('terminal','stage_seconds'),('receipt','budget'),('invocation','deadline')):
            root=self.copy('cost-'+name); value=load(root/(name+'.json')); value[key]=-1
            (root/(name+'.json')).write_text(json.dumps(value))
            with self.assertRaises(ValueError): audit(root)
        for name,key in (('finish','receipt_sha256'),('receipt','terminal_sha256'),('terminal','invocation_sha256')):
            root=self.copy('null-'+name); value=load(root/(name+'.json')); value[key]=None
            (root/(name+'.json')).write_text(json.dumps(value))
            # Rebind only outer test-chain links, so null parsing itself is tested.
            if name=='terminal':
                v=load(root/'receipt.json'); v['terminal_sha256']=sha(root/'terminal.json'); (root/'receipt.json').write_text(json.dumps(v))
            if name in ('terminal','receipt'):
                v=load(root/'finish.json'); v['receipt_sha256']=sha(root/'receipt.json'); (root/'finish.json').write_text(json.dumps(v))
            with self.assertRaises(ValueError): audit(root)
        with self.assertRaises(ValueError): audit(self.success,{**self.observed,'seconds':0})
        with self.assertRaises(ValueError): audit(self.success,{**self.observed,'status':'TIMEOUT','complete_declared_source_proof':False})

    def test_archive_missing_calls_rejected(self):
        empty=self.root/'empty-archive'; empty.mkdir()
        with self.assertRaisesRegex(ValueError,'missing registered call'): declared_calls(empty)

    def rechain(self,root):
        terminal=load(root/'terminal.json'); terminal['invocation_sha256']=sha(root/'invocation.json')
        (root/'terminal.json').write_text(json.dumps(terminal))
        receipt=load(root/'receipt.json'); receipt['terminal_sha256']=sha(root/'terminal.json')
        (root/'receipt.json').write_text(json.dumps(receipt))
        finish=load(root/'finish.json'); finish['receipt_sha256']=sha(root/'receipt.json')
        (root/'finish.json').write_text(json.dumps(finish))

    def test_cost_semantics_after_rebinding(self):
        for name,key in (('terminal','stage_seconds'),('receipt','budget'),('invocation','deadline'),('invocation','rss_limit')):
            root=self.copy('semantic-'+name+'-'+key); v=load(root/(name+'.json')); v[key]=-1
            (root/(name+'.json')).write_text(json.dumps(v)); self.rechain(root)
            with self.assertRaisesRegex(ValueError,'cost|budget|identity|resource'): audit(root)

    def test_stage_semantics_after_rebinding(self):
        for key,value in [('cleanup_included',False),('deadline_monotonic',0),('returncode',5),
                          ('sampled_peak_rss',-1),('seconds',999),('status','TIMEOUT')]:
            root=self.copy('stage-'+key); term=load(root/'terminal.json'); stage=term['stages'][0]; stage[key]=value
            (root/'produce_stage.json').write_text(json.dumps(stage)); (root/'terminal.json').write_text(json.dumps(term))
            self.rechain(root)
            with self.assertRaises(ValueError): audit(root)

    def test_startup_exception_accounted(self):
        with patch('scoped_proof.supervisor.subprocess.Popen',side_effect=OSError('controlled process launch failure')):
            result=self.run_call('startup-error')
        self.assertEqual(result['status'],'ERROR'); path=self.root/'startup-error'
        stage=load(path/'terminal.json')['stages'][0]
        self.assertIsNone(stage['pid']); self.assertGreater(stage['seconds'],0)
        self.assertTrue(stage['cleanup_included']); self.assertFalse(audit(path,result)['pipeline_complete'])

    def test_archive_complete_inventory(self):
        self.assertEqual(sorted(unittest.defaultTestLoader.getTestCaseNames(CapacityControls)),sorted(TEST_NAMES))
        for kind in ('empty_sources','missing_test','missing_source'):
            root=self.root/('archive-'+kind); root.mkdir()
            outcome={'tests':len(TEST_NAMES),'test_names':sorted(TEST_NAMES),'success':True,'failures':[],'errors':[],'skipped':[]}
            sources=load(self.root/'implementation.json')
            if kind=='empty_sources': sources={}
            if kind=='missing_source': sources.pop('scripts/h2_capacity_worker.py')
            if kind=='missing_test': outcome['test_names'].pop()
            save(root/'implementation.json',sources); save(root/'test_outcome.json',outcome)
            with self.assertRaisesRegex(ValueError,'inventory|controls'): check_control_bindings(root)


if __name__=='__main__':
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(CapacityControls)
    result=unittest.TextTestRunner(verbosity=2).run(suite)
    root=Path(os.environ['H2_CAPACITY_CONTROL_ROOT'])
    if root.exists(): save(root/'test_outcome.json',{'tests':result.testsRun,'success':result.wasSuccessful(),
        'test_names':sorted(unittest.defaultTestLoader.getTestCaseNames(CapacityControls)),
        'failures':[(str(t),s) for t,s in result.failures],'errors':[(str(t),s) for t,s in result.errors],
        'skipped':[(str(t),s) for t,s in result.skipped],'real_requests':0})
    raise SystemExit(not result.wasSuccessful())
