"""Small synthetic process/bundle controls, no real input or production change."""
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch
from fractions import Fraction
from act.back_end.solver.lp_certificate import propose
from scoped_proof.io import PYTHON, load, save, sha
from scoped_proof.supervisor import group_rss
from scoped_source.sparse_controls import source
from scoped_source.sparse_build import build
from scoped_source.sparse_portable import pack
from scoped_source.sparse_supervised import supervise, audit, POSITIVE
from source_enclosure.format import identity

BASE = Path('/data1/Kane/MOE/ACT/data/moe/tmp')


def observed_run(root, spec, **kwargs):
    result = supervise(root,spec,**kwargs)
    # Test driver's observation happens only after the full API has returned.
    # Its administrative serialization is not an extra proof stage.
    save(root/'caller_observation.json',result)
    return result


def specification(**changes):
    options = changes.pop('fixture', {'relational':True})
    return {'schema':'H1_SYNTHETIC_SUPERVISION_V1','fixture':options,
            'source_sha256':identity(source(**options)), 'mode':'dependency','reuse':[],
            'control':'', **changes}


class H1SupervisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keep = os.environ.get('H1_CONTROL_ROOT')
        if keep:
            cls.root = Path(keep); cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp = tempfile.TemporaryDirectory(prefix='h1-supervision-',dir=BASE)
            cls.root = Path(cls.tmp.name)
        repo = Path(__file__).resolve().parents[1]
        code = list((repo/'scoped_source').glob('sparse_*.py'))
        code += [repo/n for n in ('scoped_source/graph.py','scoped_proof/supervisor.py',
                 'scoped_proof/io.py','source_enclosure/format.py','router_source/checker.py',
                 'upstream_source/checker.py','act/back_end/solver/lp_certificate.py',
                 'act/back_end/solver/sparse_lp_certificate.py')]
        bindings = {}
        for file in code:
            name = str(file.relative_to(repo)); target = cls.root/'implementation'/name
            target.parent.mkdir(parents=True,exist_ok=True); shutil.copyfile(file,target)
            bindings[name] = sha(target)
        save(cls.root/'implementation.json',bindings)
        cls.doc = source(relational=True); cls.source_hash = identity(cls.doc)
        proof = build(cls.doc,expected_source_sha256=cls.source_hash,deadline=time.monotonic()+300,proposer=propose)
        cls.original = cls.root/'original'
        cls.bundle = pack(cls.original,cls.doc,proof,cls.source_hash,'portable-control',time.monotonic()+300)
        cls.success = cls.root/'success'
        cls.run_result = observed_run(cls.success,specification(),budget=30)

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def copy_bundle(self, name):
        root = self.root/name; shutil.copytree(self.original,root); return root

    def verify_bundle(self, root, manifest=None, **kwargs):
        return subprocess.run([PYTHON,'-B','-I','-S',str(root/'verify.py'),
            '--manifest-sha',manifest or self.bundle['sha256'],'--source-sha',self.source_hash],
            cwd=self.root,env=dict(os.environ,PYTHONPATH='/does/not/exist'),
            capture_output=True,text=True,timeout=10,**kwargs)

    def rebind(self, root, name, data):
        (root/name).write_text(json.dumps(data,sort_keys=True))
        manifest = load(root/'manifest.json'); manifest['files'][name] = sha(root/name)
        (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
        return sha(root/'manifest.json')

    def test_complete_supervised_proof_and_charged_cost(self):
        self.assertEqual(self.run_result['status'],POSITIVE)
        result = audit(self.success,self.run_result); self.assertTrue(result['positive_execution_accepted'])
        self.assertFalse(audit(self.success)['positive_execution_accepted'])
        accepted = load(self.success/'accepted.json')
        self.assertEqual((accepted['required'],accepted['positive']), (6,6))
        terminal = load(self.success/'terminal.json')
        self.assertEqual([r['phase'] for r in terminal['stages']], ['produce','check','receive'])
        self.assertGreater(terminal['stage_seconds'], 0)
        self.assertGreater(result['seconds_before_finish_marker'], terminal['stage_seconds'])
        self.assertFalse(accepted['result']['deployed_float_SAFE'])

    def test_relocation_without_original_or_repository_cwd(self):
        root = self.copy_bundle('moved')
        hidden = self.root/'original-hidden'; self.original.rename(hidden)
        try:
            result = self.verify_bundle(root)
        finally: hidden.rename(self.original)
        self.assertEqual(result.returncode,0,result.stderr)
        checked = json.loads(result.stdout)
        self.assertEqual(checked['result']['status'],POSITIVE)
        self.assertFalse(checked['solver_imported'])

    def test_file_tampering_rejected_before_import(self):
        for i,name in enumerate(['source.json','proof.json','code/scoped_source/sparse_check.py']):
            root = self.copy_bundle('tamper'+str(i))
            with (root/name).open('a') as stream: stream.write('\n# mutation')
            result = self.verify_bundle(root)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('bundle member identity',result.stderr)

    def test_substituted_checker_rebinding_cannot_cross_supervisor(self):
        from scoped_source.sparse_supervised import bind_checker
        root = self.root/'fake-checker'; shutil.copytree(self.success,root)
        name = 'code/scoped_source/sparse_check.py'
        path = root/'bundle'/name
        path.write_text('def check(*a,**k): return {"status":"CHECKED_DECLARED_SOURCE_POSITIVE"}')
        manifest = load(root/'bundle/manifest.json'); manifest['files'][name] = sha(path)
        (root/'bundle/manifest.json').write_text(json.dumps(manifest))
        built = load(root/'built.json'); built['sha256'] = sha(root/'bundle/manifest.json')
        (root/'built.json').write_text(json.dumps(built))
        with self.assertRaisesRegex(ValueError,'trusted mathematical checker'):
            bind_checker(root,manifest,load(root/'invocation.json')['checker_sources'])

    def test_full_and_dependency_supervised_bounds_agree(self):
        root = self.root/'full'
        result = observed_run(root,specification(mode='full'),budget=30)
        self.assertEqual(result['status'],POSITIVE); audit(root)
        a = load(self.success/'accepted.json')['result']; b = load(root/'accepted.json')['result']
        self.assertEqual([r['lower_bound'] for r in a['obligations']], [r['lower_bound'] for r in b['obligations']])
        self.assertLess(a['source_blocks_checked'],b['source_blocks_checked'])

    def test_ties_dimension_change_and_partial_reuse_supervised(self):
        root = self.root/'ties'
        spec = specification(fixture={'experts':4,'classes':4,'width':2,'tied':True,'constant':True},reuse=[[[0,1],1]])
        result = observed_run(root,spec,budget=30)
        self.assertEqual(result['status'],POSITIVE); audit(root)
        checked = load(root/'accepted.json')['result']
        self.assertEqual((checked['required'],checked['positive'],checked['reused']), (18,18,1))

    def test_math_mutations_rejected_even_after_file_rebinding(self):
        for i,kind in enumerate(['missing','range','wrong_lp']):
            root = self.copy_bundle('math'+str(i)); proof = load(root/'proof.json')
            if kind == 'missing': proof['obligations'].pop()
            elif kind == 'range': next(iter(proof['bank'].values()))['bounds'] = ['0','0']
            else: proof['obligations'][0]['certificate']['lp_sha256'] = '0'*64
            digest = self.rebind(root,'proof.json',proof)
            self.assertNotEqual(self.verify_bundle(root,digest).returncode,0)

    def test_extra_file_symlink_and_duplicate_json_rejected(self):
        for i,kind in enumerate(['extra','symlink','duplicate']):
            root = self.copy_bundle('inventory'+str(i))
            digest = self.bundle['sha256']
            if kind == 'extra': (root/'code/local_check.py').write_text('raise Exception("must not import")')
            elif kind == 'symlink': (root/'external').symlink_to(self.root/'original')
            else:
                path = root/'manifest.json'; raw = path.read_text()
                path.write_text(raw[:-1]+',"source_sha256":"duplicate"}')
                digest = sha(path)
            self.assertNotEqual(self.verify_bundle(root,digest).returncode,0)

    def test_read_restriction_network_and_import_controls(self):
        path = self.original/'verify.py'
        for expression in ["open('/etc/hosts').read()", "open(str(m['ROOT']/'forbidden'),'w')",
                           "__import__('scipy')", "__import__('socket').socket()",
                           "__import__('os').posix_spawn('/bin/true',['/bin/true'],{})"]:
            code = f"import runpy,sys; m=runpy.run_path({str(path)!r}); sys.addaudithook(m['guard']); {expression}"
            result = subprocess.run([PYTHON,'-B','-I','-S','-c',code],capture_output=True,text=True,timeout=5)
            self.assertNotEqual(result.returncode,0,expression)

    def test_partial_certificate_is_unknown_with_complete_roster(self):
        root = self.root/'missing-cert'
        result = observed_run(root,specification(control='missing_certificate'),budget=30)
        self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE')
        self.assertEqual(load(root/'accepted.json')['required'],6)
        self.assertFalse(audit(root)['complete_declared_source_proof'])

    def test_negative_control_not_unsafe(self):
        root = self.root/'negative'
        result = observed_run(root,specification(fixture={'unsafe':True,'tied':True}),budget=30)
        self.assertEqual(result['status'],'UNKNOWN_NONPOSITIVE')
        audit(root)

    def test_source_mismatch_and_no_overwrite(self):
        with self.assertRaises(FileExistsError): supervise(self.success,specification())
        root = self.root/'source-mismatch'
        result = observed_run(root,specification(source_sha256='0'*64),budget=5)
        self.assertEqual(result['status'],'ERROR'); audit(root)

    def test_missing_obligation_exception_and_partial_stdout(self):
        for fault in ['omit_property','exception_after_bundle','partial_output','wrong_invocation']:
            root = self.root/fault; result = observed_run(root,specification(control=fault),budget=30)
            self.assertEqual(result['status'],'ERROR',fault)
            self.assertFalse(audit(root)['complete_declared_source_proof'])
            self.assertTrue((root/'bundle/proof.json').exists())

    def test_hard_deadline_each_phase_and_late_candidate(self):
        for fault,phase,budget in [('produce_delay','produce',.4),('check_delay','check',6),
                                   ('receive_delay','receive',6),('late_publish','receive',6)]:
            root = self.root/fault; result = observed_run(root,specification(control=fault),budget=budget)
            self.assertEqual(result['status'],'TIMEOUT',fault)
            self.assertEqual(load(root/'terminal.json')['stages'][-1]['phase'],phase)
            self.assertFalse((root/'accepted.json').exists())
            self.assertFalse(audit(root)['complete_declared_source_proof'])

    def test_resource_limit_and_owned_descendants_only(self):
        root = self.root/'rss'
        result = observed_run(root,specification(control='memory'),budget=5,rss_limit=32*2**20)
        self.assertEqual(result['status'],'RESOURCE_LIMIT'); audit(root)
        sentinel = subprocess.Popen([PYTHON,'-B','-c','import time;time.sleep(20)'],start_new_session=True)
        try:
            root = self.root/'descendant'
            result = observed_run(root,specification(control='descendant'),budget=5)
            self.assertEqual(result['status'],'ERROR')
            pid = load(root/'terminal.json')['stages'][0]['pid']
            for _ in range(20):
                if not group_rss(pid)[1]: break
                time.sleep(.01)
            self.assertFalse(group_rss(pid)[1]); self.assertIsNone(sentinel.poll())
            audit(root)
        finally: sentinel.terminate(); sentinel.wait()

    def test_audit_rejects_corrupted_cost_or_checked_artifact(self):
        for kind in ('cost','proof'):
            root = self.root/('audit-'+kind); shutil.copytree(self.success,root)
            if kind == 'cost':
                terminal = load(root/'terminal.json'); terminal['stage_seconds'] = -1
                (root/'terminal.json').write_text(json.dumps(terminal))
            else: (root/'bundle/proof.json').write_text('{}')
            with self.assertRaises(ValueError): audit(root)

    def test_late_publication_marker_dominates_positive(self):
        root = self.root/'late-marker'; shutil.copytree(self.success,root)
        save(root/'publication_timeout.json', {'status':'TIMEOUT','seconds':301})
        self.assertEqual(audit(root)['execution_status'],'TIMEOUT')
        self.assertFalse(audit(root)['complete_declared_source_proof'])

    def test_actual_final_publication_overrun_is_not_accepted(self):
        import scoped_source.sparse_supervised as module
        real_save = module.save
        def slow(path, value):
            result = real_save(path,value)
            if path.name == 'finish.json': time.sleep(5)
            return result
        root = self.root/'actual-publication-overrun'
        with patch.object(module,'save',side_effect=slow):
            result = observed_run(root,specification(),budget=5)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertGreaterEqual(result['seconds'],5)
        self.assertTrue((root/'publication_timeout.json').exists())
        self.assertFalse(audit(root,result)['positive_execution_accepted'])

    def test_prelaunch_expiry_and_rebound_budget_mutation(self):
        root = self.root/'prelaunch-expiry'
        result = observed_run(root,specification(),budget=1e-9)
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertIsNone(load(root/'terminal.json')['stages'][0]['pid'])
        audit(root,result)
        changed = self.root/'changed-budget'; shutil.copytree(self.success,changed)
        inv = load(changed/'invocation.json'); inv['budget'] = 301
        (changed/'invocation.json').write_text(json.dumps(inv))
        with self.assertRaises(ValueError): audit(changed)

    def test_wrong_mode_rejected_by_receiver(self):
        from scoped_source.sparse_supervised import receive
        root = self.root/'wrong-mode'; shutil.copytree(self.success,root)
        proof = load(root/'bundle/proof.json'); proof['mode'] = 'full'
        digest = self.rebind(root/'bundle','proof.json',proof)
        built = load(root/'built.json'); built['sha256'] = digest
        (root/'built.json').write_text(json.dumps(built))
        with self.assertRaisesRegex(ValueError,'construction arm'):
            receive(root,load(root/'spec.json'),load(root/'invocation.json')['invocation'],sha(root/'invocation.json'))

    def test_producer_cannot_rebind_trusted_checker_context(self):
        root = self.root/'rebind-checker-context'
        result = observed_run(root,specification(control='rebind_checker_context'),budget=30)
        self.assertEqual(result['status'],'ERROR')
        self.assertEqual(len(load(root/'terminal.json')['stages']),1)
        self.assertFalse((root/'check.stdout').exists())
        with self.assertRaisesRegex(ValueError,'invocation identity'): audit(root,result)

    def test_invalid_budget(self):
        for budget in (0,301,float('nan'),True):
            with self.assertRaises(ValueError): supervise(self.root/'invalid-budget',specification(),budget=budget)


if __name__ == '__main__': unittest.main()
