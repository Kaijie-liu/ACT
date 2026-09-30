"""H2 complete execution controls; frozen synthetic sources, no real requests."""
from copy import deepcopy
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
from scoped_proof.supervisor import group_rss
from scoped_source.endpoint_source_controls import cases
from scoped_source.endpoint_portable import CODE
from scoped_source.endpoint_supervised import supervise, audit, receive, POSITIVE, PRODUCER_FILES, PROTOCOL_SHA256
from source_enclosure.format import identity

BASE = ROOT/'data/moe/tmp'
IMPLEMENTATION = sorted(set(CODE) | set(PRODUCER_FILES) | {
    'scoped_source/endpoint_verify.py','scoped_source/endpoint_supervision_tests.py',
    'scripts/archive_h2_supervision.py'})


def specification(case='weighted_sign', mode='endpoints', **changes):
    doc, reuse = next((doc,reuse) for name,doc,reuse in cases() if name == case)
    return {'schema':'H2_SYNTHETIC_SUPERVISION_V1', 'case':case, 'mode':mode,
            'reuse':[[list(p),j] for p,j in reuse], 'source_sha256':identity(doc),
            'protocol_sha256':PROTOCOL_SHA256, 'control':'', **changes}


def observed_run(root, spec, **kwargs):
    result = supervise(root,spec,**kwargs)
    save(root/'caller_observation.json',result)  # administrative observation after return
    return result


class H2SupervisionControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        keep = os.environ.get('H2_CONTROL_ROOT')
        if keep:
            cls.root = Path(keep); cls.root.mkdir(parents=True,exist_ok=False)
        else:
            cls.tmp = tempfile.TemporaryDirectory(prefix='h2-supervision-',dir=BASE)
            cls.root = Path(cls.tmp.name)
        bindings = {}
        for name in IMPLEMENTATION:
            target = cls.root/'implementation'/name; target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/name,target); bindings[name] = sha(target)
        save(cls.root/'implementation.json',bindings)
        cls.success = cls.root/'success'
        cls.run_result = observed_run(cls.success,specification(),budget=30)
        if cls.run_result['status'] != POSITIVE:
            raise AssertionError('initial H2 complete call failed: '+str(cls.run_result))
        cls.bundle = load(cls.success/'built.json')
        cls.source_hash = specification()['source_sha256']

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls,'tmp'): cls.tmp.cleanup()

    def copy_bundle(self, name):
        path = self.root/name; shutil.copytree(self.success/'bundle',path); return path

    def verify_bundle(self, root, digest=None, mode='endpoints', deadline=None):
        command = [PYTHON,'-B','-I','-S',str(root/'verify.py'),'--manifest-sha',digest or self.bundle['sha256'],
                   '--source-sha',self.source_hash,'--mode',mode]
        if deadline is not None: command += ['--deadline',str(deadline)]
        return subprocess.run(command,cwd=self.root,env=dict(os.environ,PYTHONPATH='/absent'),
                              text=True,capture_output=True,timeout=10)

    def rebind(self, root, name, value):
        (root/name).write_text(json.dumps(value,sort_keys=True))
        manifest = load(root/'manifest.json'); manifest['files'][name] = sha(root/name)
        (root/'manifest.json').write_text(json.dumps(manifest,sort_keys=True))
        return sha(root/'manifest.json')

    def test_complete_proof_full_cost_and_external_observation(self):
        result = audit(self.success,self.run_result)
        self.assertTrue(result['positive_execution_accepted'])
        self.assertFalse(audit(self.success)['positive_execution_accepted'])
        accepted = load(self.success/'accepted.json'); terminal = load(self.success/'terminal.json')
        self.assertEqual((accepted['required'],accepted['positive']), (3,3))
        self.assertEqual(accepted['result']['lp_bounds_checked'],6)
        self.assertEqual([s['phase'] for s in terminal['stages']], ['produce','check','receive'])
        self.assertEqual(terminal['checker_stdout_sha256'],sha(self.success/'check.stdout'))
        self.assertEqual(terminal['stages'][1]['checker_stdout_sha256'],accepted['checker_stdout_sha256'])
        self.assertGreater(self.run_result['seconds'],terminal['stage_seconds'])
        self.assertFalse(accepted['result']['deployed_float_SAFE'])
        self.assertFalse(accepted['result']['hard_budget_supervision'])
        events = [json.loads(s) for s in (self.success/'produce_events.jsonl').read_text().splitlines()]
        names = {e['operation'] for e in events}
        for name in ('source_creation_imports','source_publication','proposal_imports',
                     'source_construct_and_propose','serialize_and_bundle','native_candidate_0','candidate_publication_0'):
            self.assertIn(name,names)

    def test_all_fixed_sources_both_arms_and_common_facts(self):
        for case in ('weighted_sign','tied_partial_reuse','unsafe_tied','unresolved_sign'):
            paths = {}
            for mode in ('endpoints','mccormick'):
                if (case,mode) == ('weighted_sign','endpoints'): path = self.success; result = self.run_result
                else:
                    path = self.root/(case+'-'+mode)
                    result = observed_run(path,specification(case,mode),budget=30)
                expected = ('UNKNOWN_NONPOSITIVE' if case == 'unsafe_tied' or
                            (case,mode) == ('weighted_sign','mccormick') else POSITIVE)
                self.assertEqual(result['status'],expected,(case,mode)); audit(path,result); paths[mode] = path
                checked = load(path/'accepted.json')['result']
                if case == 'tied_partial_reuse':
                    self.assertEqual((checked['positive'],checked['required']), (18,18))
                    self.assertEqual(checked['lp_bounds_checked'],18 if mode=='endpoints' else 15)
                    self.assertEqual(checked['origins'].count('SOURCE_BOX_REUSE'),3)
            left,right = [load(paths[m]/'bundle/proof.json') for m in ('endpoints','mccormick')]
            self.assertEqual(left['request'],right['request'])
            self.assertEqual(left['reuse_requested'],right['reuse_requested'])

    def test_relocated_solver_free_check_and_wrong_mode(self):
        moved = self.copy_bundle('moved'); original = self.success/'bundle'; hidden = self.success/'bundle-hidden'
        original.rename(hidden)
        try: result = self.verify_bundle(moved)
        finally: hidden.rename(original)
        self.assertEqual(result.returncode,0,result.stderr)
        checked = json.loads(result.stdout)
        self.assertEqual(checked['result']['status'],POSITIVE)
        self.assertFalse(checked['solver_imported']); self.assertFalse(checked['producer_imported'])
        self.assertNotEqual(self.verify_bundle(moved,mode='mccormick').returncode,0)

    def test_members_and_math_tampering_rejected(self):
        for i,name in enumerate(('source.json','proof.json','code/scoped_source/endpoint_source_check.py','code/scoped_source/endpoint_check.py')):
            root = self.copy_bundle('bytes'+str(i))
            with (root/name).open('a') as stream: stream.write('changed')
            self.assertNotEqual(self.verify_bundle(root).returncode,0)
        for i,kind in enumerate(('duty','endpoint','gate','range','dual')):
            root = self.copy_bundle('math'+str(i)); proof = load(root/'proof.json')
            if kind == 'duty': proof['proof']['duties'].pop()
            if kind == 'endpoint': proof['proof']['duties'][0]['endpoints'].pop()
            if kind == 'gate': proof['request']['duties'][0]['gate'] = ['0','0']
            if kind == 'range': proof['bank']['input/0']['bounds'] = ['0','0']
            if kind == 'dual': proof['proof']['duties'][0]['endpoints'][0]['certificate']['claimed_lower_bound'] = '100'
            digest = self.rebind(root,'proof.json',proof)
            self.assertNotEqual(self.verify_bundle(root,digest).returncode,0)

    def test_complete_inventory_and_no_external_access(self):
        for kind in ('extra','symlink','duplicate'):
            root = self.copy_bundle(kind); digest = self.bundle['sha256']
            if kind == 'extra': (root/'code/local_check.py').write_text('raise Exception()')
            if kind == 'symlink': (root/'outside').symlink_to('/etc/hosts')
            if kind == 'duplicate':
                manifest = root/'manifest.json'; text = manifest.read_text()
                manifest.write_text(text[:-1]+',"mode":"duplicate"}'); digest = sha(manifest)
            self.assertNotEqual(self.verify_bundle(root,digest).returncode,0)
        path = self.success/'bundle/verify.py'
        for expression in ("open('/etc/hosts').read()", "__import__('scipy')", "__import__('socket').socket()",
                           "__import__('os').posix_spawn('/bin/true',['/bin/true'],{})"):
            code = f"import runpy,sys; m=runpy.run_path({str(path)!r}); sys.addaudithook(m['guard']); {expression}"
            output = subprocess.run([PYTHON,'-B','-I','-S','-c',code],text=True,capture_output=True,timeout=5)
            self.assertNotEqual(output.returncode,0,expression)

    def test_missing_one_two_endpoints_and_mc_certificate(self):
        for mode,fault,missing in (('endpoints','missing_certificate',1),('endpoints','missing_both',2),
                                   ('mccormick','missing_certificate',1)):
            root = self.root/(mode+'-'+fault)
            result = observed_run(root,specification(mode=mode,control=fault),budget=30)
            self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE'); audit(root,result)
            accepted = load(root/'accepted.json')
            self.assertEqual(accepted['missing'],missing)
            self.assertEqual(accepted['required'],3)
            self.assertEqual(sum(d['lower_bound'] is None for d in accepted['result']['duties']),1)

    def test_errors_and_partial_evidence_are_preserved(self):
        for fault in ('omit_property','missing_endpoint','exception_after_bundle','partial_output','wrong_invocation','wrong_mode'):
            root = self.root/fault; result = observed_run(root,specification(control=fault),budget=30)
            self.assertEqual(result['status'],'ERROR',fault)
            self.assertFalse(audit(root,result)['positive_execution_accepted'])
            self.assertTrue((root/'bundle/proof.json').exists())

    def test_proposal_failure_is_unknown_not_false_safe(self):
        root = self.root/'proposal-exception'
        result = observed_run(root,specification(control='proposal_exception'),budget=30)
        self.assertEqual(result['status'],'UNKNOWN_MISSING_EVIDENCE'); audit(root,result)
        self.assertEqual(len(load(root/'generation.json')['errors']),4)

    def test_source_identity_no_overwrite_and_budget_admission(self):
        with self.assertRaises(FileExistsError): supervise(self.success,specification())
        root = self.root/'wrong-source'; result = observed_run(root,specification(source_sha256='0'*64),budget=5)
        self.assertEqual(result['status'],'ERROR'); audit(root,result)
        for budget in (0,301,True,float('nan')):
            with self.assertRaises(ValueError): supervise(self.root/'invalid',specification(),budget=budget)
        spec = specification(); spec['reuse'] = [[[0,1],1]]
        with self.assertRaises(ValueError): supervise(self.root/'changed-reuse',spec)

    def test_hard_phase_cutoffs_partial_candidates_and_late_publication(self):
        for fault,phase,budget in (('produce_delay','produce',.5),('proposal_delay','produce',6),
                                  ('serialization_delay','produce',6),('check_delay','check',6),
                                  ('receive_delay','receive',6),('late_publish','receive',6)):
            root = self.root/fault; result = observed_run(root,specification(control=fault),budget=budget)
            self.assertEqual(result['status'],'TIMEOUT',fault)
            self.assertEqual(load(root/'terminal.json')['stages'][-1]['phase'],phase)
            self.assertFalse((root/'accepted.json').exists())
            self.assertFalse(audit(root,result)['positive_execution_accepted'])
            if fault == 'proposal_delay': self.assertTrue((root/'proposal_partial.json').exists())
            if fault == 'serialization_delay': self.assertTrue((root/'serialize_partial.json').exists())
        self.assertNotEqual(self.verify_bundle(self.success/'bundle',deadline=time.monotonic()-1).returncode,0)

    def test_resource_limit_and_only_owned_descendants_cleaned(self):
        root = self.root/'rss'; result = observed_run(root,specification(control='memory'),budget=5,rss_limit=64*2**20)
        self.assertEqual(result['status'],'RESOURCE_LIMIT'); audit(root,result)
        sentinel = subprocess.Popen([PYTHON,'-B','-c','import time;time.sleep(20)'],start_new_session=True)
        try:
            root = self.root/'descendant'; result = observed_run(root,specification(control='descendant'),budget=5)
            self.assertEqual(result['status'],'ERROR'); audit(root,result)
            pid = load(root/'terminal.json')['stages'][0]['pid']
            for _ in range(20):
                if not group_rss(pid)[1]: break
                time.sleep(.01)
            self.assertFalse(group_rss(pid)[1]); self.assertIsNone(sentinel.poll())
        finally: sentinel.terminate(); sentinel.wait()

    def test_actual_final_publication_overrun_and_prelaunch_expiry(self):
        import scoped_source.endpoint_supervised as module
        real_save = module.save
        def slow(path,value):
            result = real_save(path,value)
            if path.name == 'finish.json': time.sleep(5)
            return result
        root = self.root/'publication-overrun'
        with patch.object(module,'save',side_effect=slow): result = observed_run(root,specification(),budget=5)
        self.assertEqual(result['status'],'TIMEOUT'); self.assertGreaterEqual(result['seconds'],5)
        self.assertTrue((root/'publication_timeout.json').exists())
        self.assertFalse(audit(root,result)['positive_execution_accepted'])
        root = self.root/'prelaunch-expiry'; result = observed_run(root,specification(),budget=1e-9)
        self.assertEqual(result['status'],'TIMEOUT'); audit(root,result)
        self.assertIsNone(load(root/'terminal.json')['stages'][0]['pid'])

    def test_invocation_and_checker_rebinding_rejected_before_execution(self):
        root = self.root/'rebind-checker-context'
        result = observed_run(root,specification(control='rebind_checker_context'),budget=30)
        self.assertEqual(result['status'],'ERROR')
        self.assertFalse((root/'check.stdout').exists())
        with self.assertRaisesRegex(ValueError,'invocation identity'): audit(root,result)

    def test_cost_and_completed_call_tampering_rejected(self):
        for kind in ('cost','budget','proof','observation'):
            root = self.root/('audit-'+kind); shutil.copytree(self.success,root)
            if kind == 'cost':
                value = load(root/'terminal.json'); value['stage_seconds'] = -1
                (root/'terminal.json').write_text(json.dumps(value))
            if kind == 'budget':
                value = load(root/'invocation.json'); value['budget'] = 301
                (root/'invocation.json').write_text(json.dumps(value))
            if kind == 'proof': (root/'bundle/proof.json').write_text('{}')
            if kind == 'observation':
                value = deepcopy(self.run_result); value['seconds'] = 0
                with self.assertRaises(ValueError): audit(self.success,value)
            else:
                with self.assertRaises(ValueError): audit(root)
        root = self.root/'late-marker'; shutil.copytree(self.success,root)
        save(root/'publication_timeout.json',{'status':'TIMEOUT','seconds':301})
        self.assertEqual(audit(root)['execution_status'],'TIMEOUT')

    def test_required_hashes_cannot_disable_identity_checks(self):
        for digest in (None,'',0,'0'*63,'g'*64,'A'*64):
            call = deepcopy(self.run_result); call['finish_sha256'] = digest
            with self.assertRaisesRegex(ValueError,'mandatory SHA-256'): audit(self.success,call)
        call = deepcopy(self.run_result); del call['finish_sha256']
        with self.assertRaisesRegex(ValueError,'mandatory SHA-256'): audit(self.success,call)
        # No caller anchor is allowed only for NON-accepting stored-file audit.
        for layer in ('finish','receipt','terminal'):
            root = self.root/('null-hash-'+layer); shutil.copytree(self.success,root)
            finish,receipt,terminal = [load(root/(name+'.json')) for name in ('finish','receipt','terminal')]
            if layer == 'terminal':
                terminal['invocation_sha256'] = None
                (root/'terminal.json').write_text(json.dumps(terminal))
                receipt['terminal_sha256'] = sha(root/'terminal.json')
            if layer == 'receipt': receipt['terminal_sha256'] = None
            (root/'receipt.json').write_text(json.dumps(receipt))
            finish['receipt_sha256'] = None if layer == 'finish' else sha(root/'receipt.json')
            (root/'finish.json').write_text(json.dumps(finish))
            with self.assertRaises(ValueError): audit(root)

    def test_receiver_recomputes_endpoint_minimum_and_count(self):
        # Supply the mutated digest deliberately to exercise aggregation behind
        # the hash guard. Actual execution pins the ORIGINAL output in the parent.
        for kind in ('minimum','missing','count','mode'):
            root = self.root/('receiver-'+kind); shutil.copytree(self.success,root)
            value = load(root/'check.stdout')
            if kind == 'minimum': value['result']['duties'][0]['lower_bound'] = '100'
            if kind == 'missing': value['result']['missing'] = 7
            if kind == 'count': value['result']['lp_bounds_checked'] = 1
            if kind == 'mode': value['mode'] = 'mccormick'
            (root/'check.stdout').write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                receive(root,load(root/'spec.json'),load(root/'invocation.json')['invocation'],
                        sha(root/'invocation.json'),sha(root/'check.stdout'))

    def test_receiver_checks_mc_fact_kind_and_bound_count(self):
        parent = self.root/'tied_partial_reuse-mccormick'
        for kind in ('fact-kind','fact-as-lp'):
            root = self.root/('receiver-'+kind); shutil.copytree(parent,root)
            checked = load(root/'check.stdout'); result = checked['result']
            if kind == 'fact-kind':
                index = result['origins'].index('SOURCE_BOX_REUSE')
                result['origins'][index] = 'NATIVE_PROPOSAL'
            else: result['lp_bounds_checked'] += result['origins'].count('SOURCE_BOX_REUSE')
            (root/'check.stdout').write_text(json.dumps(checked))
            with self.assertRaisesRegex(ValueError,'whole-request aggregation'):
                receive(root,load(root/'spec.json'),load(root/'invocation.json')['invocation'],
                        sha(root/'invocation.json'),sha(root/'check.stdout'))

    def test_self_consistent_stdout_replacement_rejected(self):
        root = self.root/'self-consistent-stdout'
        result = observed_run(root,specification(mode='mccormick',control='rewrite_check_stdout'),budget=30)
        self.assertEqual(result['status'],'ERROR'); audit(root,result)
        terminal = load(root/'terminal.json')
        self.assertEqual(terminal['stages'][-1]['phase'],'receive')
        self.assertNotEqual(terminal['checker_stdout_sha256'],sha(root/'check.stdout'))
        self.assertFalse((root/'accepted.json').exists())
        self.assertIn('file identity mismatch',(root/'receive.log').read_text())
        inv = load(root/'invocation.json')
        for anchor in (terminal['checker_stdout_sha256'],None):
            with self.assertRaises(ValueError):
                receive(root,load(root/'spec.json'),inv['invocation'],sha(root/'invocation.json'),anchor)
        # A fresh solver-free check still finds the ORIGINAL proof nonpositive.
        built = load(root/'built.json')
        checked = subprocess.run([PYTHON,'-B','-I','-S',str(root/'bundle/verify.py'),
            '--manifest-sha',built['sha256'],'--source-sha',self.source_hash,'--mode','mccormick'],
            text=True,capture_output=True,timeout=10)
        self.assertEqual(checked.returncode,0,checked.stderr)
        self.assertEqual(json.loads(checked.stdout)['result']['status'],'UNKNOWN_NONPOSITIVE')


if __name__ == '__main__': unittest.main()
