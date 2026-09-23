"""Synthetic proof-mode controls. No frozen real model/request is opened."""
import copy
from dataclasses import asdict
from fractions import Fraction as F
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from source_enclosure.format import identity
from scoped_source.build import build
from scoped_source.capture import capture
from scoped_proof.io import save, load, sha, PYTHON, ROOT
from scoped_proof.evidence import aggregate, context, lower_bound, reconstruct_lp, roster, POSITIVE
from scoped_proof.supervisor import supervise, execute, accept
from scoped_proof.audit import audit


def make_fixture(root, bias=2.):
    import torch
    from act.back_end.moe.factory import OutputMoEFactoryConfig, build_output_moe
    from act.back_end.moe.schema import GateKind
    cfg = OutputMoEFactoryConfig([2], 3, 3, 2, GateKind.SELECTED_SOFTMAX, (), (2,))
    model = build_output_moe(cfg).double().eval()
    with torch.no_grad():
        for p in model.parameters(): p.zero_()
        for expert in model.experts: expert[-1].bias[2] = bias
    center = torch.zeros((1,2), dtype=torch.float64)
    doc = capture(model, center, label=2, radius='1/8', margin='1/100', clip=['-1','1'], deadline=time.monotonic()+300)
    scope = {k: doc['request'][k] for k in ('experts','classes','label','radius','margin','clip','center','model_state')}
    config = asdict(cfg); config['gate'] = cfg.gate.value
    torch.save({'format':'act-output-moe-v1','factory_config':config,'state_dict':model.state_dict()}, root/'checkpoint.pt')
    torch.save({'center':center, 'lower':torch.full_like(center, 99), 'upper':torch.full_like(center, -99)}, root/'input.pt')
    spec = {'scope': scope, 'sources': {},
        'checkpoint': {'path':str(root/'checkpoint.pt'),'sha256':sha(root/'checkpoint.pt')},
        'input': {'path':str(root/'input.pt'),'sha256':sha(root/'input.pt')}}
    bundle = build(doc, expected_source_sha256=identity(doc), deadline=time.monotonic()+300)
    candidates = {}; expected = roster(scope)
    for i, obligation in enumerate(expected):
        p = next(p for p in bundle['pairs'] if p['pair'] == obligation['pair'])
        row = next(r for r in p['obligations']['rows'] if r['competitor'] == obligation['competitor'])
        lp = reconstruct_lp(p['base'], row)
        cert = {'lp_sha256':identity(lp), 'inequality_dual':[0]*len(lp['b']), 'equality_dual':[0]*len(lp['h'])}
        candidates[i] = {'schema':'SCOPED_LP_CANDIDATE_V1', 'context':context(scope,identity(doc),identity(bundle),'control',i,obligation,lp),
            'status':'CANDIDATE','certificate':cert,'native':{'solver_status':1,'fake_primal_optimum':1000}}
    return spec, doc, bundle, candidates


class EvidenceControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir='/data1/Kane/MOE'); self.root = Path(self.temp.name)
        self.spec, self.doc, self.bundle, self.candidates = make_fixture(self.root)

    def tearDown(self): self.temp.cleanup()

    def check(self, candidates=None, complete=True, doc=None, bundle=None):
        return aggregate(self.spec['scope'], doc or self.doc, bundle or self.bundle,
            self.candidates if candidates is None else candidates, invocation='control',
            proposal_complete=complete, deadline=time.monotonic()+300)

    def test_complete_nonoptimal_dual_accepted_only_by_exact_bound(self):
        result = self.check()
        self.assertEqual(result['status'], POSITIVE); self.assertEqual(result['checked_bounds'], 6)
        self.assertFalse(result['native_float_proof']); self.assertFalse(result['route_changing_established'])
        self.assertEqual({r['checked_lower_bound'] for r in result['rows']}, {'199/100'})

    def test_missing_partial_and_no_candidate_never_close(self):
        for mode in ('missing','partial','null','empty'):
            with self.subTest(mode=mode):
                rows = copy.deepcopy(self.candidates)
                if mode == 'missing': rows.pop(5)
                if mode == 'null': rows[0].update(status='NO_CANDIDATE', certificate=None)
                if mode == 'empty': rows = {}
                self.assertEqual(self.check(rows, complete=mode != 'partial')['status'], 'NOT_CLOSED')

    def test_wrong_request_run_source_property_and_lp_rejected(self):
        for key in ('request_sha256','source_sha256','bundle_sha256','invocation','competitor','pair','lp_sha256'):
            with self.subTest(key=key):
                rows = copy.deepcopy(self.candidates); rows[0]['context'][key] = 'wrong'
                with self.assertRaises(ValueError): self.check(rows)
        rows = copy.deepcopy(self.candidates); rows[1] = rows[0]
        with self.assertRaises(ValueError): self.check(rows)
        rows = copy.deepcopy(self.candidates); rows[99] = rows[0]
        with self.assertRaises(ValueError): self.check(rows)

    def test_source_change_and_missing_guard_or_output(self):
        doc = copy.deepcopy(self.doc); doc['request']['margin'] = '0'
        with self.assertRaises(ValueError): self.check(doc=doc)
        for key in ('guard','property','pair'):
            bundle = copy.deepcopy(self.bundle)
            if key == 'guard': bundle['pairs'][0]['guarded']['hz']['ub'][-1] = '1'
            if key == 'property': bundle['pairs'][0]['obligations']['rows'].pop()
            if key == 'pair': bundle['pairs'].pop()
            with self.assertRaises(ValueError): self.check(bundle=bundle)

    def test_nonpositive_is_not_unsafe_or_relaxation_impossibility(self):
        spec, doc, bundle, rows = make_fixture(self.root, bias=-1.)
        result = aggregate(spec['scope'], doc, bundle, rows, invocation='control', proposal_complete=True, deadline=time.monotonic()+300)
        self.assertEqual(result['status'], 'NOT_CLOSED'); self.assertEqual(len(result['nonpositive']), 6)
        self.assertFalse(result['complete_output_positive_proof'])

    def test_exact_residual_and_kernel_differential(self):
        from source_enclosure.format import sparse
        from act.back_end.solver.sparse_lp_certificate import check as old_check
        lp = {'matrix_format':'csr_v1','c':['3/2'],'offset':'1/7','lower':['-2'],'upper':['4'],
            'A':sparse([{0:F(2)}],1),'b':['3'],'E':sparse([],1),'h':[]}
        cert = {'lp_sha256':identity(lp),'inequality_dual':[-.1],'equality_dual':[]}
        result = lower_bound(lp, cert); cert['claimed_lower_bound'] = result['checked_lower_bound']
        self.assertEqual(old_check(lp, cert)['checked_lower_bound'], result['checked_lower_bound'])
        self.assertNotEqual(F(result['residual_box_term']), 0)
        for value in (.1, float('nan'), float('inf')):
            bad = dict(cert, inequality_dual=[value])
            with self.assertRaises(ValueError): lower_bound(lp, bad)
        with self.assertRaises(ValueError): lower_bound(lp, dict(cert, lp_sha256='old'))

    def test_real_intake_float64_binding_and_old_endpoints_ignored(self):
        from scoped_proof.worker import intake
        self.assertEqual(intake(self.spec, time.monotonic()+300), self.doc)
        bad = copy.deepcopy(self.spec); bad['scope']['center']['sha256'] = '0'*64
        with self.assertRaises(ValueError): intake(bad, time.monotonic()+300)

    def test_expired_evidence_and_atomic_no_overwrite(self):
        with self.assertRaises(TimeoutError):
            aggregate(self.spec['scope'],self.doc,self.bundle,self.candidates,invocation='control',proposal_complete=True,deadline=time.monotonic()-1)
        save(self.root/'exclusive.json', {'ok':1})
        with self.assertRaises(FileExistsError): save(self.root/'exclusive.json', {'ok':2})
        self.assertEqual(load(self.root/'exclusive.json'), {'ok':1})

    def test_changed_intake_file_and_nonboolean_completion_refused(self):
        from scoped_proof.worker import intake
        bad = copy.deepcopy(self.spec); bad['input']['sha256'] = '0'*64
        with self.assertRaises(ValueError): intake(bad,time.monotonic()+300)
        with self.assertRaises(ValueError): self.check(complete='true')

    def test_receipt_rejects_wrong_identity_and_coverage(self):
        report = self.check(); root = self.root/'receipt'; root.mkdir()
        record = save(root/'evidence_check.json',report)
        small = {k:report[k] for k in ('status','request_sha256','invocation','required','checked_bounds','positive_bounds',
            'proposal_complete','complete_output_positive_proof','native_float_proof','route_changing_established')}
        small.update(evidence_check=record,missing=0,nonpositive=0)
        save(root/'result_candidate.json',small)
        accept(root,self.spec['scope'],'control')
        with self.assertRaises(ValueError): accept(root,self.spec['scope'],'different_run')
        # Mutations are in fresh synthetic directories; original receipt is not overwritten.
        for mode in ('missing','overclaim'):
            target = self.root/mode; target.mkdir(); bad = copy.deepcopy(report)
            if mode == 'missing': bad['rows'].pop()
            if mode == 'overclaim': bad['native_float_proof'] = True
            rec = save(target/'evidence_check.json',bad); header = dict(small,evidence_check=rec)
            if mode == 'overclaim': header['native_float_proof'] = True
            save(target/'result_candidate.json',header)
            with self.assertRaises(ValueError): accept(target,self.spec['scope'],'control')


class SupervisionControls(EvidenceControls):
    # Inherits arithmetic controls intentionally? No: load_tests below avoids duplication.
    def test_complete_synthetic_pipeline_and_fresh_solver_free_audit(self):
        root = self.root/'run'; result = supervise(root, self.spec, budget=30)
        self.assertEqual(result['status'], POSITIVE, '\n'.join(p.read_text() for p in root.glob('*.log')))
        reviewed = audit(root); self.assertTrue(reviewed['complete_output_positive_proof'])
        cost = load(root/'cost.json'); self.assertLess(cost['end_to_end_seconds'], 30)
        self.assertAlmostEqual(cost['stage_seconds']+cost['overhead_seconds'], cost['end_to_end_seconds'])
        command = [PYTHON,'-S','-m','scoped_proof.audit',str(root)]
        raw = subprocess.run(command,cwd=ROOT,text=True,capture_output=True,timeout=15)
        self.assertEqual(raw.returncode,0,raw.stderr)
        self.assertEqual(json.loads(raw.stdout)['checked_bounds'],6)
        with self.assertRaises(FileExistsError): supervise(root,self.spec,budget=30)

    def test_source_mismatch_is_early_error_with_full_cost(self):
        bad = copy.deepcopy(self.spec); bad['scope']['model_state']['sha256'] = '0'*64
        root = self.root/'bad_source'; result = supervise(root,bad,budget=10)
        self.assertEqual(result['status'],'ERROR'); self.assertFalse((root/'source.json').exists())
        terminal = load(root/'terminal.json'); self.assertEqual(len(terminal['stages']),1)
        self.assertGreater(audit(root)['end_to_end_seconds'],0)

    def test_complete_nonpositive_pipeline_stays_not_closed(self):
        spec, _, _, _ = make_fixture(self.root,bias=-1.)
        root = self.root/'negative'; result = supervise(root,spec,budget=30)
        self.assertEqual(result['status'],'NOT_CLOSED'); reviewed = audit(root)
        self.assertEqual(reviewed['checked_bounds'],6); self.assertFalse(reviewed['complete_output_positive_proof'])

    def test_hard_deadline_kills_owned_descendants_retains_partial(self):
        root = self.root/'timeout'
        def command(phase, folder, deadline):
            if phase != 'construct': return [PYTHON,'-S','-c','pass']
            code = ("import pathlib,subprocess,time; p=subprocess.Popen(['sleep','10']); "
                f"pathlib.Path({str(folder/'child.pid')!r}).write_text(str(p.pid)); "
                f"pathlib.Path({str(folder/'partial.txt')!r}).write_text('not proof'); time.sleep(10)")
            return [PYTHON,'-S','-c',code]
        result = supervise(root,self.spec,budget=.5,command_factory=command)
        self.assertEqual(result['status'],'TIMEOUT'); self.assertFalse(result['complete_output_positive_proof'])
        self.assertTrue((root/'partial.txt').exists()); self.assertEqual(audit(root)['effective_status'],'TIMEOUT')
        pid = int((root/'child.pid').read_text()); stat = Path(f'/proc/{pid}/stat')
        if stat.exists(): self.assertEqual(stat.read_text().rsplit(')',1)[1].split()[0], 'Z')

    def test_exception_resource_and_missing_result_fail_closed(self):
        for mode in ('exception','resource','missing'):
            with self.subTest(mode=mode):
                def command(phase, folder, deadline):
                    if mode == 'exception': raise RuntimeError('injected parent dispatch')
                    return [PYTHON,'-S','-c',"import time;time.sleep(.05)"]
                root = self.root/mode
                result = supervise(root,self.spec,budget=2,rss_limit=1 if mode=='resource' else 8*2**30,command_factory=command)
                self.assertEqual(result['status'],'RESOURCE_LIMIT' if mode=='resource' else 'ERROR')
                self.assertFalse(result['complete_output_positive_proof']); audit(root)

    def test_partial_candidate_then_proposer_exception(self):
        root = self.root/'partial'
        def command(phase, folder, deadline):
            if phase != 'propose':
                return [PYTHON]+(['-S'] if phase in ('source_check','aggregate') else [])+['-m','scoped_proof.worker',phase,str(folder),'--deadline',str(deadline)]
            code = ("from pathlib import Path; import scoped_proof.worker as w; original=w.propose; count=0\n"
                "def fail(lp, seconds):\n global count\n count+=1\n if count>1: raise RuntimeError('injected after first candidate')\n return original(lp,seconds)\n"
                f"w.propose=fail; w.work('propose',Path({str(folder)!r}),{deadline!r})")
            return [PYTHON,'-c',code]
        result = supervise(root,self.spec,budget=30,command_factory=command)
        self.assertEqual(result['status'],'ERROR'); self.assertFalse(result['complete_output_positive_proof'])
        checked = load(root/'evidence_check.json'); self.assertEqual(checked['checked_bounds'],1)
        self.assertEqual(len(checked['missing']),5); self.assertFalse(checked['proposal_complete']); audit(root)

    def test_late_complete_candidate_cannot_override_hard_deadline(self):
        root = self.root/'late'
        def command(phase, folder, deadline):
            if phase != 'aggregate': return [PYTHON]+(['-S'] if phase=='source_check' else [])+['-m','scoped_proof.worker',phase,str(folder),'--deadline',str(deadline)]
            code = ("import time; from pathlib import Path; from scoped_proof.worker import work; "
                f"work('aggregate',Path({str(folder)!r}),{deadline!r}); time.sleep(20)")
            return [PYTHON,'-S','-c',code]
        result = supervise(root,self.spec,budget=8,command_factory=command)
        self.assertEqual(result['status'],'TIMEOUT'); self.assertTrue((root/'result_candidate.json').exists())
        self.assertFalse(result['complete_output_positive_proof']); audit(root)

    def test_receipt_delay_is_charged_and_invalidates_success(self):
        from scoped_proof import supervisor
        real_save = supervisor.save
        def slow(path, value):
            if Path(path).name == 'receipt.json': time.sleep(.7)
            return real_save(path,value)
        with patch.object(supervisor,'save',side_effect=slow), patch.object(supervisor,'accept',return_value={'status':POSITIVE}):
            result = supervise(self.root/'publication',self.spec,budget=.6,
                command_factory=lambda *args:[PYTHON,'-S','-c','pass'])
        self.assertEqual(result['status'],'TIMEOUT')
        self.assertTrue((self.root/'publication/publication_timeout.json').exists())
        self.assertGreater(load(self.root/'publication/cost.json')['end_to_end_seconds'], .6)


def load_tests(loader, tests, pattern):
    suite = unittest.TestSuite()
    for cls in (EvidenceControls,SupervisionControls):
        suite.addTests(cls(name) for name in sorted(cls.__dict__) if name.startswith('test_'))
    return suite


if __name__ == '__main__': unittest.main()
