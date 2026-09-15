"""Analytic reserve races and proof-chain controls; no cohort reruns."""
from contextlib import contextmanager
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from scripts.optional_evidence_budget import EvidenceBudget, EvidenceBudgetExpired
from scripts.optional_evidence_dev_contract import ROOT, save, read
from scripts.test_general_evidence import fixture, box_certificate
from moe_evidence.checker import check_manifest
from moe_evidence.storage import loader
from moe_evidence.generate import propose_all
from evidence_handoff.proposal import ProposalBudget, ReserveHandoff, propose_with_handoff


@contextmanager
def source(experts=2, classes=3, base=3, partial=False):
    manifest, req, files = fixture(experts, classes, base, partial)
    for item in manifest['supports'].values():
        item.update(status='PENDING', certificate=None)
    for row in manifest['obligations']:
        if row['kind'] == 'residual':
            row.update(weighted_status='RANGE_UNAVAILABLE', weighted=None, certificate=None)
            row.pop('gate_bounds'); row.pop('difference_bounds')
    with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
        root = Path(tmp)
        for name, obj in files.items(): save(root/name, obj)
        save(root/'manifest.json', manifest)
        yield root, req


def checked(root, request):
    return check_manifest(read(root/'manifest.json'), request, loader(root))


class HandoffControls(unittest.TestCase):
    def test_only_reserve_exhaustion_is_handoff(self):
        now = [220.1]; budget = EvidenceBudget(0, clock=lambda: now[0]); proxy = ProposalBudget(budget)
        with self.assertRaises(ReserveHandoff): proxy.grant(60, 80)
        with self.assertRaises(EvidenceBudgetExpired): proxy.grant(60, 90)
        now[0] = 299
        with self.assertRaises(EvidenceBudgetExpired): proxy.grant(60, 80)
        with self.assertRaises(ValueError): ProposalBudget(EvidenceBudget(0, total=600, clock=lambda: 1))

    def test_crossing_during_support_validation(self):
        from act.back_end.solver.check_hz_lp_export import check_export
        now = [219.9]
        def slow(*a, **kw):
            result = check_export(*a, **kw); now[0] = 220.1; return result
        with source() as (root, req), patch('act.back_end.solver.check_hz_lp_export.check_export', side_effect=slow), \
                patch('act.back_end.solver.lp_certificate.propose', side_effect=AssertionError('solver called')):
            result = propose_with_handoff(root, EvidenceBudget(0, clock=lambda: now[0]))
            self.assertEqual(result['reason'], 'RESERVE_EXHAUSTED_HANDOFF')
            self.assertGreater(result['remaining_request_seconds'], 79)
            self.assertEqual(checked(root, req)['status'], 'UNKNOWN_MISSING_EVIDENCE')

    def test_crossing_between_two_grants(self):
        now = [219.9]
        def delayed_save(path, value):
            save(path, value)
            if path.name == 'query_log.json': now[0] = 220.1
        with source() as (root, req), patch('moe_evidence.generate.save', side_effect=delayed_save), \
                patch('moe_evidence.generate.time.monotonic', side_effect=lambda: now[0]), \
                patch('act.back_end.solver.lp_certificate.propose', side_effect=AssertionError('solver called')):
            result = propose_with_handoff(root, EvidenceBudget(0, clock=lambda: now[0]))
            self.assertEqual(result['reason'], 'RESERVE_EXHAUSTED_HANDOFF')
            self.assertEqual(read(root/'query_log.json')[-1]['status'], 'PENDING')
            self.assertEqual(checked(root, req)['positive_obligations'], 0)

    def test_weighted_crossing_preserves_prior_proof_and_portable_unknown(self):
        from act.back_end.solver.rational_mccormick import build
        from moe_evidence.bundle import pack
        now = [0.]; calls = [0]
        def delayed(*a, **kw):
            result = build(*a, **kw); calls[0] += 1
            if calls[0] == 2: now[0] = 220.1
            return result
        with source() as (root, req), patch('act.back_end.solver.rational_mccormick.build', side_effect=delayed), \
                patch('act.back_end.solver.lp_certificate.propose', side_effect=lambda lp, **kw: box_certificate(lp)), \
                patch('moe_evidence.generate.time.monotonic', side_effect=lambda: now[0]):
            event = propose_with_handoff(root, EvidenceBudget(0, clock=lambda: now[0]))
            self.assertEqual(event['reason'], 'RESERVE_EXHAUSTED_HANDOFF')
            result = checked(root, req)
            self.assertEqual((result['positive_obligations'], result['missing_obligations']), (1, 1))
            self.assertEqual(result['status'], 'UNKNOWN_MISSING_EVIDENCE')
            meta = pack(root, root/'portable', result)
            run = subprocess.run([sys.executable, '-I', '-S', str(root/'portable/verify.py'),
                                  '--bundle-hash', meta['bundle_sha256'], '--statement-hash', meta['statement_sha256']],
                                 cwd=root, capture_output=True, text=True, check=True, timeout=30)
            self.assertEqual(json.loads(run.stdout)['result'], result)
            with self.assertRaises(FileExistsError): propose_with_handoff(root, EvidenceBudget(0, clock=lambda: 0))

    def test_real_deadline_during_construction_is_not_handoff(self):
        from act.back_end.solver.check_hz_lp_export import check_export
        now = [0.]
        def delayed(*a, **kw):
            result = check_export(*a, **kw); now[0] = 299; return result
        with source() as (root, req), patch('act.back_end.solver.check_hz_lp_export.check_export', side_effect=delayed):
            with self.assertRaises(EvidenceBudgetExpired): propose_with_handoff(root, EvidenceBudget(0, clock=lambda: now[0]))
            self.assertFalse((root/'handoff.json').exists())

    def test_semantic_and_io_errors_not_converted(self):
        for error in (ValueError('invalid export'), OSError('disk unavailable')):
            with source() as (root, req), patch('act.back_end.solver.check_hz_lp_export.check_export', side_effect=error):
                with self.assertRaises(type(error)): propose_with_handoff(root, EvidenceBudget(time.monotonic()))
                self.assertFalse((root/'handoff.json').exists())

    def test_unchanged_full_results_multidimension_ties_reuse(self):
        for e, c, base, partial in ((2, 2, 3, False), (3, 3, 3, True), (4, 4, -2, False)):
            with source(e, c, base, partial) as (old, req), source(e, c, base, partial) as (new, _), \
                    patch('act.back_end.solver.lp_certificate.propose', side_effect=lambda lp, **kw: box_certificate(lp)):
                propose_all(old, EvidenceBudget(time.monotonic()))
                result = propose_with_handoff(new, EvidenceBudget(time.monotonic()))
                self.assertEqual(result['reason'], 'PROPOSAL_LOOP_RETURNED')
                self.assertEqual(read(old/'manifest.json'), read(new/'manifest.json'))
                self.assertEqual(checked(old, req), checked(new, req))
                expected = 'CHECKED_CONDITIONAL' if base > 0 else 'UNKNOWN_NONPOSITIVE'
                self.assertEqual(checked(new, req)['status'], expected)

    def test_unfinished_check_never_promotes_handoff(self):
        from moe_evidence.execution import accept, PHASES
        stages = {k: {'state': 'COMPLETED'} for k in PHASES['evidence']}
        stages['check']['state'] = 'OUTER_TIMEOUT'
        with patch('json.loads', side_effect=AssertionError):
            self.assertEqual(accept('evidence', stages, lambda: self.fail('late proof read'), 299), ('TIMEOUT', False))
        stages['check']['state'] = 'COMPLETED'
        self.assertEqual(accept('evidence', stages, lambda: self.fail('late proof read'), 301), ('TIMEOUT', False))

    def test_actual_worker_driver_chain_on_analytic_model(self):
        from dataclasses import asdict
        import torch
        from act.back_end.moe import OutputMoEFactoryConfig, GateKind
        from act.pipeline.moe.test_route_complexity_schedule import model, config
        from act.pipeline.moe.staged_verifier import _model_state_identity, _tensor_identity
        from evidence_cohort.run import wait_owned, environment
        from portable_proof.runtime import digest
        from moe_evidence.audit import audit_request
        net = model(((-.2, 0., -2.), (1., 0., -2.), (2., 0., -2.)))
        fc = OutputMoEFactoryConfig(input_shape=(2,), num_classes=3, num_experts=3, top_k=2,
                                   gate=GateKind.SELECTED_SOFTMAX, router_hidden=(), expert_hidden=(), seed=7)
        x = torch.full((1, 2), .5, dtype=torch.float64); tensors = {'center': x, 'lower': x-.01, 'upper': x+.01}
        with tempfile.TemporaryDirectory(dir=ROOT/'data/moe/results') as tmp:
            base = Path(tmp); root = base/'request'; control = base/'control'; root.mkdir(); control.mkdir()
            cp = base/'model.pt'; tp = base/'input.pt'; cfg = base/'config.json'
            torch.save({'format': 'act-output-moe-v1', 'factory_config': asdict(fc), 'state_dict': net.state_dict()}, cp)
            torch.save(tensors, tp); save(cfg, config('monolithic_f0'))
            sample = {'dataset_index': -1, 'label': 0, **{k: _tensor_identity(v) for k, v in tensors.items()}}
            r = fixture()[1]; r.update(epsilon=.01, model_state=_model_state_identity(net), **{k: sample[k] for k in tensors})
            req = {'subject': {'checkpoint': str(cp), 'checkpoint_sha256': digest(cp.read_bytes()), 'model_state': r['model_state']},
                   'sample': sample, 'epsilon': .01, 'config': {'path': str(cfg), 'sha256': digest(cfg.read_bytes())},
                   'tensors': {'path': str(tp), 'sha256': digest(tp.read_bytes())}, 'evidence_request': r, 'head': 'analytic', 'method': 'evidence'}
            save(root/'request.json', req); started = time.monotonic()
            with (control/'driver.log').open('w') as log:
                p = subprocess.Popen([sys.executable, '-m', 'evidence_handoff.driver', str(root), str(control), '--started', repr(started)],
                                     cwd=ROOT, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                outcome = wait_owned(p, started+298)
            self.assertEqual(outcome['return_code'], 0, (control/'driver.log').read_text())
            self.assertFalse(outcome['killed'])
            terminal = read(control/'candidate.json')
            self.assertEqual(terminal['status'], 'CHECKED_CONDITIONAL', terminal)
            self.assertLess(terminal['wall_seconds'], 300)
            self.assertEqual(read(root/'handoff.json')['schema'], 'EVIDENCE_RESERVE_HANDOFF_V1')
            save(root/'terminal.json', terminal)
            self.assertTrue(audit_request(root, req, 'evidence')['conditional_check'])


if __name__ == '__main__':
    unittest.main()
