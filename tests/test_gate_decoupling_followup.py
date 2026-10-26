"""New reconstruction/provenance controls, not another graph benchmark."""
import copy
import os
from pathlib import Path
import time
import unittest

import gate_decoupling_followup as F
import gate_decoupling_worker as W
import gate_decoupling_stage4 as S


def bounds(prefix, e=3, replacements=None):
    replacements = replacements or {}
    return dict(prefix=[list(p) for p in prefix],
        upper=[[i, j, replacements.get((i, j), 1.)] for i in range(e) for j in range(e) if i != j],
        guarantee='WHOLE_BOX_CROWN_NUMERICAL_UPPER')


class FollowupControls(unittest.TestCase):
    def setUp(self):
        self.deadline = float(os.environ['GATE_CONTROL_DEADLINE'])
        self.folder = Path(os.environ['GATE_CONTROL_ROOT'])/self._testMethodName
        self.folder.mkdir()

    def test_fixed_selection_keeps_old_order_and_no_partial_pair(self):
        obs = [dict(id=str(i), group='multilayer') for i in range(4)]
        old = [dict(object_id=str(i), variant=v, status='SOURCE_MISSING' if i == 1 else 'UNKNOWN')
               for i in range(3) for v in ('V2', 'V3')]
        self.assertEqual([o['id'] for o in F.supplement_objects(obs, old)], ['1', '3'])
        for bad in (old+[old[0]], old[:-1], old+[dict(object_id='foreign', variant='V2')]):
            with self.assertRaises(ValueError): F.supplement_objects(obs, bad)
        t = F.frozen_t1([dict(seconds=i*7) for i in range(1, 39)])
        self.assertEqual((t['rank'], t['p90_seconds'], t['T1_seconds']), (35, 245, 120))
        with self.assertRaises(ValueError): F.frozen_t1([dict(seconds=1)]*37)

    def test_pruning_direction_tie_and_multilayer_first_history(self):
        seen = []
        def bound(prefix, deadline):
            self.assertEqual(deadline, self.deadline); seen.append(prefix)
            return bounds(prefix, replacements={(0, 2): -1.} if not prefix else {(0, 2): 0.})
        result = F.first_unexcluded([3, 3], bound, self.deadline)
        self.assertEqual(result['history'], [[0, 2], [0, 1]])
        self.assertEqual(seen, [(), ((0, 2),)])
        self.assertFalse(result['old_unproved_identity_recovered'])
        self.assertFalse(result['route_reachability_proved'])
        result['source_id'] = 'synthetic'
        self.assertEqual(F.check_selection(result, [3, 3], 'synthetic', self.deadline), result['history'])
        # Missing/nonfinite evidence cannot exclude a pair; ties are legal.
        self.assertEqual(F.first_unexcluded([3], lambda p, d: bounds(p, replacements={(0, 2): None}),
            self.deadline)['history'], [[0, 1]])

    def test_reconstruction_mutations_and_complete_empty_tree(self):
        result = F.first_unexcluded([3], lambda p, d: bounds(p), self.deadline)
        result['source_id'] = 'fixed'
        mutations = []
        for key, value in [('source_id', 'foreign'), ('history', [[0, 2]]),
                           ('old_unproved_identity_recovered', True), ('prefix_records', [])]:
            bad = copy.deepcopy(result); bad[key] = value; mutations.append(bad)
        bad = copy.deepcopy(result); bad['prefix_records'] *= 2; mutations.append(bad)
        bad = copy.deepcopy(result); bad['prefix_records'][0]['upper'].pop(); mutations.append(bad)
        for bad in mutations:
            with self.assertRaises(ValueError): F.check_selection(bad, [3], 'fixed', self.deadline)
        # Inconsistent synthetic upper bounds intentionally exercise no-history receipt.
        empty = F.first_unexcluded([3], lambda p, d: bounds(p, replacements={
            (i, j): -1. for i in range(3) for j in range(3) if i != j}), self.deadline)
        self.assertIsNone(empty['history'])

    def test_deadline_partial_prefix_and_legacy_missing_stays_missing(self):
        visited = []
        def fail(prefix, deadline):
            visited.append(prefix)
            if prefix: raise TimeoutError('fixed prefix interruption')
            return bounds(prefix)
        with self.assertRaises(TimeoutError): F.first_unexcluded([3, 3], fail, self.deadline)
        self.assertEqual(visited, [(), ((0, 1),)])
        with self.assertRaises(TimeoutError): F.first_unexcluded([3], fail, time.monotonic()-1)
        obj = dict(history=None, original_sources=[])
        spec = dict(schema=W.SCHEMAS[0], source_files={}, backend_identity='fixed')
        j = W.Journal(self.folder, self.deadline, 2**20)
        result = W.diagnose(obj, 'V2', spec, self.deadline, j)
        self.assertEqual(result['status'], 'SOURCE_MISSING')
        self.assertFalse(result['closed'])

    def test_z_native_reconstruction_graph_and_reception(self):
        import torch
        import genbab_cpu_ready as backend
        from b2_observed_worker import verify_graph
        import test_gate_decoupling as old
        G = old.G
        backend.probe()
        model, rows, th = old.fixture(2, classes=2)
        x = torch.zeros((1, 1), dtype=torch.float64)
        tensors = dict(center=x, lower=x-.25, upper=x+.25, rows=rows, thresholds=th)
        obj = dict(id='synthetic', source_kind='multilayer', history=None, experts=[3, 3],
            original_request=dict(model=dict(architecture=dict(classes=2))))
        catalog = self.folder/'catalog.json'; W.save(catalog, dict(objects=[obj]))
        spec = dict(schema=W.SCHEMAS[1], reconstruct_missing_history=True,
            catalog=dict(path=str(catalog), sha256=W.sha(catalog)), source_files={}, backend_identity='fixed')
        source_id = W.digest(dict(object=obj, sources={}, backend='fixed'))
        journal = W.Journal(self.folder, self.deadline, 4*2**20)
        selected = W.reconstruct_history(obj, model, tensors, source_id, self.deadline, journal)
        self.assertEqual(selected['history'], [[0, 1], [0, 1]])
        domain = G.domain_from_prefix_bounds(selected['history'], obj['experts'], selected['prefix_records'],
            source_id=source_id, expected_source_id=source_id)
        W.save(self.folder/'domain.json', dict(source_id=source_id, history=domain.history,
            margins=domain.margins, intervals=domain.intervals, domain_sha256=G.domain_digest(domain)))
        def prepare(variant):
            return (G.build_graph(model, domain.history, rows, th, domain, variant=variant,
                    input_features=1, expected_source_id=source_id),
                    G.extend_box(x, x-.25, x+.25, domain, variant=variant, expected_source_id=source_id))
        def verify(*args):
            result, attacks = verify_graph(*args)
            old.API_RESULTS.append(result)
            self.assertEqual(result['status'], 'verified')
            return result, attacks
        for group in ('V2', 'V3'):
            result = W.endpoints(G, domain, group, 1, self.deadline, prepare=prepare, verify=verify,
                replay=lambda *_: [], journal=journal)
            result.update(source_id=source_id, history=selected['history'], history_origin=F.MODE,
                selection_sha256=W.sha(self.folder/'history_selection.json'),
                property_count=1, domain_sha256=G.domain_digest(domain))
            call = dict(object_id=obj['id'], variant=group)
            W.check_received(self.folder, result, spec, call, self.deadline)
            for key, value in [('history', [[0, 2], [0, 1]]), ('selection_sha256', 'bad'),
                               ('history_origin', 'OLD_FIRST_FAILED')]:
                with self.assertRaises(ValueError):
                    W.check_received(self.folder, dict(result, **{key: value}), spec, call, self.deadline)
            self.assertFalse(result['complete_MoE']); self.assertFalse(result['formal_SAFE'])
        self.assertEqual(len(selected['prefix_records']), 2)
