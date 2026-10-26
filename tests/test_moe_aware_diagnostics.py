"""Ordinary stage5a aggregation/worker tests. Native cases use synthetic MoEs."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'scripts'), str(ROOT/'tests')]
import moe_aware_diagnostics as D
import gate_decoupling_worker as W
from scoped_proof.io import save

SOURCE = 'a'*64
G = D.gate_module()


def domain():
    margins = ((-.4, .4), (-2., 2.))
    return G.GateDomain(SOURCE, ((0, 1), (0, 1)), margins,
                        tuple(G.sigmoid_interval(*m) for m in margins))


def endpoint_rows(parent, group):
    rows = []
    for i, cell in enumerate(D.cells_for(parent, group)):
        scoped = D.conditional_domain(parent, cell)
        for variant in ('V2_LO', 'V2_HI'):
            rows.append(dict(cell_index=i, cell=cell, variant=variant,
                source_id=scoped.source_id, parent_domain_sha256=G.domain_digest(parent),
                domain_sha256=G.domain_digest(scoped), initial_lower=[.1, .2], outcome=None))
    return rows


def mb_row(on=True):
    return dict(source_id=SOURCE, history=[[0, 1], [0, 1]], cells=[],
        objective_count=2, guard_count=4, use_guards=on,
        lower=[.1, .2], zero_lower=[-.1, -.2], beta_count=8,
        beta_nonzero=2 if on else 0,
        selected_beta=[[.5 if on else 0., 0., 0., 0.]]*2,
        complete_MoE=False, formal_SAFE=False)


class ReceiptTests(unittest.TestCase):
    def test_fixed_call_table_and_mechanism_gate_denominators(self):
        from moe_aware_stage5a import call_table, comparison
        records = json.loads((ROOT/'docs/moe_aware_bab_records_20261026.json').read_text())
        table = call_table(records)
        self.assertEqual(len(table), 103)
        self.assertEqual(sum(r['variant'].startswith('MB_') for r in table), 22)
        self.assertEqual(sum(r['variant'].startswith('MC_') for r in table), 81)
        self.assertEqual(len({(r['object_id'], r['variant']) for r in table}), 103)
        indexed = {}
        for i in range(5):
            indexed[i, 'off'] = dict(status='UNKNOWN', accepted=True, score=-.1, identity=str(i))
            indexed[i, 'on'] = dict(status='POSITIVE', accepted=True, score=.1, identity=str(i))
        result = comparison(list(range(6)), indexed, 'off', 'on')
        self.assertTrue(result['passes']); self.assertEqual(result['population'], 6)
        self.assertEqual(result['complete_score_pairs'], 5); self.assertEqual(len(result['gained']), 5)
        # Missing and other-history answers cannot supply an extra gain.
        indexed[0, 'on']['identity'] = 'other'
        self.assertFalse(comparison(list(range(6)), indexed, 'off', 'on')['passes'])

    def test_all_cells_endpoints_and_properties(self):
        parent = domain(); rows = endpoint_rows(parent, 'MC_2'); end = time.monotonic()+10
        result = D.aggregate_mc(parent, 'MC_2', rows, 2, end)
        self.assertTrue(result['closed']); self.assertEqual(result['score'], .1)
        self.assertFalse(result['formal_SAFE']); self.assertEqual(result['expected_endpoints'], 4)
        missing = D.aggregate_mc(parent, 'MC_2', rows[:-1], 2, end)
        self.assertFalse(missing['closed']); self.assertIsNone(missing['score'])
        rows[0]['initial_lower'][0] = -.1
        self.assertFalse(D.aggregate_mc(parent, 'MC_2', rows, 2, end)['closed'])
        # A later BaB proof may close the row without pretending its initial
        # bound was positive or converting its scalar into a property bound.
        rows[0]['outcome'] = dict(status='verified', actual_raw_spec=D.raw_spec(2))
        result = D.aggregate_mc(parent, 'MC_2', rows, 2, end)
        self.assertTrue(result['closed']); self.assertEqual(result['score'], -.1)
        self.assertIn('NOT_PROPERTY_LOWER', result['metric_kind'])

    def test_cell_source_property_mutations_and_deadline(self):
        parent = domain(); original = endpoint_rows(parent, 'MC_4'); end = time.monotonic()+10
        mutations = []
        bad = copy.deepcopy(original); bad[0]['source_id'] = 'b'*64; mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['cell'][1] += .1; mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['initial_lower'].pop(); mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['initial_lower'][0] = float('nan'); mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['outcome'] = dict(status='verified', actual_raw_spec=[]); mutations.append(bad)
        mutations.append(original+[copy.deepcopy(original[0])])
        for bad in mutations:
            with self.assertRaises(ValueError): D.aggregate_mc(parent, 'MC_4', bad, 2, end)
        with self.assertRaises(TimeoutError): D.aggregate_mc(parent, 'MC_4', original, 2, time.monotonic()-1)

    def test_mb_multiplier_and_full_property_receipt(self):
        row = mb_row(); h = row['history']; end = time.monotonic()+10
        result = D.aggregate_mb(row, SOURCE, h, 'MB_ON', 2, 4, end)
        self.assertTrue(result['closed'])
        row['lower'][0] = -.05
        self.assertFalse(D.aggregate_mb(row, SOURCE, h, 'MB_ON', 2, 4, end)['closed'])
        for key, value in (('source_id', 'b'*64), ('formal_SAFE', True),
                           ('selected_beta', [[-.1]*4]*2), ('lower', [.1]), ('use_guards', False)):
            bad = mb_row(); bad[key] = value
            with self.assertRaises(ValueError): D.aggregate_mb(bad, SOURCE, h, 'MB_ON', 2, 4, end)
        with self.assertRaises(ValueError): D.aggregate_mb(mb_row(), SOURCE, h, 'MB_OFF', 2, 4, end)

    def test_standard_library_worker_dispatch_and_identity(self):
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as directory:
            folder = Path(directory); parent = domain()
            obj = dict(id='synthetic', history=[[0, 1], [0, 1]], experts=[3, 3],
                       original_request=dict(model=dict(architecture=dict(classes=3))))
            save(folder/'catalog.json', dict(objects=[obj]))
            spec = dict(schema=W.SCHEMAS[2], catalog=dict(path=str(folder/'catalog.json'),
                sha256=W.sha(folder/'catalog.json')), source_files={}, backend_identity='b'*64)
            sid = W.digest(dict(object=obj, sources={}, backend='b'*64))
            parent = G.GateDomain(sid, parent.history, parent.margins, parent.intervals)
            save(folder/'domain.json', dict(source_id=sid, history=parent.history,
                margins=parent.margins, intervals=parent.intervals, domain_sha256=G.domain_digest(parent)))
            rows = endpoint_rows(parent, 'MC_2')
            for row in rows:
                save(folder/f"cell_{row['cell_index']}_{row['variant']}_initial.json", row)
            end = time.monotonic()+10
            value = dict(**D.aggregate_mc(parent, 'MC_2', rows, 2, end), source_id=sid,
                group='MC_2', history=obj['history'], property_count=2,
                domain_sha256=G.domain_digest(parent), endpoints=rows)
            W.check_received(folder, value, spec, dict(object_id='synthetic', variant='MC_2'), end)
            bad = copy.deepcopy(value); bad['property_count'] = 1
            with self.assertRaises(ValueError):
                W.check_received(folder, bad, spec, dict(object_id='synthetic', variant='MC_2'), end)
            value['endpoints'].pop()
            with self.assertRaises(ValueError):
                W.check_received(folder, value, spec, dict(object_id='synthetic', variant='MC_2'), end)


@unittest.skipUnless(os.environ.get('MOE_DIAGNOSTIC_NATIVE') == '1', 'registered isolated engine synthetic controls')
class NativeTests(unittest.TestCase):
    def test_sliced_native_bounds_complete_coverage_and_deadline(self):
        import torch
        from test_gate_decoupling import fixture
        torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
        model, rows, th = fixture(2, classes=3)
        tensors = dict(center=torch.zeros((1, 1)), lower=torch.full((1, 1), -.25),
            upper=torch.full((1, 1), .25), rows=rows, thresholds=th)
        deadline = float(os.environ['MOE_CONTROL_DEADLINE'])
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as directory:
            folder = Path(directory); journal = W.Journal(folder, deadline, W.CAP)
            result = D.sliced_calls(model, tensors, domain(), 'MC_2', deadline, journal)
            self.assertTrue(result['closed']); self.assertEqual(len(result['endpoints']), 4)
            self.assertEqual(len(result['initial_lower']), 2)
            # Check the exact same JSON boundary as the production reception.
            rebuilt = D.aggregate_mc(domain(), 'MC_2', json.loads(json.dumps(result['endpoints'])), 2, deadline)
            self.assertEqual(rebuilt['score'], result['score'])
            print('M-C synthetic full-slice initial lower:', result['initial_lower'], flush=True)
        # Inject one interrupted evaluator. The already-produced prefix is
        # preserved, but there is no aggregate or accepted complete result.
        with tempfile.TemporaryDirectory(dir=ROOT.parent/'results') as directory:
            folder = Path(directory); journal = W.Journal(folder, deadline, W.CAP)
            calls = []
            def interrupted(graph, box, end):
                self.assertEqual(end, deadline); calls.append(1)
                if len(calls) == 2: raise TimeoutError('fixed partial control')
                return [.1, .2]
            with self.assertRaises(TimeoutError):
                D.sliced_calls(model, tensors, domain(), 'MC_2', deadline, journal, bound=interrupted)
            self.assertEqual(len(list(folder.glob('*_initial.json'))), 1)
            self.assertFalse((folder/'initial_aggregate.json').exists())


if __name__ == '__main__':
    unittest.main()
