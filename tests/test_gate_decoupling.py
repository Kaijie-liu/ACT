"""Fixed synthetic source/graph/receipt controls; no trained checkpoint or data."""
import copy
from dataclasses import replace
from fractions import Fraction
import importlib.util
import math
import operator
import os
from pathlib import Path
import sys
import time
import unittest

PATH = Path(__file__).resolve().parents[1]/'act/back_end/moe/gate_decoupling.py'
spec = importlib.util.spec_from_file_location('gate_decoupling_control_source', PATH)
G = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = G
spec.loader.exec_module(G)
SOURCE = 'a'*64
API_RESULTS = []


def prefix_records(history, margin=(-2., 2.), experts=3):
    result = []
    for depth, pair in enumerate(history):
        upper = [[i, j, 4.] for i in range(experts) for j in range(experts) if i != j]
        for row in upper:
            if row[:2] == list(pair): row[2] = margin[1]
            if row[:2] == list(reversed(pair)): row[2] = -margin[0]
        result.append(dict(prefix=[list(p) for p in history[:depth]], upper=upper,
                           guarantee='WHOLE_BOX_CROWN_NUMERICAL_UPPER'))
    return result


def domain(history, margin=(-2., 2.), experts=3):
    return G.domain_from_prefix_bounds(history, [experts]*len(history),
        prefix_records(history, margin, experts), source_id=SOURCE, expected_source_id=SOURCE)


def receipt(d, variant, status='verified', properties=2):
    return dict(variant=variant, source_id=SOURCE, domain_sha256=G.domain_digest(d),
        outcome=dict(status=status, actual_raw_spec=[([[float(i == j) for j in range(properties)]], [0.])
                     for i in range(properties)]))


class SourceControls(unittest.TestCase):
    def test_ordered_margin_direction_and_sigmoid_policy(self):
        d = domain(((0, 1),), (-3., 2.))
        self.assertEqual(d.margins, ((-3., 2.),))
        lo, hi = d.intervals[0]
        self.assertLess(lo, .05); self.assertGreater(hi, .88)
        self.assertLess(1-hi, .12); self.assertGreater(1-lo, .95)
        self.assertAlmostEqual(lo, 1/(1+math.exp(3)), places=14)
        self.assertAlmostEqual(hi, 1/(1+math.exp(-2)), places=14)
        tie = domain(((0, 1),), (0., 0.)).intervals[0]
        self.assertLess(tie[0], .5); self.assertGreater(tie[1], .5)
        self.assertEqual(G.sigmoid_interval(-1000., 1000.), (0., 1.))

    def test_missing_wrong_direction_nonfinite_duplicate_and_binding(self):
        h = ((0, 1), (1, 2)); original = prefix_records(h)
        mutations = []
        bad = copy.deepcopy(original); bad.pop(); mutations.append(bad)
        bad = copy.deepcopy(original); bad[1]['prefix'] = [[0, 2]]; mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['upper'][0][2] = None; mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['upper'][0][2] = -8.; mutations.append(bad)
        bad = copy.deepcopy(original); bad[0]['upper'].append(bad[0]['upper'][0]); mutations.append(bad)
        mutations.append(original+[copy.deepcopy(original[0])])
        for bad in mutations:
            with self.assertRaises(ValueError):
                G.domain_from_prefix_bounds(h, [3, 3], bad, source_id=SOURCE, expected_source_id=SOURCE)
        with self.assertRaises(ValueError):
            G.domain_from_prefix_bounds(h, [3, 3], original, source_id=SOURCE, expected_source_id='b'*64)
        with self.assertRaises(ValueError): domain(((1, 0),))

    def test_owned_ranges_not_aliased_and_domain_mutation_rejected(self):
        h = ((0, 1),); raw = prefix_records(h)
        d = G.domain_from_prefix_bounds(h, [3], raw, source_id=SOURCE, expected_source_id=SOURCE)
        old = G.domain_digest(d); raw[0]['upper'][0][2] = 100.
        self.assertEqual(G.domain_digest(d), old)
        for bad in (replace(d, intervals=((.4, .6),)), replace(d, source_id=''),
                    replace(d, history=((0, 2),))):
            with self.assertRaises(ValueError): G.validate_domain(bad, h, expected_source_id=SOURCE)

    def test_inner_endpoint_counterexample_exact(self):
        # A later ReLU destroys affinity in an earlier gate.
        f = lambda gate: abs(gate-Fraction(1, 2))-Fraction(1, 4)
        self.assertEqual(f(Fraction(0)), Fraction(1, 4))
        self.assertEqual(f(Fraction(1)), Fraction(1, 4))
        self.assertEqual(f(Fraction(1, 2)), -Fraction(1, 4))
        d = domain(((0, 1), (0, 1), (0, 1)))
        self.assertEqual(G.fixed_weights(d, 'V2_LO')[:2], (None, None))
        self.assertTrue(all(x is not None for x in G.fixed_weights(d, 'V3_MIDPOINT')))

    def test_receipt_all_endpoints_properties_scope_and_deadline(self):
        d = domain(((0, 1),))
        def receive(group, rows, deadline=None):
            return G.receive_diagnosis(d, group, rows, property_count=2,
                expected_source_id=SOURCE, deadline=time.monotonic()+5 if deadline is None else deadline)
        lo, hi = receipt(d, 'V2_LO'), receipt(d, 'V2_HI')
        self.assertFalse(receive('V2', [lo])['closed'])
        positive = receive('V2', [lo, hi])
        self.assertTrue(positive['original_history_closed'])
        self.assertFalse(positive['complete_MoE']); self.assertFalse(positive['formal_SAFE'])
        v3 = receive('V3', [receipt(d, 'V3_MIDPOINT')])
        self.assertTrue(v3['closed']); self.assertFalse(v3['original_history_closed'])
        negative = receive('V2', [lo, receipt(d, 'V2_HI', 'falsified')])
        self.assertFalse(negative['closed']); self.assertFalse(negative['original_UNSAFE'])
        bad = copy.deepcopy(hi); bad['outcome']['actual_raw_spec'].pop()
        for rows in ([lo, bad], [lo, lo], [lo, receipt(domain(((0, 2),)), 'V2_HI')]):
            with self.assertRaises(ValueError): receive('V2', rows)
        with self.assertRaises(TimeoutError): receive('V2', [lo, hi], time.monotonic()-1)
        with self.assertRaises(ValueError): receive('V2', [lo, hi], float('nan'))


def affine(weights, bias):
    import torch
    layer = torch.nn.Linear(len(weights[0]), len(weights), dtype=torch.float64)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weights, dtype=torch.float64))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float64))
    return layer.eval()


def fixture(depth, classes=10, residual=False, output_level=False):
    import torch
    from act.back_end.moe.multilayer import RoutedLayer
    from act.back_end.moe.multilayer_external import SequentialTop2
    from act.pipeline.moe.multilayer_training import TrainableResidual
    layers = []
    for index in range(depth):
        count = 1 if residual or index+1 < depth else classes
        router = affine([[.2], [-.2], [0.]], [0., 0., -4.])
        experts = [affine([[sign]]+[[0.]]*(count-1), [2. if count > 1 else .3]+[0.]*(count-1))
                   for sign in (1., -.5, 0.)]
        layers.append(RoutedLayer(router, experts, k=2, mode='selected_softmax'))
    if output_level:
        from act.back_end.moe.model import OutputLevelMoE
        from act.back_end.moe.schema import OutputLevelMoESpec, GateKind
        assert depth == 1
        model = OutputLevelMoE(layers[0].router, layers[0].experts,
            OutputLevelMoESpec(num_experts=3, top_k=2, gate=GateKind.SELECTED_SOFTMAX, normalized=True))
    elif residual:
        model = TrainableResidual(affine([[.5]], [.1]), layers,
            affine([[1.]]+[[0.]]*(classes-1), [2.]+[0.]*(classes-1)))
    else:
        model = SequentialTop2(layers)
    rows = torch.zeros((classes-1, classes), dtype=torch.float64)
    for i in range(classes-1): rows[i, 0], rows[i, i+1] = 1., -1.
    return model.eval(), rows, torch.arange(classes-1, dtype=torch.float64)/64


class GraphControls(unittest.TestCase):
    def setUp(self):
        import torch
        self.torch = torch
        torch.set_num_threads(1)

    def graph(self, model, h, rows, thresholds, d, variant):
        return G.build_graph(model, h, rows, thresholds, d, variant=variant,
                             input_features=1, expected_source_id=SOURCE)

    def test_original_embedding_and_final_endpoint_rule_1_2_3_layers(self):
        torch = self.torch
        from act.back_end.moe.multilayer_external import StaticPrefix, GuardedHistory
        for depth, residual in ((1, False), (2, False), (3, False), (2, True)):
            model, rows, th = fixture(depth, residual=residual)
            h = ((0, 1),)*depth; d = domain(h)
            ref, guarded = StaticPrefix(model, h), GuardedHistory(model, h, rows, th)
            low = self.graph(model, h, rows, th, d, 'V2_LO')
            high = self.graph(model, h, rows, th, d, 'V2_HI')
            for value in (-.25, 0., .25):
                x = torch.tensor([[value]], dtype=torch.float64)
                output, scores = ref._run(x)
                gates = [torch.sigmoid(r[:, :1]-r[:, 1:2]) for r in scores]
                for gate, (lo, hi) in zip(gates, d.intervals): self.assertTrue(lo <= float(gate) <= hi)
                extended = torch.cat((x, *gates[:-1]), dim=1)
                lo_out, lo_scores = low._run(extended); hi_out, hi_scores = high._run(extended)
                alpha = (gates[-1]-d.intervals[-1][0])/(d.intervals[-1][1]-d.intervals[-1][0])
                self.assertTrue(torch.allclose((1-alpha)*lo_out+alpha*hi_out, output, atol=1e-12, rtol=1e-12))
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(lo_scores, hi_scores)))
                self.assertEqual(low(extended).shape, (1, 9))
                self.assertTrue(bool((torch.minimum(low(extended), high(extended)) <= guarded(x)+1e-12).all()))

    def test_box_constants_ties_and_single_output_model(self):
        torch = self.torch
        model, rows, th = fixture(1, output_level=True); h = ((0, 1),); d = domain(h, (0., 0.))
        with torch.no_grad(): model.router.weight.zero_(); model.router.bias.zero_()
        from act.back_end.moe.genbab_pair import GenBaBGuardedProperties
        properties = [dict(q=r.tolist(), c=-float(t)) for r, t in zip(rows, th)]
        ref = GenBaBGuardedProperties(model, h[0], properties)
        x = torch.zeros((1, 1), dtype=torch.float64)
        for variant in G.VARIANTS:
            box = G.extend_box(x, x-.25, x+.25, d, variant=variant, expected_source_id=SOURCE)
            graph = self.graph(model, h, rows, th, d, variant)
            self.assertEqual(box['center'].shape, (1, 1))
            self.assertTrue(torch.allclose(graph(box['center']), ref(x), atol=1e-12, rtol=1e-12))
            # Every canonical tied pair remains representable.
        for pair in ((0, 1), (0, 2), (1, 2)):
            self.graph(model, (pair,), rows, th, domain((pair,), (0., 0.)), 'V2_LO')
        d2 = domain(h*2)
        box = G.extend_box(x, x-.25, x+.25, d2, variant='V2_LO', expected_source_id=SOURCE)
        self.assertEqual(box['center'].shape, (1, 2))
        self.assertEqual(float(box['lower'][0, 1]), d2.intervals[0][0])
        with self.assertRaises(ValueError):
            G.extend_box(x, x+1, x-1, d, variant='V2_LO', expected_source_id=SOURCE)

    def test_v3_positive_is_not_original_safety(self):
        torch = self.torch
        from act.back_end.moe.multilayer import RoutedLayer
        from act.back_end.moe.multilayer_external import SequentialTop2
        layer = RoutedLayer(affine([[1.], [-1.]], [0., 0.]),
            [affine([[-1.], [0.]], [.25, 0.]), affine([[1.], [0.]], [.25, 0.])],
            k=2, mode='selected_softmax')
        model = SequentialTop2([layer]).eval(); rows = torch.tensor([[1., -1.]], dtype=torch.float64)
        h = ((0, 1),); d = domain(h, experts=2)
        graph = self.graph(model, h, rows, torch.zeros(1, dtype=torch.float64), d, 'V3_MIDPOINT')
        x = torch.tensor([[1.]], dtype=torch.float64)
        self.assertGreater(float(graph(x)), 0.)
        self.assertLess(float(model(x)[0, 0]), 0.)

    def test_fx_v2_inner_products_v3_affine_relu_and_bad_suffix(self):
        torch = self.torch
        model, rows, th = fixture(2, residual=True); h = ((0, 1),)*2; d = domain(h)
        for variant, expected in (('V2_LO', 1), ('V3_MIDPOINT', 0)):
            graph = torch.fx.symbolic_trace(self.graph(model, h, rows, th, d, variant))
            products = [n for n in graph.graph.nodes if n.op == 'call_function' and n.target in (operator.mul, torch.mul)
                        and all(isinstance(a, torch.fx.Node) for a in n.args[:2])]
            self.assertEqual(len(products), expected)
            self.assertFalse(any(n.target == torch.sigmoid for n in graph.graph.nodes))
        model.head = torch.nn.Sequential(model.head, torch.nn.ReLU()).eval()
        with self.assertRaises(ValueError): self.graph(model, h, rows, th, d, 'V2_LO')

    def test_z_external_three_graphs(self):
        import genbab_cpu_ready as backend
        from b2_observed_worker import verify_graph
        backend.probe()
        torch = self.torch
        model, rows, th = fixture(2, classes=2); h = ((0, 1),)*2; d = domain(h)
        x = torch.zeros((1, 1), dtype=torch.float64)
        # The enclosing frozen control passes one shared deadline through all3.
        deadline = float(__import__('os').environ['GATE_CONTROL_DEADLINE'])
        receipts = []
        for variant in G.VARIANTS:
            graph = self.graph(model, h, rows, th, d, variant)
            box = G.extend_box(x, x-.25, x+.25, d, variant=variant, expected_source_id=SOURCE)
            outcome, _ = verify_graph(graph, box['lower'], box['upper'], len(rows), deadline)
            API_RESULTS.append(dict(variant=variant, **outcome))
            self.assertEqual(outcome['status'], 'verified')
            receipts.append(dict(variant=variant, source_id=SOURCE, domain_sha256=G.domain_digest(d), outcome=outcome))
        self.assertTrue(G.receive_diagnosis(d, 'V2', receipts[:2], property_count=1,
            expected_source_id=SOURCE, deadline=deadline)['original_history_closed'])
        self.assertFalse(G.receive_diagnosis(d, 'V3', receipts[2:], property_count=1,
            expected_source_id=SOURCE, deadline=deadline)['original_history_closed'])


class WorkerControls(unittest.TestCase):
    """No numerical calls: exercise the real endpoint orchestration/receivers."""
    def setUp(self):
        import gate_decoupling_worker as worker
        self.worker = worker
        self.folder = Path(os.environ['GATE_CONTROL_ROOT'])/self._testMethodName
        self.folder.mkdir()
        self.deadline = time.monotonic()+10
        self.journal = worker.Journal(self.folder, self.deadline, 1024*1024)

    def execute(self, statuses, *, fail_second=False):
        d = domain(((0, 1),)); calls = []
        def prepare(variant):
            return variant, dict(lower='lo', upper='hi')
        def verify(graph, lo, hi, count, deadline):
            calls.append((graph, deadline))
            self.assertEqual((lo, hi, count), ('lo', 'hi', 2))
            if fail_second and len(calls) == 2:
                raise TimeoutError('fixed second-endpoint interruption')
            return receipt(d, graph, statuses[len(calls)-1])['outcome'], None
        result = self.worker.endpoints(G, d, 'V2', 2, self.deadline,
            prepare=prepare, verify=verify, replay=lambda *_: [], journal=self.journal)
        self.assertEqual(calls, [('V2_LO', self.deadline), ('V2_HI', self.deadline)])
        return result

    def test_endpoints_one_deadline_and_ordinary_unknown_keeps_second(self):
        result = self.execute(['unknown', 'verified'])
        self.assertFalse(result['closed']); self.assertFalse(result['complete_MoE'])
        self.assertTrue((self.folder/'V2_LO.json').is_file())
        self.assertTrue((self.folder/'V2_HI.json').is_file())
        phases = [self.worker.load(p) for p in sorted(self.folder.glob('phase_*.json'))]
        self.assertEqual([p['phase'] for p in phases if p['state'] == 'ENTER'],
            ['V2_LO:construct', 'V2_LO:solve', 'V2_LO:replay', 'V2_HI:construct', 'V2_HI:solve', 'V2_HI:replay'])

    def test_interruption_retains_first_endpoint_without_aggregate(self):
        with self.assertRaises(TimeoutError): self.execute(['verified'], fail_second=True)
        self.assertTrue((self.folder/'V2_LO.json').is_file())
        self.assertFalse((self.folder/'V2_HI.json').exists())
        self.assertFalse((self.folder/'outcome.json').exists())
        phase = self.worker.load(sorted(self.folder.glob('phase_*.json'))[-1])
        self.assertEqual((phase['phase'], phase['state']), ('V2_HI:solve', 'ENTER'))

    def test_journal_cap_expiry_and_exclusive_publication(self):
        self.journal.put('retained.json', dict(value=1))
        with self.assertRaises(FileExistsError): self.journal.put('retained.json', dict(value=2))
        self.assertEqual(self.worker.load(self.folder/'retained.json'), dict(value=1))
        tiny = self.worker.Journal(self.folder, self.deadline, 1)
        with self.assertRaises(ValueError): tiny.put('over.json', dict(value=3))
        expired = self.worker.Journal(self.folder, time.monotonic()-1, 1024)
        with self.assertRaises(TimeoutError): expired.put('late.json', dict(value=4))
        self.assertFalse((self.folder/'over.json').exists()); self.assertFalse((self.folder/'late.json').exists())

    def test_receiver_source_and_counterfactual_scope(self):
        w = self.worker
        obj = dict(id='fixed', source_kind='output_mlp', history=[[0, 1]], properties=[{}, {}])
        catalog = self.folder/'catalog.json'; w.save(catalog, dict(objects=[obj]))
        spec = dict(catalog=dict(path=str(catalog), sha256=w.sha(catalog)), source_files={}, backend_identity='fixed-stub')
        call = dict(object_id='fixed', variant='V3')
        sid = w.digest(dict(object=obj, sources={}, backend='fixed-stub'))
        d = G.domain_from_prefix_bounds(obj['history'], [3], prefix_records(obj['history']),
            source_id=sid, expected_source_id=sid)
        row = receipt(d, 'V3_MIDPOINT'); row['source_id'] = sid
        w.save(self.folder/'V3_MIDPOINT.json', row)
        row = w.load(self.folder/'V3_MIDPOINT.json')
        w.save(self.folder/'domain.json', dict(source_id=sid, history=d.history, margins=d.margins,
            intervals=d.intervals, domain_sha256=G.domain_digest(d)))
        value = G.receive_diagnosis(d, 'V3', [row], property_count=2, expected_source_id=sid, deadline=self.deadline)
        value.update(source_id=sid, domain_sha256=G.domain_digest(d), property_count=2, endpoints=[row])
        w.check_received(self.folder, value, spec, call, self.deadline)
        for fields in (dict(status='POSITIVE'), dict(original_history_closed=True),
                       dict(source_id='b'*64), dict(property_count=1), dict(endpoints=[])):
            with self.assertRaises(ValueError): w.check_received(self.folder, dict(value, **fields), spec, call, self.deadline)
        with self.assertRaises(TimeoutError): w.check_received(self.folder, value, spec, call, time.monotonic()-1)


if __name__ == '__main__':
    unittest.main()
