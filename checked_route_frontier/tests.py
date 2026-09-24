import copy
from fractions import Fraction as F
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from checked_route_frontier import build as producer
from checked_route_frontier.check import check, check_frontier, reconstruct_margin
from checked_route_frontier.evidence import aggregate
from checked_route_frontier.fixtures import analytic
from scoped_proof.evidence import context, lower_bound, reconstruct_lp, roster
from scoped_source.build import build as exhaustive
from source_construction_lab.fixtures import document
from source_enclosure.format import identity, unpack


def deadline():
    return time.monotonic() + 300


def make(doc, candidates=None):
    end = deadline()
    pre = producer.prefix(doc, expected_source_sha256=identity(doc), deadline=end)
    if candidates is None:
        candidates = producer.propose_final_affine(doc, pre, deadline=end)
    return producer.finish(doc, pre, candidates, expected_source_sha256=identity(doc), deadline=end)


def inspect(doc, bundle):
    return check(doc, bundle, expected_source_sha256=identity(doc), deadline=deadline())


def scope(doc):
    return {k: v for k, v in doc['request'].items()
            if k not in ('top_k', 'gate', 'tie_policy', 'training')}


def output_candidates(doc, bundle, invocation='control'):
    candidates = {}
    pairs = {tuple(p['pair']): p for p in bundle['pairs']}
    for index, obligation in enumerate(roster(scope(doc))):
        pair = pairs.get(tuple(obligation['pair']))
        if pair is None:
            continue
        row = next(r for r in pair['obligations']['rows'] if r['competitor'] == obligation['competitor'])
        lp = reconstruct_lp(pair['base'], row)
        candidates[index] = {'schema': 'SCOPED_LP_CANDIDATE_V1', 'status': 'CANDIDATE',
            'context': context(scope(doc), identity(doc), identity(bundle), invocation, index, obligation, lp),
            'certificate': {'lp_sha256': identity(lp), 'inequality_dual': ['0'] * len(lp['b']),
                            'equality_dual': ['0'] * len(lp['h'])}}
    return candidates


class FrontierControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.doc = analytic()
        cls.bundle = make(cls.doc)

    def test_strict_dominance_skips_only_checked_empty_routes(self):
        result = inspect(self.doc, self.bundle)
        self.assertEqual(result['frontier']['retained_pairs'], 1)
        self.assertEqual(result['frontier']['needed_experts'], [0, 1])
        self.assertEqual(result['original_output_obligations'], 12)
        self.assertEqual(result['output_obligations'], 2)
        self.assertEqual(result['excluded_output_obligations'], 10)
        self.assertFalse(result['complete_output_positive_proof'])

    def test_252_roster_kept_even_when_only_nine_bounds_needed(self):
        doc = analytic(experts=8, classes=10)
        bundle = make(doc)
        result = aggregate(scope(doc), doc, bundle, output_candidates(doc, bundle),
            invocation='control', proposal_complete=True, deadline=deadline())
        self.assertEqual((result['required'], result['discharged_by_exclusion'], result['positive_bounds']), (252, 243, 9))
        self.assertEqual(len(result['rows']), 252)
        self.assertTrue(result['complete_output_positive_proof'])

    def test_ties_nonpositive_and_missing_certificates_retain_routes(self):
        doc = analytic('tied')
        for candidates in (None, []):
            bundle = make(doc, candidates)
            result = inspect(doc, bundle)
            self.assertEqual(result['frontier']['retained_pairs'], 6)
            self.assertEqual(result['frontier']['needed_experts'], [0, 1, 2, 3])
        pre = producer.prefix(self.doc, expected_source_sha256=identity(self.doc), deadline=deadline())
        cand = producer.propose_final_affine(self.doc, pre, deadline=deadline())
        for value in cand:
            value['certificate'] = None
        result = check_frontier(self.doc, pre, cand, expected_source_sha256=identity(self.doc), deadline=deadline())
        self.assertEqual(result['retained_pairs'], 6)

    def test_multiple_routes_and_boundary_tie_remain_covered(self):
        doc = analytic('crossing')
        bundle = make(doc)
        result = inspect(doc, bundle)
        retained = {tuple(p['pair']) for p in bundle['pairs']}
        self.assertEqual(retained, {(0, 1), (0, 2), (1, 2)})
        from itertools import combinations
        for x in (F(-1), F(-1, 7), F(0), F(1, 7), F(1)):
            scores = [x, -x, F(0), F(-3)]
            legal = {p for p in combinations(range(4), 2)
                     if all(scores[i] >= scores[j] for i in p for j in range(4) if j not in p)}
            self.assertTrue(legal <= retained)
        positive = aggregate(scope(doc), doc, bundle, output_candidates(doc, bundle),
            invocation='control', proposal_complete=True, deadline=deadline())
        self.assertTrue(positive['complete_output_positive_proof'])
        self.assertFalse(result['route_changing_established'])  # no witness protocol here

    def test_last_affine_dual_recovers_shared_factor_cancellation(self):
        doc = analytic('tied')
        pre = producer.prefix(doc, expected_source_sha256=identity(doc), deadline=deadline())
        state = pre['router']['steps'][-1]['state']
        lp = producer.margin_lp(state, 0, 1)
        zero = {'lp_sha256': identity(lp), 'inequality_dual': ['0'] * len(lp['b']),
                'equality_dual': ['0'] * len(lp['h'])}
        self.assertLess(F(lower_bound(lp, zero)['checked_lower_bound']), 0)
        candidate = producer.propose_final_affine(doc, pre, deadline=deadline())[0]
        self.assertEqual(F(lower_bound(lp, candidate['certificate'])['checked_lower_bound']), 0)
        self.assertEqual(lp, reconstruct_margin(state, 0, 1))

    def test_retained_matrices_and_traces_equal_exhaustive_reference(self):
        for doc in (self.doc, analytic('crossing'), analytic('tied'),
                    document(experts=3, classes=4, width=2, depth=2)):
            with self.subTest(source=identity(doc)):
                bundle = make(doc)
                old = exhaustive(doc, expected_source_sha256=identity(doc), deadline=deadline())
                self.assertEqual(bundle['prefix']['input'], old['input'])
                self.assertEqual(bundle['prefix']['router'], old['networks'][0])
                by_name = {v['name']: v for v in old['networks']}
                for trace in bundle['experts']:
                    self.assertEqual(trace, by_name[trace['name']])
                by_pair = {tuple(v['pair']): v for v in old['pairs']}
                for pair in bundle['pairs']:
                    self.assertEqual(pair, by_pair[tuple(pair['pair'])])
                inspect(doc, bundle)

    def test_dimensions_depth_zero_radius_and_constant_outputs(self):
        for e, c, w, d in ((2, 2, 1, 0), (3, 5, 3, 1), (4, 2, 2, 3)):
            doc = analytic(experts=e, classes=c, width=w, depth=d)
            doc['request']['radius'] = '0'
            inspect(doc, make(doc))

    def test_wrong_binding_direction_dual_and_duplicate_rejected(self):
        pre = self.bundle['prefix']
        original = self.bundle['route_candidates']
        for mode in ('source', 'router', 'direction', 'self', 'bool', 'duplicate', 'lp', 'dual', 'fake_claim'):
            with self.subTest(mode=mode):
                candidates = copy.deepcopy(original)
                first = candidates[0]
                if mode == 'source': first['source_sha256'] = '0' * 64
                if mode == 'router': first['router_sha256'] = '0' * 64
                if mode == 'direction': first['higher'], first['lower'] = first['lower'], first['higher']
                if mode == 'self': first['lower'] = first['higher']
                if mode == 'bool': first['higher'] = False
                if mode == 'duplicate': candidates.append(copy.deepcopy(first))
                if mode == 'lp': first['certificate']['lp_sha256'] = '0' * 64
                if mode == 'dual': first['certificate']['equality_dual'].append('0')
                if mode == 'fake_claim': first['certificate']['claimed_positive'] = True
                with self.assertRaises(ValueError):
                    check_frontier(self.doc, pre, candidates, expected_source_sha256=identity(self.doc), deadline=deadline())

    def test_inward_input_wrong_layer_and_exclusion_transplant_rejected(self):
        for mode in ('input', 'layer', 'frontier', 'pair', 'expert', 'property', 'property_value', 'borrowed'):
            with self.subTest(mode=mode):
                bad = copy.deepcopy(self.bundle)
                if mode == 'input': bad['prefix']['input']['hz']['Gc']['data'][0] = '1/2'
                if mode == 'layer': bad['prefix']['router']['steps'][-1]['state']['hz']['c'][0] = '999'
                if mode == 'frontier': bad['frontier']['pairs'].pop()
                if mode == 'pair': bad['pairs'].pop()
                if mode == 'expert': bad['experts'].pop()
                if mode == 'property': bad['pairs'][0]['obligations']['rows'].pop()
                if mode == 'property_value': bad['pairs'][0]['obligations']['rows'][0]['offset'] = '999'
                if mode == 'borrowed': bad['lower_bound_certificates'] = [{'old': 'positive'}]
                with self.assertRaises(ValueError):
                    inspect(self.doc, bad)

    def test_partial_output_and_incomplete_proposal_never_close(self):
        candidates = output_candidates(self.doc, self.bundle)
        for mode in ('missing', 'incomplete', 'none'):
            partial = copy.deepcopy(candidates)
            if mode == 'missing': partial.pop(next(iter(partial)))
            if mode == 'none': partial = {}
            result = aggregate(scope(self.doc), self.doc, self.bundle, partial,
                invocation='control', proposal_complete=mode != 'incomplete', deadline=deadline())
            self.assertEqual(result['status'], 'NOT_CLOSED')
            self.assertEqual(result['required'], 12)

    def test_output_candidate_wrong_run_and_excluded_index_rejected(self):
        candidates = output_candidates(self.doc, self.bundle)
        with self.assertRaises(ValueError):
            aggregate(scope(self.doc), self.doc, self.bundle, candidates,
                invocation='other', proposal_complete=True, deadline=deadline())
        candidates[11] = copy.deepcopy(next(iter(candidates.values())))
        with self.assertRaises(ValueError):
            aggregate(scope(self.doc), self.doc, self.bundle, candidates,
                invocation='control', proposal_complete=True, deadline=deadline())

    def test_nonpositive_output_is_not_unsafe(self):
        doc = copy.deepcopy(self.doc)
        doc['request']['margin'] = '4'
        bundle = make(doc)
        result = aggregate(scope(doc), doc, bundle, output_candidates(doc, bundle),
            invocation='control', proposal_complete=True, deadline=deadline())
        self.assertEqual(result['status'], 'NOT_CLOSED')
        self.assertEqual(len(result['nonpositive']), 2)

    def test_expired_or_late_check_rejects(self):
        for end in (time.monotonic() - 1, time.monotonic() + 301, float('inf')):
            with self.assertRaises((ValueError, TimeoutError)):
                check(self.doc, self.bundle, expected_source_sha256=identity(self.doc), deadline=end)
        from checked_route_frontier.check import check_outputs
        with patch('scoped_source.graph.time.monotonic', return_value=10) as now:
            def late(*args):
                value = check_outputs(*args)
                now.return_value = 12
                return value
            with patch('checked_route_frontier.check.check_outputs', side_effect=late):
                with self.assertRaises(TimeoutError):
                    check(self.doc, self.bundle, expected_source_sha256=identity(self.doc), deadline=11)

    def test_checker_independent_and_fresh_python_no_site(self):
        with patch.object(producer, 'prefix', side_effect=AssertionError('producer')), \
             patch.object(producer, 'margin_lp', side_effect=AssertionError('producer')), \
             patch.object(producer, 'finish', side_effect=AssertionError('producer')):
            inspect(self.doc, self.bundle)
        with tempfile.TemporaryDirectory(dir='/data1/Kane/MOE') as folder:
            path = Path(folder) / 'control.json'
            path.write_text(json.dumps([self.doc, self.bundle, output_candidates(self.doc, self.bundle)]))
            code = '''import json,sys,time
from checked_route_frontier.evidence import aggregate
d,b,c=json.load(open(sys.argv[1]))
s={k:v for k,v in d['request'].items() if k not in ('top_k','gate','tie_policy','training')}
r=aggregate(s,d,b,{int(k):v for k,v in c.items()},invocation='control',proposal_complete=True,deadline=time.monotonic()+30)
assert r['complete_output_positive_proof']
assert not any(n.split('.')[0] in ('numpy','scipy','torch','act','highspy') for n in sys.modules)
assert 'checked_route_frontier.build' not in sys.modules
print(json.dumps(r))
'''
            value = subprocess.run([sys.executable, '-S', '-c', code, str(path)],
                capture_output=True, text=True, timeout=35)
            self.assertEqual(value.returncode, 0, value.stderr)
            self.assertEqual(json.loads(value.stdout)['required'], 12)


if __name__ == '__main__':
    unittest.main()
