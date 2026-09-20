"""No-solver controls for retrospective applicability; runnable with python -S."""
import copy
from fractions import Fraction as Q
import importlib.util
import json
import math
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('source_audit', HERE / 'audit_moe_main_source.py')
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


def fixture():
    sample = {'label': 2, **{k: {'sha256': k} for k in ('center', 'lower', 'upper')}}
    model = {'model_state': {'sha256': 'model'}, 'checkpoint': 'checkpoint', 'checkpoint_sha256': 'ckpt'}
    cfg = {'policy': 'fixed'}
    identity = a.expected_identity(sample, model, cfg, 2/255)
    e = {'identity': identity, 'request_id': a.sha(a.canonical(identity)),
         'route_coverage': {'coverage_complete': True, 'route_sets_exact': True,
                            'feasible_route_sets': [[0, 1], [0, 2]], 'candidate_experts': [0, 1, 2]},
         'numerical_safety': {'safe_positive_margin': 1e-7},
         'verdict': {'decision_tier': 'TIER2_F0', 'status': 'SAFE'},
         'tier2': {'pairs': [{'pair': p, 'property_rows': [
             {'property_index': k, 'status': 'SAFE', 'accepted_minimum': 1.,
              'solver_bound_kind': 'mip_dual_bound', 'solver_status': 0}
             for k in range(9)]} for p in [[0, 1], [0, 2]]]}}
    return e, sample, model, cfg


class Controls(unittest.TestCase):
    def test_import_has_no_act_model_solver_or_tensor_dependency(self):
        import sys
        self.assertFalse(any(k.split('.')[0] in ('act', 'torch', 'numpy', 'scipy', 'highspy') for k in sys.modules))

    def test_exact_containment_accepts_boundary_and_degenerate(self):
        self.assertEqual(a.inclusion((Q(0), Q(1)), (Q(0), Q(1))), (0, 0))
        self.assertEqual(a.inclusion((Q(1, 2), Q(1, 2)), (Q(1, 2), Q(1, 2))), (0, 0))

    def test_inward_endpoint_and_reversed_bounds(self):
        tiny = Q(1, 2**100)
        self.assertEqual(a.inclusion((tiny, Q(1)), (Q(0), Q(1))), (tiny, 0))
        self.assertEqual(a.inclusion((Q(0), Q(1)-tiny), (Q(0), Q(1))), (0, tiny))
        with self.assertRaises(ValueError): a.inclusion((Q(1), Q(0)), (Q(0), Q(1)))

    def test_literal_and_machine_epsilon_are_distinct(self):
        self.assertNotEqual(Q(2, 255), Q(2/255))
        x = .5; e = 2/255
        v = a.coordinate(x, x-e, x+e, e)
        self.assertNotEqual(v['gaps'][0], v['gaps'][1])

    def test_tiny_positive_radius_is_dropped_by_frozen_threshold(self):
        v = a.coordinate(0., 0., 1e-13, 1e-13)
        self.assertTrue(v['positive_radius_dropped'])
        self.assertTrue(any(v['gaps'][2]))

    def test_summary_survives_json_roundtrip(self):
        e = 2/255
        s = {'sample_rank': 0, 'dataset_index': 1, 'center': {}, 'lower': {}, 'upper': {}}
        r = a.summarize_box(s, {'center': [.5], 'lower': [.5-e], 'upper': [.5+e]}, e)
        self.assertEqual(json.loads(json.dumps(r)), r)

    def test_corrupt_box_and_nonfinite_fail(self):
        for lo, hi in ((.4, .6), (float('nan'), .6), (.6, .4)):
            with self.assertRaises(ValueError): a.coordinate(.5, lo, hi, 2/255)

    def test_hash_binding_rejects_changed_source(self):
        with self.assertRaises(ValueError): a.bound_ref(HERE/'audit_moe_main_source.py', '0'*64)

    def test_request_binding_accepts_only_same_identity(self):
        e, s, m, c = fixture()
        a.bind_identity(e, s, m, c, 2/255)
        for key in ('lower', 'property', 'model_state', 'config_sha256'):
            bad = copy.deepcopy(e); bad['identity'][key] = 'other'
            with self.assertRaises(ValueError): a.bind_identity(bad, s, m, c, 2/255)
        e['request_id'] = 'other'
        with self.assertRaises(ValueError): a.bind_identity(e, s, m, c, 2/255)

    def test_complete_record_inventory_is_not_new_proof(self):
        result = a.gain_basis(fixture()[0])
        self.assertEqual(len(result['obligations']), 18)
        self.assertFalse(result['source_complete_positive_proof'])
        self.assertEqual(result['historical_evidence_level'], 'HZ_POLICY_ACCEPTED')

    def test_missing_or_duplicate_pair_property_rejected(self):
        base = fixture()[0]
        for mutation in ('pair', 'property', 'duplicate', 'coverage'):
            e = copy.deepcopy(base)
            if mutation == 'pair': e['tier2']['pairs'].pop()
            elif mutation == 'property': e['tier2']['pairs'][0]['property_rows'].pop()
            elif mutation == 'duplicate': e['tier2']['pairs'][0]['property_rows'][0]['property_index'] = 1
            else: e['route_coverage']['coverage_complete'] = False
            with self.assertRaises(ValueError): a.gain_basis(e)

    def test_nonpositive_and_nonoptimal_not_accepted(self):
        for key, value in (('accepted_minimum', 0.), ('accepted_minimum', math.nan), ('solver_status', 1)):
            e = fixture()[0]; e['tier2']['pairs'][0]['property_rows'][0][key] = value
            with self.assertRaises(ValueError): a.gain_basis(e)

    def test_reuse_cannot_import_another_request(self):
        e = fixture()[0]
        v = e['tier2']['pairs'][0]['property_rows'][0]
        v.update(solver_bound_kind='scoped_tier1_interval', proof_sources=[
            {'expert': i, 'property_index': 0, 'scope': {'request_id': 'other'}} for i in (0, 1)])
        with self.assertRaises(ValueError): a.gain_basis(e)

    def test_archive_keeps_complete_cohort_and_historical_counts(self):
        r = json.loads(a.OUTPUT.read_bytes())
        self.assertEqual((r['terminals'], r['packages_decoded'], r['missing_packages_retained']), (900, 739, 161))
        self.assertEqual((r['unique_boxes_checked'], r['coordinates_checked']), (100, 307200))
        self.assertEqual([r['historical_totals'][k]['SAFE'] for k in ('adaptive', 'matched', 'legacy')], [179, 156, 141])
        self.assertEqual(len(r['gain_records']), 23)
        self.assertFalse(r['source_complete_positive_proof'])
        self.assertTrue(r['historical_statuses_preserved'])
        for k in ('new_solver_calls', 'new_model_forwards', 'new_source_propagations', 'new_output_bounds'):
            self.assertEqual(r[k], 0)


if __name__ == '__main__':
    unittest.main()
