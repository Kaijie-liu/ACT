"""Static arithmetic and binding tests; no source or numerical queries."""
import ast
from fractions import Fraction
import json
import sys
import unittest
from unittest.mock import patch

from scripts import analyze_full_resource_ledger as a
from scripts.assess_model_scaled_resources import shape_inventory, current_policy


class Ledger(unittest.TestCase):
    def inventory(self):
        return shape_inventory(3072,8,10,[128],[256,128])

    def test_full_counts_against_independent_static_derivation(self):
        expected = {
            1: ((26635776,2161406048,4273366725),6461408549,6415178465),
            8: ((15909396,1298519408,4273366725),5587795529,5544103505),
        }
        for width,(counts,total,portable) in expected.items():
            r=a.bounds(self.inventory(),width)
            self.assertEqual(tuple(r['production_counts_upper'][k] for k in a.COUNTERS),counts)
            self.assertEqual(r['production_work_upper'],total)
            self.assertEqual(r['portable_work_upper'],portable)
            self.assertFalse(r['execution_admitted'])

    def test_complete_roster_and_changed_factor(self):
        r=a.analyze(); v=r['inventory']; ref=r['reference_contract']
        self.assertEqual((v['pairs'],v['property_duties'],v['endpoints_upper'],v['candidates_upper']),
                         (28,252,504,1008))
        self.assertEqual([r['methods'][m]['pair_local_tiles_upper'] for m in ('reference','proposed')],[504,84])
        self.assertFalse(ref['consumer_only_ablation'])
        self.assertFalse(ref['same_generated_candidates_claimed'])
        self.assertTrue(ref['same_candidate_exact_differential_required'])

    def test_shared_work_not_shared_arithmetic_and_global_norm(self):
        x,y=(a.bounds(self.inventory(),w) for w in (1,8))
        self.assertEqual(x['production_counts_upper']['arithmetic'],y['production_counts_upper']['arithmetic'])
        self.assertEqual(x['phases']['objective_pullback'],y['phases']['objective_pullback'])
        self.assertGreater(x['phases']['dual_check_merge_and_support']['counts']['coefficient_visits'],
                           y['phases']['dual_check_merge_and_support']['counts']['coefficient_visits'])
        self.assertEqual(y['closure'],dict(router=dict(factors=4240,entries=399248),
            pair=dict(factors=11192,entries=2051792),distinct_scales_upper=3073,
            native_reference_levels_upper=2))

    def test_native_work_is_not_float_operations(self):
        r=a.bounds(self.inventory(),8)
        for p in ('native_preparation','native_candidates'):
            self.assertEqual(r['phases'][p]['counts']['arithmetic'],0)
        self.assertEqual(r['native_schedule']['adjoint_calls_upper'],10836)
        self.assertEqual(r['native_schedule']['forward_calls_upper'],10752)
        self.assertTrue(r['native_schedule']['global_sparse_bank_used'])

    def test_digit_formula_and_small_json_encoding(self):
        for n in (1,9,10,11,99,100,101,3072):
            self.assertEqual(a.digit_sum_below(n),sum(len(str(j)) for j in range(n)))
            raw=json.dumps([[j,'1/2'] for j in range(n)],separators=(',',':')).encode()
            self.assertEqual(len(raw),1+a.digit_sum_below(n)+9*n)
        for n in (0,-1,True,1.5):
            with self.assertRaises(ValueError): a.digit_sum_below(n)

    def test_constructive_bytes_not_a_trained_observation(self):
        w=a.storage(self.inventory())['encoding_witness']
        for key,value in dict(first_affine_rows=2176,id_digit_sum=11178,
            one_local_array_lower_bytes=38827,first_local_arrays_lower_bytes=84487552,
            single_bundle_lower_bytes=140203072,excess_bytes=5985344,
            two_bundle_lower_bytes=280406144).items():
            self.assertEqual(w[key],value,key)
        self.assertFalse(w['observed_checkpoint_values']); self.assertFalse(w['constructed_source'])

    def test_binary64_string_bound_and_candidate_components(self):
        for value in (0.,-0.,sys.float_info.max,-sys.float_info.max,
                      float.fromhex('0x0.0000000000001p-1022'),sys.float_info.min,
                      -float.fromhex('0x1.fffffffffffffp-1022')):
            self.assertLessEqual(len(str(Fraction(value))),342)
        r=a.storage(self.inventory()); c=r['candidate_components']
        self.assertEqual(c['native_binary64']['tile8_element_bytes_upper'],7538400)
        self.assertEqual(c['native_binary64']['full_element_bytes_upper'],474919200)
        self.assertEqual(c['general_checker']['full_element_bytes_upper'],3370701600)
        self.assertEqual(r['source_bytes_upper'],59909824)
        self.assertEqual(r['forest_value_components_upper'],3130874528)
        self.assertIsNone(r['full_forest_bytes_upper'])
        self.assertIsNone(r['full_archive_bytes_upper'])

    def test_no_quota_or_memory_closure_from_work_upper(self):
        r=a.analyze()
        for key in ('candidate_work_quota','absolute_work_ceiling','walltime_forecast','actual_counts'):
            self.assertIsNone(r[key])
        self.assertIsNone(r['memory']['peak_rss_upper'])
        self.assertIsNone(r['work_scope']['new_live_capture_or_fallback_work_upper'])
        for key in ('production_limits_changed','implementation_admitted','real_execution_admitted'):
            self.assertFalse(r[key])
        for key in ('new_source_runs','new_bound_queries','checkpoint_loads','gpu_calls'):
            self.assertEqual(r[key],0)
        self.assertEqual(current_policy()['visits'],8000000)

    def test_scope_and_identity_refusal(self):
        v=self.inventory(); v['expert_widths']=[3072,256,129,10]
        with self.assertRaisesRegex(ValueError,'registered topology'): a.bounds(v,8)
        for width in (0,2,9,True):
            with self.assertRaises(ValueError): a.bounds(self.inventory(),width)
        with patch.object(a,'PARENT_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'identity changed'): a.analyze()

    def test_analysis_imports_only_static_standard_library_path(self):
        tree=ast.parse((a.ROOT/'scripts/analyze_full_resource_ledger.py').read_bytes())
        actual=[]
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): actual.extend(v.name for v in node.names)
            elif isinstance(node,ast.ImportFrom): actual.append(node.module)
        self.assertEqual(set(actual),{'argparse','hashlib','json','pathlib','scripts.assess_model_scaled_resources'})
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))

    def test_saved_report_is_exactly_recomputable(self):
        p=a.ROOT/'docs/hz_full_resource_ledger_20261014.json'
        self.assertEqual(a.analyze(),json.loads(p.read_bytes()))


if __name__=='__main__': unittest.main()
