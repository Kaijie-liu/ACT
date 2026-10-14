"""Arithmetic and evidence-binding tests, not production or timing controls."""
from fractions import Fraction
import json
import sys
import unittest
from unittest.mock import patch

from scripts import assess_runtime_lifetime as a


class RuntimeLifetime(unittest.TestCase):
    def test_expression_tree_does_not_apply_a_postcheck_early(self):
        bits=a.exact_subledger()['expression_bits']
        self.assertEqual(bits['binary_add'],(16385,16384))
        self.assertEqual(bits['fused_multiply_add'],(24577,24576))
        self.assertEqual(bits['relu_four_absolute_differences'],(65540,65536))
        self.assertGreater(bits['relu_four_absolute_differences'][0],8*8192+2)

    def test_magnitude_rules_against_small_exact_arithmetic(self):
        values=[Fraction(n,d) for n in (-7,-1,0,1,7) for d in (1,3,7)]
        for x in values:
            for y in values:
                for op, value in [('add',x+y),('sub',x-y),('mul',x*y)]+([('div',x/y)] if y else []):
                    n,d=a.magnitude(op,(3,3),(3,3))
                    self.assertLessEqual(value.numerator.bit_length(),n)
                    self.assertLessEqual(value.denominator.bit_length(),d)
        self.assertEqual(a.magnitude('compare',(3,5),(7,11)),(14,12))
        with self.assertRaises(ValueError): a.magnitude('pow',(3,3),(3,3))

    def test_old_literal_and_candidate_gates_are_not_conflated(self):
        r=a.exact_subledger()
        self.assertEqual(r['old_decimal_parse_magnitude_bits'],
            {'candidate_2470_chars':8206,'request_4096_chars':13607})
        self.assertTrue(r['preserve_request_domain'])
        self.assertIsNone(r['new_request_bit_limit'])

    def test_global_cache_is_not_a_freed_temporary(self):
        r=a.exact_subledger()['hash_cache']
        self.assertEqual(r['key_integer_magnitude_payload_upper'],32*2**20)
        self.assertIn('fresh checker process',r['conditions'])
        self.assertFalse(r['includes_python_objects_or_allocator'])
        self.assertFalse(r['clear_cache_implemented'])

    def test_batching_changes_calls_not_total_coordinate_upper(self):
        one,eight=a.native_subledger(1),a.native_subledger(8)
        self.assertEqual((one['tile_count'],eight['tile_count']),(504,84))
        self.assertEqual((one['spmm_calls'],eight['spmm_calls']),(129528,21588))
        self.assertEqual(one['coo_axpy_calls_upper'],872075756160)
        self.assertEqual(eight['coo_axpy_calls_upper'],145345959360)
        self.assertEqual(one['axpy_coordinate_updates_upper'],eight['axpy_coordinate_updates_upper'])
        self.assertEqual(eight['summed_tile_width'],504)
        self.assertNotEqual(eight['summed_tile_width'],84*8)
        with self.assertRaises(ValueError): a.native_subledger(True)

    def test_cumulative_tensor_payload_is_not_rss(self):
        for w,peak in ((1,512768),(8,4102144)):
            r=a.native_subledger(w)
            self.assertEqual(r['largest_two_spmm_output_payload_bytes'],peak)
            self.assertEqual(r['cumulative_two_spmm_output_payload_bytes'],53461739520)
            self.assertEqual(r['persistent_operator_tensor_payload_upper'],330081664)
            self.assertFalse(r['array_cumulative_is_peak'])
            self.assertFalse(r['logical_reads_are_physical_dram'])
            self.assertIsNone(r['complete_native_peak_upper'])

    def test_binding_drift_rejected_without_native_execution(self):
        a.verify_files()
        with patch.dict(a.ANCHORS,{'act/back_end/moe/factory.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.verify_files()
        with patch.object(a,'SLICES',(('changed',0x20166e0,1458,'0'*64),)):
            with self.assertRaisesRegex(ValueError,'slice'): a.verify_files()

    def test_capture_and_supervisor_do_not_inherit_closure(self):
        r=a.analyze()
        self.assertEqual(len(r['capture']['historical_bindings']),3)
        self.assertIsNone(r['capture']['selected_live_object'])
        self.assertIsNone(r['capture']['complete_capture_upper'])
        self.assertFalse(r['capture']['file_size_bounds_unpickling'])
        self.assertFalse(r['capture']['cpu_map_location_excludes_cuda_probe'])
        self.assertFalse(r['supervision']['descendants_cannot_outlive_phase_proved'])

    def test_report_policy_and_no_execution(self):
        r=a.analyze()
        self.assertEqual(r['current_policy_unchanged']['work_bits'],8192)
        self.assertEqual(r['current_policy_unchanged']['visits'],8000000)
        self.assertEqual(r['invariant']['request_seconds'],300)
        self.assertFalse(r['resource_limits_changed']); self.assertFalse(r['execution_admitted'])
        self.assertIsNone(r['full_runtime_upper']); self.assertIsNone(r['new_quota'])
        for key in ('model_runs','gpu_calls','solver_calls','source_builds','native_calls'):
            self.assertEqual(r[key],0)
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(json.loads(json.dumps(r)),json.loads((a.ROOT/'docs/hz_runtime_lifetime_20261014.json').read_bytes()))


if __name__ == '__main__': unittest.main()
