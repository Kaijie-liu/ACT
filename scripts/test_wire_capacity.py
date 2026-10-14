"""Static capacity assertions; no source, proof, model or numerical run."""
import ast
from fractions import Fraction
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

from scripts import assess_wire_capacity as a


def compact(d):
    return json.dumps(d,sort_keys=True,separators=(',',':')).encode()


class Capacity(unittest.TestCase):
    def inventory(self):
        return a.shape_inventory(3072,8,10,[128],[256,128])

    def test_current_binary_layout_component_arithmetic(self):
        one,eight=(a.storage(self.inventory(),w) for w in (1,8))
        self.assertEqual(one['forest_bytes_upper'],17277720)
        self.assertEqual(one['forest_parts']['inline_and_tail_entries'],48048*16)
        self.assertEqual(one['native_wire_tile_bytes_upper'],11323368)
        self.assertEqual(eight['native_wire_tile_bytes_upper'],11055408)
        self.assertEqual((one['single_component_sum_upper'],eight['single_component_sum_upper']),
                         (88510912,88242952))
        self.assertEqual((one['two_copy_component_sum_upper'],eight['two_copy_component_sum_upper']),
                         (177021824,176485904))
        self.assertEqual((one['tile_count'],eight['tile_count']),(504,84))
        self.assertEqual(eight['per_pair_tile_widths'],[8,8,2])

    def test_native_header_formula_matches_actual_small_metadata(self):
        h='a'*64
        record=dict(slot=h,query=h,missing=False,lower=dict(proposed='0',zero='0'),zero='EXPLICIT_ZERO_V1')
        self.assertEqual(len(compact(record)),228)
        for w in (1,2,8):
            for eq,ine in ((0,0),(9,10),(100,2700),(2700,2700)):
                header=dict(schema='WIRE_CANDIDATE_TILE_V1',
                    binding={k:h for k in ('source','forest','request','factors','contract','rows')},
                    counts=dict(eq=eq,ine=ine),records=[record]*w,logical=h)
                self.assertEqual(len(compact(header)),a.native_header(w,len(str(eq)),len(str(ine))))
                self.assertLessEqual(len(compact(header)),a.native_header(w))
        self.assertEqual((a.native_header(1),a.native_header(8)),(855,2458))

    def test_logical_tile_component_bound_preserves_both_roles(self):
        # Small metadata/strings only; no full source or candidate bank.
        h='a'*64
        for eq,ine in ((0,0),(1,0),(0,1),(2,3)):
            candidate=dict(query=h,lower='0',equality=['1/2']*eq,inequality=['-1/2']*ine)
            zero=dict(query=h,lower='0',equality=['0']*eq,inequality=['0']*ine)
            record=dict(slot=h,proposed=candidate,zero=zero)
            self.assertLessEqual(len(compact(record)),331+349*(eq+ine))
            for w in (1,8):
                self.assertLessEqual(len(compact([record]*w)),w*(332+349*(eq+ine))+1)
        one,eight=(a.storage(self.inventory(),w) for w in (1,8))
        self.assertEqual(one['logical_candidate_tile_upper'],942633)
        self.assertEqual(eight['logical_candidate_tile_upper'],7541057)
        self.assertTrue(eight['logical_tile_upper_exceeds_4mib'])
        self.assertTrue(eight['physical_tiles_do_not_replace_logical_caps'])

    def test_counterexample_is_exact_binary64_not_actual_source(self):
        r=a.logical_obstruction(self.inventory()); q=Fraction(1,2**53)
        self.assertEqual(Fraction(float(q)),q)
        self.assertEqual(str(q),r['coefficient_literal'])
        self.assertEqual((r['one_local_array_bytes'],r['first_local_arrays_bytes'],r['excess_before_other_fields']),
                         (84907,184757632,50539904))
        for n in (1,9,10,99,100):
            self.assertEqual(len(compact([[j,str(q)] for j in range(n)])),
                             1+a.digit_sum(n)+(len(str(q))+6)*n)
        self.assertFalse(r['actual_values_observed']); self.assertFalse(r['full_source_constructed'])

    def test_new_work_lower_is_not_old_upper_or_new_quota(self):
        r=a.changed_work(self.inventory())
        self.assertEqual(r['two_lazy_first_weight_scans']['work_lower'],13369344)
        self.assertEqual(r['repeated_input_node_decodes']['added_work'],26738688)
        self.assertEqual(r['first_source_row_pass_hash_read_bytes_upper'],4616880128)
        self.assertGreater(r['two_lazy_first_weight_scans']['work_lower'],a.current_policy()['visits'])
        self.assertIsNone(r['complete_wire_production_work_upper'])
        self.assertIsNone(r['complete_wire_portable_work_upper'])
        self.assertFalse(r['old_full_work_totals_reusable'])
        self.assertTrue(r['native_129_evaluations_do_not_reread_source_rows'])

    def test_no_admission_from_component_uppers(self):
        r=a.analyze()
        self.assertEqual(r['mandatory_roots'],dict(lower=126632,retained_cap=65536,refusal_proved=True))
        for method in r['methods'].values():
            self.assertIsNone(method['complete_archive_upper'])
            self.assertFalse(method['upper_overflow_proves_actual_overflow'])
        for k in ('new_work_quota','absolute_work_ceiling','walltime_forecast'):
            self.assertIsNone(r[k])
        self.assertIsNone(r['memory']['complete_peak_rss_upper'])
        self.assertIsNone(r['memory']['complete_native_peak_upper'])
        for k in ('execution_admitted','caps_changed','G1_G6_closed','historical_results_changed'):
            self.assertFalse(r[k])
        for k in ('model_loads','source_builds','solver_calls','native_calls','gpu_calls'):
            self.assertEqual(r[k],0)

    def test_finite_driver_terminal_cannot_be_upscaled_unchanged(self):
        row=dict(pair=[0,1],other=0,lower='1',positive=True)
        self.assertEqual(len(compact(row)),52)
        r=a.analyze()['control_driver_terminal']
        self.assertEqual(r['result_array_bytes_lower'],13357)
        self.assertGreater(r['result_array_bytes_lower'],r['terminal_cap_bytes'])
        self.assertTrue(r['inline_full_roster_refused'])

    def test_bad_scope_width_or_anchor_refused(self):
        v=self.inventory(); v['classes']=9
        with self.assertRaisesRegex(ValueError,'registered'): a.storage(v,8)
        for w in (0,2,9,True):
            with self.assertRaises(ValueError): a.storage(self.inventory(),w)
        for args in ((0,), (True,), (8,5,1)):
            with self.assertRaises(ValueError): a.native_header(*args)
        with patch.dict(a.ANCHORS,{'scoped_source/wire_forest.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.analyze()

    def test_historical_artifacts_bound_without_reinterpreting_them(self):
        r=a.analyze()
        for name in ('hz_full_resource_ledger_20261014.json','hz_lossless_evidence_design_20261014.json'):
            path='docs/'+name
            self.assertEqual(hashlib.sha256((a.ROOT/path).read_bytes()).hexdigest(),r['anchors'][path])
        old=json.loads((a.ROOT/'docs/hz_full_resource_ledger_20261014.json').read_bytes())
        self.assertEqual(old['methods']['proposed']['production_work_upper'],5587795529)
        self.assertNotEqual(r['methods']['8']['single_component_sum_upper'],110611136)

    def test_analysis_is_static_and_report_recomputable(self):
        tree=ast.parse((a.ROOT/'scripts/assess_wire_capacity.py').read_bytes())
        imports=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom): imports.add(node.module)
        self.assertEqual(imports,{'argparse','hashlib','json','pathlib','scripts.assess_model_scaled_resources'})
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(a.analyze(),json.loads((a.ROOT/'docs/hz_wire_capacity_20261014.json').read_bytes()))


if __name__=='__main__': unittest.main()
