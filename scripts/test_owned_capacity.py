"""Static arithmetic/source-binding checks, never a source/endpoint execution."""
import ast
import json
import sys
import unittest
from unittest.mock import patch

from scripts import assess_owned_capacity as a


class Capacity(unittest.TestCase):
    def test_layout_bound_to_current_code(self):
        self.assertEqual(a.layout(),dict(scalar_header=16,working_rational_slot=2065,stored_rational=1040,
                                         root=16,prefix=128,record=72))

    def test_independent_lifetime_formula_and_cleanup(self):
        for k in (1,2,8):
            r=a.workspace(3,4,9,2,k); root=16*3+4*9
            self.assertEqual(r['prepare_peak'],root+2065*((k+1)*3+k*4))
            self.assertEqual(r['consume_peak'],root+2065*k*(3*3+2*4)+a.frame_bytes(2,k,0))
            self.assertEqual(r['events'][-1]['requested_live_bytes'],root)
            self.assertEqual(r['requested_peak'],max(r['prepare_peak'],r['seal_peak'],r['consume_peak']))

    def test_necessary_width8_refusal_not_upper_as_observation(self):
        r=a.analyze(); lo=r['methods']['8']['owned_lower']
        self.assertEqual(r['lower_counts'],dict(n=6696,f=3072,h=126632,m=12))
        self.assertEqual(lo['first_refusal']['event'],'prepare 6 objective factor vector')
        self.assertEqual(lo['first_refusal']['requested_live_bytes'],135466424)
        self.assertGreater(lo['first_refusal']['requested_live_bytes'],a.ARENA)
        self.assertTrue(r['methods']['8']['arena_refusal_proved_by_lower'])
        self.assertFalse(r['methods']['1']['arena_refusal_proved_by_lower'])
        self.assertEqual(r['methods']['8']['largest_minimum_pair_tile'],8)
        self.assertIsNone(lo['actual_peak'])
        self.assertTrue(lo['consumption_assumes_present_candidate'])
        self.assertIn('scoped_source/forest_endpoints.py',r['anchors'])

    def test_full_envelope_has_three_node_two_factor_banks(self):
        r=a.analyze(); one=r['methods']['1']['owned_conditional_upper']; eight=r['methods']['8']['owned_conditional_upper']
        self.assertEqual((one['prepare_peak'],eight['prepare_peak']),(147554464,892622984))
        self.assertEqual((one['consume_peak'],eight['consume_peak']),(254038304,2026307488))
        self.assertEqual(one['root_index_bytes'],856864)
        self.assertEqual(one['first_refusal']['event'],'prepare 1 objective factor vector')

    def test_frame_inventory_is_actual_binary_format(self):
        r=a.analyze(); one,eight=(r['methods'][str(w)] for w in (1,8))
        self.assertEqual(eight['native_frames']['per_pair_tile_widths'],[8,8,2])
        self.assertEqual((one['native_frames']['all_frames'],eight['native_frames']['all_frames']),(22921920,22868160))
        self.assertEqual((one['generic_frames']['all_frames'],eight['generic_frames']['all_frames']),(2831613120,2831559360))
        self.assertEqual(a.frame_bytes(2700,1,1),5618280)
        self.assertEqual(a.frame_bytes(2700,8,0),362944)
        for method in r['methods'].values():
            self.assertTrue(method['native_frames']['tile_upper_fits'])
            self.assertFalse(method['native_frames']['total_upper_fits'])
            self.assertFalse(method['generic_frames']['tile_upper_fits'])
            self.assertFalse(method['native_frames']['upper_overflow_proves_actual_overflow'])

    def test_upstream_still_refuses_before_new_index(self):
        r=a.analyze(); u=r['upstream']
        self.assertEqual((u['pair_aliases_alone'],u['mandatory_memberships'],u['membership_cap']),(86016,126632,65536))
        self.assertEqual(u['builder_plus_checker_first_weight_visit_lower'],13369344)
        self.assertTrue(u['new_index_is_after_old_source_check'])
        for k in ('new_quota',): self.assertIsNone(r[k])
        for k in ('cost_closed','execution_admitted','production_changed'): self.assertFalse(r[k])
        for k in ('real_model_runs','source_builds','solver_calls','gpu_calls','large_allocations'): self.assertEqual(r[k],0)

    def test_bound_identity_and_bad_counts(self):
        with patch.dict(a.ANCHORS,{'scoped_source/bounded_runtime.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.analyze()
        for k in (0,-1,True,9):
            with self.assertRaises(ValueError): a.workspace(3,4,9,2,k)
        for profile in (True,-1,2):
            with self.assertRaises(ValueError): a.frame_bytes(12,1,profile)

    def test_no_production_import_and_report_recomputes(self):
        tree=ast.parse((a.ROOT/'scripts/assess_owned_capacity.py').read_bytes())
        imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports|={v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names}
        self.assertEqual(imports,{'argparse','ast','hashlib','json','pathlib','struct'})
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(a.analyze(),json.loads((a.ROOT/'docs/hz_owned_capacity_20261014.json').read_bytes()))


if __name__=='__main__': unittest.main()
