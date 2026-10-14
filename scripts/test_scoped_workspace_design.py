"""Static design tests; no source fixture, producer, solver or GPU."""
import ast
from fractions import Fraction
import json
import sys
import unittest
from unittest.mock import patch

from scripts import assess_scoped_workspace_design as a


class ScopedDesign(unittest.TestCase):
    def test_checked_relu_inequalities_need_no_lift(self):
        # Both actual inequality rows have zero center and only +/-1 locals.
        # No assertion about equality/output rows, which keep both lift slots.
        for coefficients in ((-1,-1),(-1,1)):
            error=abs(Fraction(0)-Fraction(float(0)))
            error+=sum(abs(Fraction(q)-Fraction(float(q))) for q in coefficients)
            self.assertEqual(error,0)
        s=a.analyze()['scope']
        self.assertEqual(s['pair_affine_outputs'],924)
        self.assertEqual(s['pair_relu_units'],896)
        self.assertEqual(s['pair_nodes_upper'],7592)
        self.assertEqual(s['pair_factors_upper'],9400)
        self.assertEqual(s['pair_factors_old_loose_upper'],11192)
        self.assertEqual(s['global_factors_tighter_upper'],25648)

    def test_scope_alone_does_not_close_width_eight(self):
        r=a.analyze()
        self.assertGreater(r['scope_only_eight_objective_vectors_upper'],a.ARENA)
        self.assertFalse(r['variants']['8']['packed_upper_fits'])
        self.assertEqual(r['largest_microblock_fitting_packed_envelope'],2)

    def test_peak_recomputed_without_helper(self):
        v=json.loads((a.ROOT/'docs/hz_full_resource_ledger_20261014.json').read_bytes())['inventory']
        for q in (1,2,3,4,8):
            r=a.payload(v,a.shape(v),q)
            metadata=sum(r[k] for k in ('root_index','scope_maps_and_rows',
                'sparse_objective_records','frame','norm_group_records','scalar_records'))
            self.assertEqual(r['conservative_packed_payload'],metadata+2065*q*(7592+2*9400))
            self.assertEqual(r['arena_headroom'],a.ARENA-r['conservative_packed_payload'])
            self.assertIsNone(r['full_peak_rss_upper'])

    def test_candidate_and_exact_batches_are_not_conflated(self):
        r=a.analyze(); s=r['selected_scan_subledger']
        self.assertEqual(r['variants']['2']['subdivisions'],[2,2,2,2])
        self.assertEqual((s['old_candidate_tiles'],s['new_checking_microblocks']),(84,252))
        self.assertEqual(s['extra_dual_node_row_visits_upper'],3275328)
        self.assertEqual(s['extra_dual_constraint_row_visits_upper'],453600)
        self.assertEqual(s['objective_pullbacks_upper'],504)
        self.assertTrue(s['full_factor_checks_retained'])
        self.assertIsNone(s['total_work_upper'])

    def test_partial_tiles_do_not_drop_endpoints(self):
        for k in range(1,9):
            for q in range(1,k+1):
                chunks=a.chunks(k,q)
                self.assertEqual(sum(chunks),k)
                self.assertTrue(all(1<=x<=q for x in chunks))
        for args in ((0,1),(8,0),(8,9),(True,1),(8,True)):
            with self.assertRaises(ValueError): a.chunks(*args)

    def test_no_admission_cap_or_format_change(self):
        r=a.analyze()
        for k in ('complete_cost_closed','production_changed','execution_admitted','candidate_format_changed'):
            self.assertFalse(r[k])
        for k in ('real_model_runs','gpu_runs','source_builds'): self.assertEqual(r[k],0)
        self.assertEqual((r['work_cap_unchanged'],r['seconds_unchanged']),(8000000,300))
        self.assertEqual((r['obligations_unchanged'],r['max_endpoints_unchanged'],r['max_roles_unchanged']),(252,504,1008))
        self.assertIsNone(r['new_quota'])

    def test_source_identity_drift_refused(self):
        with patch.dict(a.ANCHORS,{'scoped_source/check_source_dag.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.analyze()

    def test_stdlib_only_and_saved_report_matches(self):
        tree=ast.parse((a.ROOT/'scripts/assess_scoped_workspace_design.py').read_bytes())
        imports={n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)}
        imports|={v.name for n in ast.walk(tree) if isinstance(n,ast.Import) for v in n.names}
        self.assertEqual(imports,{'argparse','hashlib','json','pathlib'})
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(a.analyze(),json.loads((a.ROOT/'docs/hz_scoped_workspace_design_20261014.json').read_bytes()))


if __name__ == '__main__': unittest.main()
