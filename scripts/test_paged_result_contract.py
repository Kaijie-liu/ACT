"""Static proposal schema/length tests only; no production checker imports."""
import copy
import json
import unittest
from unittest.mock import patch

from scripts import assess_paged_result_contract as a


class Controls(unittest.TestCase):
    def test_row_length_matches_small_materialized_schema(self):
        row = dict(pair=[1, 2], other=0, lower='-12/37', positive=False)
        self.assertEqual(a.length(row), len(json.dumps(row, sort_keys=True, separators=(',', ':')).encode()))
        self.assertEqual(a.schemas()['fraction_chars'], 4936)
        self.assertEqual(a.length(a.schemas()['row']), 52+4936)

    def test_final_short_page_and_no_duplicate_result_storage(self):
        for duties, pages in ((1, 1), (8, 1), (9, 2), (252, 32)):
            s = a.schemas(duties=duties)
            self.assertEqual(s['pages'], pages)
            self.assertLessEqual(s['page_bytes'], pages*a.length(s['page']))
        s = a.schemas(added_code_files=0)
        self.assertEqual((s['artifact_files'], s['whole_request_files'], s['bundle_members']), (351, 356, 349))
        s = a.schemas()
        self.assertEqual((s['artifact_files'], s['whole_request_files'], s['bundle_members']), (353, 358, 351))

    def test_context_separates_forest_identity_from_index(self):
        s = a.schemas(); context = s['context']; index = s['index']
        self.assertIn('forest', context); self.assertIn('forest_index', context)
        self.assertIn('roster', context); self.assertNotIn('candidates', context)
        self.assertIn('candidates', index); self.assertNotIn('bundle', index)
        self.assertNotIn('manifest', index)

    def test_versioned_reports_keep_cost_and_guarantee_fields(self):
        s = a.schemas()
        self.assertNotEqual(s['manifest']['schema'], 'DIRECT_LIFECYCLE_MANIFEST_V1')
        self.assertFalse(s['summary']['native_float_proof'])
        self.assertFalse(s['summary']['checkpoint_correspondence_checked'])
        for report in (s['candidate_report'], s['candidate_report']['replay'], s['parent_receipt']['accepted']):
            self.assertNotIn('result', report)
            self.assertIn('summary', report); self.assertIn('result_ref', report)
        for field in ('phases', 'snapshots', 'work', 'archive', 'deadline'):
            self.assertIn(field, s['candidate_report'])

    def test_nonpackaged_binding_charged_separately(self):
        base = a.schemas(added_implementation_files=2)
        protocol = a.schemas(added_implementation_files=3)
        self.assertEqual(base['whole_request_files'], protocol['whole_request_files'])
        self.assertEqual(base['code_files'], protocol['code_files'])
        self.assertEqual(protocol['implementation_files'], 47)
        self.assertGreater(a.length(protocol['invocation']), a.length(base['invocation']))

    def test_success_per_file_bounds_not_full_resource_admission(self):
        r = a.analyze()
        self.assertTrue(all(r['capacity'][x] for x in ('all_success_reports_fit', 'page_fits', 'index_fits')))
        self.assertEqual(r['files']['pages'], 32)
        self.assertEqual(r['unchanged']['seconds'], 300)
        self.assertEqual(r['unchanged']['work'], 8000000)
        self.assertFalse(r['full_capacity_admitted']); self.assertFalse(r['runtime_changed'])
        self.assertIsNone(r['open_contract']['full_rss_upper'])
        self.assertEqual(r['mathematical_checks_removed'], 0)
        for key in ('source_builds','candidate_builds','model_runs','gpu_runs','solver_calls'):
            self.assertEqual(r[key], 0)

    def test_invalid_sizes_and_bound_report_drift_rejected(self):
        for arg in (dict(duties=True), dict(duties=253), dict(tiles=85), dict(added_code_files=3),
                    dict(added_implementation_files=4), dict(added_implementation_files=1)):
            with self.assertRaises(ValueError): a.schemas(**arg)
        load = a.old.prior.load
        changed = copy.deepcopy(load(a.PARENT)); changed['runtime_changed'] = True
        with patch.object(a.old.prior, 'load', side_effect=lambda p: changed if p == a.PARENT else load(p)):
            with self.assertRaisesRegex(ValueError, 'archive contract drift'): a.analyze()


if __name__ == '__main__':
    unittest.main()
