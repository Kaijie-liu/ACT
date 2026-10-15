"""Static arithmetic/identity controls only; no extra implementation micro-runs."""
import copy
import unittest
from unittest.mock import patch

from scripts import assess_paged_execution_resources as a


class Controls(unittest.TestCase):
    def test_frozen_implementation_identity(self):
        bindings = a.load(a.CONTROLS)['bindings']
        a.check_bindings(bindings)
        wrong = dict(bindings)
        wrong[next(iter(wrong))] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'implementation drift'):
            a.check_bindings(wrong)

    def test_schema_matches_archived_controls(self):
        result = a.analyze()
        self.assertEqual(result['success_layout'], a.load(a.CONTROLS)['success_layout'])
        self.assertEqual(result['success_layout']['whole_request_files'], 358)
        self.assertEqual(result['success_layout']['bundle_members'], 351)

    def test_inherited_source_and_account_binding(self):
        archive = copy.deepcopy(a.load(a.ARCHIVE))
        a.check_parent_links(archive)
        for name in (a.WORK, a.FLOOR, 'scoped_source/capture.py'):
            wrong = copy.deepcopy(archive)
            wrong['anchors'][name] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'inherited layout/account drift'):
                a.check_parent_links(wrong)

    def test_complete_archive_is_not_stream_admission(self):
        r = a.analyze()['conditional_success_archive']
        self.assertEqual(r['binary64']['request_upper'], 89165848)
        self.assertEqual(r['general_rational']['request_upper'], 2897857048)
        self.assertTrue(r['binary64']['whole_upper_fits_archive_cap'])
        for v in r.values():
            self.assertFalse(v['candidate_upper_fits_stream_cap'])
            self.assertFalse(v['upper_excess_proves_actual_overflow'])
            self.assertIsNone(v['actual_bytes'])

    def test_page_reads_include_index_and_all_online_passes(self):
        r = a.analyze()['result_file_reads']
        self.assertEqual(r['child_passes'] + r['parent_passes'], r['online_passes'])
        self.assertEqual(r['online_logical_bytes_upper'], 12672990)
        self.assertEqual(r['offline_audit_passes_separate'], 4)
        self.assertFalse(r['physical_device_io'])

    def test_counted_work_not_redefined_as_machine_cost(self):
        r = a.analyze()['counted_work']
        self.assertEqual(r['new_result_modules_increment'], 0)
        self.assertFalse(r['total_machine_cost_increment_is_zero'])
        self.assertEqual(r['postcapture_upper'], dict(dense_reference=89433531656, pair_scoped=165108619256))

    def test_false_success_layout_rejected(self):
        controls = copy.deepcopy(a.load(a.CONTROLS))
        controls['success_layout']['metadata_upper'] -= 1
        original = a.load
        with patch.object(a, 'load', side_effect=lambda p: controls if p == a.CONTROLS else original(p)):
            with self.assertRaisesRegex(ValueError, 'schema drift'):
                a.analyze()

    def test_memory_and_failure_cost_are_not_promoted(self):
        r = a.analyze()
        self.assertTrue(r['lifetime']['reader_old_and_new_decoded_pages_can_overlap'])
        self.assertIsNone(r['lifetime']['full_heap_upper'])
        self.assertIsNone(r['lifetime']['full_rss_upper'])
        self.assertFalse(r['bounded_failures']['incomplete_terminal_cost_complete'])

    def test_authority_and_refusals_unchanged(self):
        r = a.analyze()
        self.assertEqual(r['refusals']['source_two_check_lower'], 13916160)
        self.assertTrue(r['refusals']['source_lower_exceeds_cap'])
        self.assertTrue(r['refusals']['membership_lower_exceeds_cap'])
        self.assertEqual(r['unchanged']['seconds'], 300)
        self.assertFalse(r['runtime_changed'])
        self.assertFalse(r['full_capacity_admitted'])
        self.assertIsNone(r['new_quota'])
        for key in ('source_builds', 'candidate_builds', 'worker_runs', 'model_runs', 'gpu_runs', 'solver_calls'):
            self.assertEqual(r[key], 0)


if __name__ == '__main__':
    unittest.main()
