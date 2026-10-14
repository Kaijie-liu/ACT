"""Static schema/formula controls only; no parser, checkpoint or worker runs."""
import json
from math import prod
import sys
import unittest
from unittest.mock import patch

from scripts import assess_source_lifetime_protocol as a


class SourceLifetimeDesign(unittest.TestCase):
    def test_tensor_directory_is_complete_without_instantiation(self):
        rows=a.tensor_roster()
        self.assertEqual(len(rows),52)
        self.assertEqual(len({r['name'] for r in rows}),52)
        self.assertEqual(sum(prod(r['shape']) for r in rows),6961368)
        self.assertEqual(next(r['shape'] for r in rows if r['name']=='experts.7.5.weight'),[10,128])
        self.assertEqual(next(r['shape'] for r in rows if r['name']=='router.3.bias'),[8])

    def test_snapshot_is_added_not_hidden_in_final_store(self):
        r=a.snapshot_cost(101,211,307)
        self.assertEqual(r['attempt_store_plus_both_snapshots_upper'],619)
        self.assertEqual(r['snapshot_full_scan_read_bytes'],624)
        self.assertEqual(r['snapshot_hash_bytes'],624)
        self.assertFalse(r['source_path_immutable_assumed'])
        self.assertFalse(r['stat_is_immutability_proof'])
        self.assertFalse(r['parser_and_gather_reads_included'])

    def test_arena_formula_does_not_expand_aliases_or_claim_rss(self):
        r=a.parser_arenas(37)
        self.assertEqual(sum(r['typed_components'].values()),313*37)
        self.assertEqual(r['graph_edges_upper'],4*37)
        self.assertFalse(r['max_memo_id_is_allocation_size'])
        self.assertFalse(r['aliases_expanded'])
        self.assertFalse(r['arbitrary_reduce_executed'])
        self.assertFalse(r['implementation_correspondence_proved'])
        self.assertIsNone(r['complete_parser_runtime_upper'])

    def test_noncontiguous_storage_and_independent_pass_both_cost(self):
        r=a.capture_counts(1000,100,a.DATA_BLOB+200)
        self.assertEqual(r['checkpoint_gather_requested_bytes_upper'],16*6961368)
        self.assertEqual(r['checkpoint_gather_stride_terms_upper'],4*6961368)
        self.assertEqual(r['sequential_parser_payload_upper'],313*200)
        self.assertEqual(r['checkpoint_directory_read_bytes_upper'],2*(121*1000+65557))
        self.assertFalse(r['requested_read_bytes_are_physical_io'])
        self.assertFalse(r['storage_aliases_or_noncontiguous_views_rejected_by_default'])
        self.assertEqual(r['independently_read_generated_source_bytes'],55715520)
        self.assertFalse(r['span_comparison_rereads_included'])
        self.assertEqual(r['historical_center_hash_payload_bytes'],49152)

    def test_invalid_static_dimensions_and_typed_addressability(self):
        for v in (True,0,-1,1.5,None):
            with self.assertRaises(ValueError): a.parser_arenas(v)
        with self.assertRaises(ValueError): a.parser_arenas(2**64)
        with self.assertRaises(ValueError): a.parser_arenas((2**64-1)//313+1)
        with self.assertRaises(ValueError): a.capture_counts(10,11,a.DATA_BLOB+1)
        with self.assertRaises(ValueError): a.capture_counts(10,1,a.DATA_BLOB)
        with self.assertRaises(ValueError): a.snapshot_cost(True,1,1)

    def test_input_conversion_is_not_silently_redefined(self):
        r=a.input_contract()
        self.assertEqual(r['string_rational_component_bits_upper'],13607)
        self.assertEqual(r['four_literal_record_bytes_upper'],13672)
        self.assertEqual(r['center_plus_literal_unreduced_bits'],[13669,13668])
        self.assertEqual(r['clipping_compare_product_bits_upper'],27276)
        self.assertEqual(r['constructed_endpoints_compare_product_bits_upper'],27337)
        self.assertFalse(r['frozen_totensor_bit_equivalence_proved'])
        self.assertFalse(r['native_256_value_control_run'])
        self.assertFalse(r['float_radius_replaced_with_rational_two_over_255'])
        self.assertIsNone(r['new_request_bit_gate'])

    def test_worker_lifetime_and_global_budget_are_conditions_not_results(self):
        r=a.worker_contract()
        self.assertEqual(r['workers'],['SOURCE','PRODUCTION','PORTABLE'])
        self.assertEqual(r['one_absolute_deadline_seconds'],300)
        self.assertFalse(r['deadline_reset']); self.assertFalse(r['quota_reset'])
        self.assertIn('sum',r['production_native_and_exact'])
        self.assertIsNone(r['os_cleanup_latency_upper'])
        self.assertFalse(r['process_group_sample_proves_containment'])
        self.assertFalse(r['full_worker_lifetime_established'])
        self.assertFalse(r['missing_partial_or_late_evidence_can_accept'])

    def test_source_anchor_drift_rejected(self):
        a.verify_files()
        with patch.dict(a.ANCHORS,{'scoped_source/capture.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.verify_files()
        with patch.dict(a.ENV_ANCHORS,{'lib/python3.12/pickle.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.verify_files()

    def test_report_no_execution_or_acceptance_upgrade(self):
        r=a.analyze()
        self.assertEqual(r['invariant']['request_seconds'],300)
        self.assertEqual(r['current_policy_unchanged']['visits'],8000000)
        for key in ('production_changed','implementation_authorized','execution_admitted',
                    'mathematical_checks_removed'):
            self.assertFalse(r[key])
        for key in ('model_runs','dataset_loads','checkpoints_loaded','gpu_calls','solver_calls'):
            self.assertEqual(r[key],0)
        self.assertIsNone(r['new_quota'])
        self.assertFalse(r['closure']['full_cost_upper_closed'])
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(json.loads(json.dumps(r)),
            json.loads((a.ROOT/'docs/hz_source_lifetime_protocol_20261014.json').read_bytes()))


if __name__=='__main__': unittest.main()
