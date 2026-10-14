"""Static design checks, not controls for an implemented V4 checker."""
import ast
import hashlib
import json
import sys
import unittest
from unittest.mock import patch

from scripts import assess_unique_root_protocol as a


class UniqueRootProtocol(unittest.TestCase):
    def test_full_obligation_roster_not_reduced(self):
        v=a.inventory()
        self.assertEqual((v['pairs'],v['property_duties'],v['endpoints_upper'],v['candidates_upper']),
                         (28,252,504,1008))
        for w in (1,8):
            c=a.exact_core(v,w)
            self.assertEqual(c['proposed_and_zero_scalar_checks'],2*504*2700)

    def test_unique_facts_and_reference_uses_are_separate(self):
        r=a.analyze(); i=r['identity']
        self.assertEqual(i['canonical_root_upper'],19496)
        self.assertEqual(i['membership_upper'],136232)
        self.assertFalse(i['membership_checks_removed'])
        self.assertFalse(i['duplicate_owner_merging'])
        self.assertFalse(i['replays_old_json'])
        self.assertFalse(i['old_hash_equals_new_hash'])
        for m in r['methods'].values():
            self.assertGreater(m['core']['role_index_access_upper_per_pass'],136232)

    def test_complete_native_store_inventory_arithmetic(self):
        for w,b,members in ((1,96769184,648),(8,96661664,228)):
            r=a.archive(a.inventory(),w)
            self.assertEqual(r['native_single_store_bytes'],b)
            self.assertEqual(sum(r['component_bytes'].values()),b)
            self.assertEqual(r['members_upper'],members)
            self.assertEqual(sum(r['member_counts'].values()),members)
            self.assertEqual(r['component_bytes']['forest'],16861088)
            self.assertEqual(r['component_bytes']['diagnostics'],4428096)
            self.assertEqual(r['component_bytes']['auxiliary'],4034560)
            self.assertTrue(r['single_native_upper_fits_old_archive'])
            self.assertFalse(r['two_native_upper_fits_old_archive'])
            self.assertFalse(r['larger_upper_proves_actual_overflow'])

    def test_source_chunk_count_from_shapes(self):
        v=a.inventory()
        self.assertEqual(a.source_chunks(v),6+8*11+1)
        self.assertEqual(a.archive(v,8)['member_counts']['source_chunks'],95)

    def test_no_free_copy_or_relocation(self):
        for w in (1,8):
            r=a.archive(a.inventory(),w); io=a.io_contract(r); b=r['native_single_store_bytes']
            self.assertEqual(r['native_two_store_bytes'],2*b)
            self.assertEqual(io['read_bytes_upper'],4*b)
            self.assertEqual(io['write_bytes_upper'],b)
            self.assertTrue(io['one_file_pass_is_not_one_hash_pass'])
            per_pass=2*b+sum(io['hash_semantic_frames_upper_per_pass'].values())+64*io['hash_context_count_upper_per_pass']
            self.assertEqual(io['hash_input_bytes_upper'],5*per_pass)
            self.assertGreater(io['hash_input_bytes_upper'],10*b)
            self.assertTrue(io['payload_second_contexts_must_partition'])
            self.assertTrue(io['full_store_root_hashes_index_not_expanded_inventory'])
            self.assertFalse(io['hash_updates_outside_named_schedule_allowed'])
            self.assertEqual(io['optional_relocation']['additional_hash_bytes'],2*b+128*r['members_upper'])
            self.assertEqual(io['optional_relocation']['retained_bytes'],2*b)
            self.assertFalse(io['optional_relocation']['authorized'])

    def test_native_and_generic_proposals_have_different_envelopes(self):
        self.assertEqual(a.rational_record(4096),1040)
        self.assertEqual(a.rational_record(8192),2064)
        for w,total in ((1,1501114784),(8,1501007264)):
            r=a.archive(a.inventory(),w)
            self.assertEqual(r['generic_rational_single_store_bytes'],total)
            self.assertEqual(total-r['native_single_store_bytes'],(1040-8)*504*2700)
            self.assertIn('advisory lower=0',r['generic_profile_conditions'])
            self.assertGreater(total,128*a.MIB)
        for n in (0,4095,8193,True):
            with self.assertRaises(ValueError): a.rational_record(n)

    def test_output_diagnostics_keep_8192_bit_results(self):
        r=a.archive(a.inventory(),8)
        self.assertEqual(r['diagnostic_parts']['endpoint_receipts'],504*(512+3*2064))
        self.assertEqual(r['diagnostic_parts']['property_results'],252*(128+2064))
        # Numeric policy stays unchanged; the layout does not force a smaller
        # output bit width simply to fit the native archive envelope.
        policy=a.analyze()['current_policy_unchanged']
        self.assertEqual((policy['stored_bits'],policy['work_bits']),(4096,8192))

    def test_exact_core_phase_sum_and_replay_not_free(self):
        expected={1:(26519040,2161406048,4273366725),8:(15792660,1298519408,4273366725)}
        for w,values in expected.items():
            r=a.exact_core(a.inventory(),w)
            self.assertEqual(tuple(r['production'].values()),values)
            for k,value in r['production'].items():
                self.assertEqual(value,sum(x['counts'][k] for x in r['phases'].values()))
                self.assertEqual(r['online_with_separate_replay'][k],value+r['portable_replay'][k])
            self.assertEqual(r['online_role_index_upper'],2*r['role_index_access_upper_per_pass'])
            self.assertIsNone(r['complete_request_work_upper'])
            self.assertFalse(r['serialization_hash_decode_native_ops_in_core'])

    def test_native_schedule_not_omitted_from_cost_contract(self):
        for w,t in ((1,504),(8,84)):
            r=a.exact_core(a.inventory(),w)
            self.assertEqual(r['native_adjoint_calls'],129*t)
            self.assertEqual(r['native_forward_calls'],128*t)
            self.assertFalse(r['implementation_correspondence_proved'])

    def test_memory_is_payload_not_full_rss(self):
        for w,slots,b in ((1,340416,779542976),(8,1062032,2269113184)):
            r=a.memory(a.inventory(),w,a.archive(a.inventory(),w))
            self.assertEqual(r['rational_slots_upper'],slots)
            self.assertEqual(r['typed_payload_bytes_upper'],b)
            self.assertEqual(r['typed_payload_components']['exact_persistent_slots'],slots*2064)
            self.assertEqual(r['example_unreduced_addition_bits_upper'],16385)
            for k in ('full_peak_rss_upper','native_workspace_upper',
                      'live_checkpoint_capture_upper','arithmetic_runtime_scratch_upper'):
                self.assertIsNone(r[k])
            self.assertTrue(r['stop_limits_are_not_completion_bounds'])

    def test_no_quota_execution_or_full_closure_inferred(self):
        r=a.analyze()
        self.assertFalse(r['closure']['full_cost_upper_closed'])
        self.assertFalse(r['closure']['execution_admitted'])
        self.assertFalse(r['implementation_authorized'])
        self.assertFalse(r['production_files_changed'])
        self.assertFalse(r['scientific_claims_upgraded'])
        self.assertIsNone(r['new_quotas']); self.assertIsNone(r['absolute_ceiling'])
        self.assertEqual(r['invariant']['request_seconds'],300)
        self.assertEqual(r['invariant']['accept'],'strictly greater than 1/10000000')
        self.assertEqual(r['current_policy_unchanged']['visits'],8000000)
        self.assertEqual(r['current_policy_unchanged']['rows'],65536)
        for k in ('model_runs','gpu_runs','solver_calls','source_builds'):
            self.assertEqual(r[k],0)

    def test_wrong_shape_width_or_anchor_rejected(self):
        v=a.inventory(); v['classes']=9
        with self.assertRaises(ValueError): a.archive(v,8)
        for w in (0,2,9,True):
            with self.assertRaises(ValueError): a.exact_core(a.inventory(),w)
        with patch.dict(a.ANCHORS,{'scoped_source/hz_request_store.py':'0'*64}):
            with self.assertRaisesRegex(ValueError,'anchor'): a.analyze()

    def test_no_historical_artifacts_changed(self):
        r=a.analyze()
        for path,h in r['anchors'].items():
            self.assertEqual(hashlib.sha256((a.ROOT/path).read_bytes()).hexdigest(),h)
        old=json.loads((a.ROOT/'docs/hz_wire_capacity_20261014.json').read_bytes())
        self.assertEqual(old['mandatory_roots']['lower'],126632)
        self.assertFalse(old['execution_admitted'])

    def test_static_imports_and_saved_report(self):
        tree=ast.parse((a.ROOT/'scripts/assess_unique_root_protocol.py').read_bytes())
        imports=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(x.name for x in node.names)
            if isinstance(node,ast.ImportFrom): imports.add(node.module)
        self.assertEqual(imports,{'argparse','copy','hashlib','json','pathlib',
            'scripts.assess_model_scaled_resources','scripts.analyze_full_resource_ledger'})
        self.assertTrue({'torch','numpy','scipy','scoped_source'}.isdisjoint(sys.modules))
        self.assertEqual(a.analyze(),json.loads((a.ROOT/'docs/hz_unique_root_protocol_20261014.json').read_bytes()))


if __name__=='__main__': unittest.main()
