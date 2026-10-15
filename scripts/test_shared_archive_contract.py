"""Static schema/length/identity controls; no production imports or execution."""
import copy
import json
import unittest
from unittest.mock import patch

from scripts import assess_shared_archive_contract as a


class Controls(unittest.TestCase):
    def test_length_algebra_against_small_json(self):
        for item in (None,True,False,1,-15,'x',{'a':[1,2],'unicode':'界'}):
            for n in (0,1,3):
                actual=len(json.dumps([item]*n,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode())
                self.assertEqual(a.length(a.array(n,item)),actual)
        with self.assertRaises(ValueError): a.Size(-1)
        with self.assertRaises(ValueError): a.array(True,0)

    def test_canonical_source_is_conditional(self):
        r=a.analyze()['source_layout']
        self.assertEqual((r['tensors'],r['chunks'],r['files']),(53,95,96))
        self.assertEqual(r['tensor_bytes'],55715520)
        self.assertEqual((r['manifest_fixed_upper'],r['manifest_domain_upper']),(34280,50656))
        self.assertFalse(r['current_supervisor_full_source_intake'])
        self.assertIsNone(r['generic_parser_layout_upper'])
        v=a.prior.load(a.PARENT)['inventory']
        for options in (dict(full_chunks=False),dict(fixed_trust=False)):
            with self.assertRaisesRegex(ValueError,'canonical'): a.source_layout(v,**options)

    def test_forest_payload_and_source_are_not_duplicated(self):
        r=a.analyze(); f=r['forest_layout']
        self.assertEqual((f['header_upper'],f['binary_upper'],f['payload_upper']),(1652770,5329156,6981926))
        self.assertEqual((f['pages'],f['index_upper'],f['files']),(107,11835,108))
        self.assertEqual(r['source_layout']['stored_source_upper']+f['stored_upper'],62759937)
        self.assertTrue(f['memberships_remain_expanded'])

    def test_whole_success_metadata_counts_all_results(self):
        m=a.analyze()['metadata']['conditional_complete_loops']
        self.assertEqual(m['report_result_copies'],dict(candidate=2,standalone_result=1,parent_receipt=1))
        self.assertEqual(m['members_upper']['candidate_report'],2666907)
        self.assertEqual(m['members_upper']['parent_receipt'],1294980)
        self.assertEqual(m['total_upper'],5438505)
        self.assertEqual((m['artifact_files'],m['whole_request_files'],m['bundle_members']),(319,324,316))
        self.assertEqual(m['members_upper']['progress'],64400)
        self.assertIsNone(m['failure_generation_upper'])
        self.assertIsNone(m['actual_generated_bytes'])

    def test_counter_width_not_quota(self):
        r=a.analyze(); c=r['counter_representation']
        self.assertLess(c['copied_bytes_upper'],2**64)
        self.assertLess(c['allocated_bytes_upper'],2**64)
        self.assertEqual(c['one_copy_payload_upper'],44945344)
        self.assertTrue(c['not_a_new_quota'])
        old=r['metadata']['current_counters']; full=r['metadata']['conditional_complete_loops']
        self.assertEqual(old['members_upper']['progress'],58880)
        self.assertLess(old['members_upper']['candidate_report'],full['members_upper']['candidate_report'])
        self.assertEqual(r['unchanged']['work'],8000000)

    def test_archive_profiles_and_no_success_inference(self):
        r=a.analyze(); arcs=r['conditional_success_archive']
        self.assertEqual(r['code_inventory']['count'],26)
        self.assertEqual(r['code_inventory']['bytes'],244614)
        self.assertEqual(len(r['code_inventory']['bindings']),44)
        self.assertEqual(arcs['binary64']['request_upper'],91311216)
        self.assertEqual(arcs['general_rational']['request_upper'],2900002416)
        for row in arcs.values():
            self.assertFalse(row['full_capacity_admitted']); self.assertIsNone(row['actual_bytes'])

    def test_actual_lifetime_not_payload_as_rss(self):
        m=a.analyze()['lifetime']
        self.assertEqual((m['packed_all_payload'],m['packed_first_payload']),(55690944,53494784))
        self.assertFalse(m['builder_and_checker_graphs_coexist'])
        self.assertFalse(m['production']['packet_released_before_consumer'])
        self.assertTrue(m['replay']['packet_released_before_consumer'])
        self.assertEqual(m['replay']['frame_copies_at_seal'],3)
        self.assertEqual(m['scratch']['row_fraction_values_upper'],6146)
        self.assertEqual(m['scratch']['plan_fraction_entries_upper'],83592)
        self.assertTrue(m['scratch']['counts_are_not_allocated_container_capacities'])
        self.assertEqual(m['scratch']['visible_memberships_upper'],2627936)
        self.assertTrue(m['supervisor']['sampled_rss_not_os_limit'])
        self.assertFalse(m['supervisor']['parent_receiving_sampled'])
        self.assertIsNone(m['full_rss_upper']); self.assertIsNone(m['actual_peak'])

    def test_parent_binding_and_authority(self):
        load=a.prior.load; changed=copy.deepcopy(load(a.PARENT)); changed['inventory']['inputs']+=1
        with patch.object(a.prior,'load',side_effect=lambda n:changed if n==a.PARENT else load(n)):
            with self.assertRaisesRegex(ValueError,'execution envelope'): a.analyze()
        r=a.analyze()
        self.assertFalse(r['runtime_changed']); self.assertFalse(r['full_capacity_admitted'])
        self.assertIsNone(r['new_quota']); self.assertEqual(r['unchanged']['seconds'],300)
        for key in ('source_builds','candidate_builds','model_runs','gpu_runs','solver_calls'):
            self.assertEqual(r[key],0)


if __name__=='__main__': unittest.main()
