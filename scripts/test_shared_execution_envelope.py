"""Arithmetic/identity tests only. No production module or worker is imported."""
import copy
import unittest
from unittest.mock import patch

from scripts import assess_shared_execution_envelope as a


class Controls(unittest.TestCase):
    def test_source_phase_total(self):
        r=a.analyze(); s=r['source_counted_work']
        self.assertEqual(s['producer_upper'],62330600)
        self.assertEqual(s['encode_upper'],668832)
        self.assertEqual(s['one_checker_upper'],76712652)
        self.assertEqual(s['build_and_two_checks_upper'],216424736)
        self.assertTrue(s['source_seal_counted_in_consumer_finish'])

    def test_consumer_complete_loops_and_roles(self):
        r=a.analyze(); d=r['consumers']['dense_reference']; s=r['consumers']['pair_scoped']
        self.assertEqual((d['production_upper'],d['replay_upper']),(44612645184,44604461736))
        self.assertEqual((s['production_upper'],s['replay_upper']),(82450188984,82442005536))
        self.assertEqual(d['postcapture_counted_work_upper'],89433531656)
        self.assertEqual(s['postcapture_counted_work_upper'],165108619256)
        self.assertEqual((d['check_blocks_upper'],s['check_blocks_upper']),(84,252))
        self.assertEqual(d['fixed_proposer_upper'],s['fixed_proposer_upper'])

    def test_seal_not_double_counted(self):
        v=a.analyze()['inventory']
        for scoped in (False,True):
            lo=a.consumer_work(v,scoped,1); hi=a.consumer_work(v,scoped,4096)
            self.assertEqual(hi['production_and_replay_upper']-lo['production_and_replay_upper'],2*4095)
            self.assertEqual(hi['phases']['finish'],504+252+4096+9)

    def test_dense_original_and_copy_coexist(self):
        r=a.analyze(); m=r['dense_owned_arena']
        self.assertEqual(m['slot_bytes'],2065)
        self.assertEqual(m['one_descriptor_upper'],8272)
        self.assertEqual(m['complete_width8_vector_lower'],322734720)
        self.assertEqual(m['owned_payload_upper'],1704299744)
        self.assertEqual(m['lower_assessment'],'CURRENT_CONTRACT_REFUSAL')
        self.assertIsNone(m['full_rss_upper']); self.assertIsNone(m['actual_peak'])

    def test_metadata_both_report_copies_without_overflow_claim(self):
        r=a.analyze(); m=r['metadata_under_current_counter_cap']
        self.assertEqual((m['integer_digits'],m['fraction_text_upper']),(2467,4936))
        self.assertEqual(m['result_upper'],1257460)
        self.assertEqual(m['candidate_report_full_result_copies'],2)
        self.assertEqual(m['progress_rows'],368)
        self.assertEqual(m['progress_bytes_upper'],58880)
        full=r['metadata_under_conditional_loop_upper']
        self.assertEqual(full['counter_digits'],12)
        self.assertGreater(full['phase_list_upper'],m['phase_list_upper'])
        self.assertFalse(m['upper_excess_proves_actual_overflow'])
        self.assertIsNone(m['total_report_upper'])

    def test_no_execution_or_authority_upgrade(self):
        r=a.analyze()
        for field in ('source_builds','candidate_builds','model_runs','gpu_runs','solver_calls'):
            self.assertEqual(r[field],0)
        self.assertFalse(r['runtime_changed']); self.assertFalse(r['full_capacity_admitted'])
        self.assertIsNone(r['new_quota']); self.assertEqual(r['unchanged']['seconds'],300)
        self.assertEqual(r['unchanged']['work'],8000000)
        self.assertTrue(all(v is None for k,v in r['open_contract'].items() if k!='reasons'))

    def test_invalid_inputs_and_changed_binding(self):
        r=a.analyze(); v=r['inventory']
        for scoped,chunks in ((1,1),(False,0),(False,4097)):
            with self.assertRaises(ValueError): a.consumer_work(v,scoped,chunks)
        for args in ((0,252,84,7),(8193,252,84,7),(8192,253,84,7)):
            with self.assertRaises(ValueError): a.metadata(*args)
        mutated=copy.deepcopy(v); mutated['local_entries_upper']-=1
        with self.assertRaisesRegex(ValueError,'local envelope'): a.source_work(mutated)
        load=a.prior.load
        bad=copy.deepcopy(load(a.PARENT)); bad['inventory']['inputs']+=1
        with patch.object(a.prior,'load',side_effect=lambda name:bad if name==a.PARENT else load(name)):
            with self.assertRaisesRegex(ValueError,'parent decision'): a.analyze()


if __name__=='__main__': unittest.main()
