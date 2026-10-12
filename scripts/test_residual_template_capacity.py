"""Arithmetic/identity controls only; no numerical verification requests."""
import unittest
from unittest.mock import patch

from scripts import analyze_residual_template_capacity as a


class Capacity(unittest.TestCase):
    def test_dense_countercase(self):
        c=a.template_countercase(3072,256,8,9)
        self.assertEqual(c['templates'],72)
        self.assertEqual(c['coefficient_column_products'],56623104)
        self.assertEqual(c['exact_arithmetic'],113246208)
        self.assertEqual(c['coefficient_visits'],12582912)
        self.assertEqual(c['partial_work'],125829120)

    def test_width_does_not_remove_per_column_arithmetic(self):
        cases=[a.template_countercase(3072,256,8,9,w) for w in (8,4,2,1)]
        self.assertEqual(len({c['exact_arithmetic'] for c in cases}),1)
        self.assertEqual(sorted(c['partial_work'] for c in cases),[c['partial_work'] for c in cases])
        with self.assertRaises(ValueError): a.template_countercase(3072,256,8,9,9)

    def test_source_conditions_and_sparsity_are_not_observations(self):
        r=a.analyze(); s=r['optimistic_source_checker']
        self.assertEqual(s['dense_zero_center'],13369344)
        self.assertEqual(s['dense_all_nonzero_center'],20054016)
        self.assertIsNone(s['actual_Z']); self.assertIsNone(s['actual_K'])
        self.assertFalse(r['template_countercase']['actual_candidate_sparsity_observed'])
        self.assertEqual(r['necessary_sparse_case']['max_active_row_template_associations'],214)
        self.assertFalse(r['implementation_admitted']); self.assertFalse(r['real_execution_admitted'])

    def test_generic_small_arithmetic(self):
        for n in (1,3):
            for h in (1,4):
                for e in (2,3):
                    for p in (1,3):
                        explicit=sum(1 for _ in range(e) for _ in range(p) for _ in range(h) for _ in range(n))
                        self.assertEqual(a.template_countercase(n,h,e,p)['coefficient_column_products'],explicit)
        self.assertEqual(a.source_floor(10,0,0),10)
        self.assertEqual(a.source_floor(10,3,2),15)
        with self.assertRaises(ValueError): a.source_floor(10,3,4)

    def test_symbolic_alternative_is_not_free_or_accepted(self):
        s=a.analyze()['symbolic_source_alternative']
        self.assertEqual(s['one_source_value_scan_if_used'],6964440)
        self.assertEqual(s['remainder_before_any_other_cost'],1035560)
        self.assertFalse(s['complete_ledger_available']); self.assertFalse(s['preserves_existing_enclosure'])

    def test_parent_identity_mutation(self):
        with patch.object(a,'PARENT_SCRIPT_SHA','0'*64):
            with self.assertRaisesRegex(ValueError,'parent static analysis identity'): a.analyze()


if __name__ == '__main__': unittest.main()
