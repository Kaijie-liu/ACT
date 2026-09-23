import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from review_bn_conversion_bindings import classify,review


class BindingReviewControls(unittest.TestCase):
    def test_no_version_is_automatically_source_proof(self):
        self.assertEqual(classify('a','a','b'),'PRE_REPAIR_SOURCE_EXPOSURE')
        self.assertEqual(classify('b','a','b'),'REPAIRED_SOURCE_BOUND_NOT_DOMAIN_PROOF')
        self.assertEqual(classify('c','a','b'),'UNREVIEWED_CONVERTER_VERSION')

    def test_read_only_inventory_keeps_old_and_new(self):
        v=review()
        self.assertGreater(v['counts']['PRE_REPAIR_SOURCE_EXPOSURE'],0)
        self.assertGreater(v['counts']['REPAIRED_SOURCE_BOUND_NOT_DOMAIN_PROOF'],0)
        self.assertEqual(v['new_solver_calls'],0)
        self.assertEqual(v['new_model_forwards'],0)
        self.assertFalse(v['historical_files_modified'])
        for r in v['rows']:
            for w in r['worker_records']:
                self.assertEqual(w['exposure_applies_to_this_backend'],w['arm']=='act')


if __name__=='__main__':unittest.main()
