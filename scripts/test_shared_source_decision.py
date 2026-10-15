"""Static arithmetic and saved-binding checks; no source/candidate controls."""
import ast
import copy
import unittest
from scripts import assess_shared_source_decision as a


class Controls(unittest.TestCase):
    def test_semantic_floor_excludes_snapshot_and_biases(self):
        r=a.analyze(); x=r['semantic_floor_min_endpoint_roster']
        self.assertEqual(x['all_affine_weight_positions'],6958080)
        self.assertEqual(x['source_coefficient_floor'],13916160)
        self.assertEqual(x['consumer_access_free_floor'],9289728)
        self.assertEqual(x['joint_floor'],23205888)
        self.assertEqual(r['semantic_floor_max_endpoint_roster']['joint_floor'],32495616)
        self.assertEqual(r['source_floor_assessment'],'CURRENT_CONTRACT_REFUSAL')
        self.assertFalse(r['full_capacity_admitted'])
        self.assertFalse(r['runtime_changed']); self.assertIsNone(r['new_quota'])

    def test_representation_bound_is_not_runtime_or_overflow_claim(self):
        r=a.analyze(); f=r['forest_envelope']
        self.assertEqual(f['explicit_local_entries_upper'],48048)
        self.assertEqual(f['payload_upper'],13717764)
        self.assertEqual(f['pages_upper'],210)
        self.assertEqual(f['source_plus_forest_upper'],69433284)
        self.assertIsNone(f['total_archive_upper']); self.assertIsNone(f['full_rss_upper'])
        self.assertEqual(r['membership']['assessment'],'CURRENT_CONTRACT_REFUSAL')
        self.assertTrue(r['membership']['changing_expanded_to_unique_count_is_new_resource_policy'])

    def test_both_candidate_roles_and_general_profile_remain(self):
        r=a.analyze(); n=r['candidate_stream']['binary64']; q=r['candidate_stream']['general_rational']
        self.assertEqual(r['candidate_layout']['stored_rational'],1040)
        self.assertEqual(n['stream_upper'],22868160)
        self.assertEqual(q['stream_upper'],2831559360)
        self.assertEqual(q['largest_tile_upper'],44945344)
        self.assertEqual(q['one_endpoint_upper'],5618280)
        for s in (n,q):
            self.assertIsNone(s['actual_bytes'])
            self.assertFalse(s['upper_excess_proves_actual_overflow'])
            self.assertEqual(s['stream_assessment'],'UPPER_EXCEEDS_CAP_ACTUAL_UNKNOWN')

    def test_bounds_classification_and_invalid_domains(self):
        self.assertEqual(a.classify(9,lower=10),'CURRENT_CONTRACT_REFUSAL')
        self.assertEqual(a.classify(9,upper=10),'UPPER_EXCEEDS_CAP_ACTUAL_UNKNOWN')
        self.assertEqual(a.classify(9,upper=9),'UPPER_FITS_THIS_COMPONENT_ONLY')
        self.assertEqual(a.classify(9),'BOUND_OPEN')
        for kwargs in ({'lower':True},{'upper':-1},{'lower':11,'upper':10}):
            with self.assertRaises(ValueError): a.classify(9,**kwargs)
        for node in ('True','f()','2**100','a','2/3'):
            with self.assertRaises(ValueError): a.integer(ast.parse(node,mode='eval').body)

    def test_mutated_binding_and_inventory_refused(self):
        parent=copy.deepcopy(a.load(a.PARENT))
        parent['anchors']['scoped_source/check_shared_source.py']='0'*64
        with self.assertRaisesRegex(ValueError,'implementation drift'): a.validate_parent(parent)
        v=copy.deepcopy(a.load(a.SHAPES[0])['inventory']); v['parameter_values']+=1
        with self.assertRaisesRegex(ValueError,'exclude biases'): a.semantic_floor(v,252)
        for widths,profile in (([9],0),([True],0),([],0),([1],True),([1],2)):
            with self.assertRaises(ValueError): a.frame_sizes(2700,widths,profile,1040,128,72)


if __name__=='__main__': unittest.main()
