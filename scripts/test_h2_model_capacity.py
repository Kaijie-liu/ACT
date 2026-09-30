"""Small static accounting controls; never load a model or run a solver."""
import ast
import unittest
from scripts.audit_h2_model_capacity import derive, footprint, integer, limits, mlp


class CapacityTests(unittest.TestCase):
    def test_registered_counts_and_definite_source_block(self):
        r = derive()
        self.assertEqual(len(r['recorded_models']), 3)
        for model in r['recorded_models']:
            self.assertEqual(model['base64_bytes_lower_bound'], 74254592)
            self.assertTrue(all(model['source_exceeds_limits'].values()))
            self.assertFalse(model['checkpoint_opened'])
        d = r['dense_recipe_scenario']
        self.assertEqual((d['duties'], d['pair_variables']), (252, 4892))
        self.assertEqual(d['pair_affine_equality_entries_if_all_weights_nonzero'], 2036124)
        self.assertEqual(d['repeated_affine_entries_in_duty_bases_if_dense'], 513103248)

    def test_lower_bound_and_layer_counts(self):
        self.assertEqual(footprint(1), {'binary64_bytes': 8, 'base64_bytes_lower_bound': 12})
        self.assertEqual(mlp([2, 3, 2]), {'weights': 12, 'affine_outputs': 5,
                                       'relu_outputs': 3, 'parameters': 17, 'tensors': 4})
        for bad in (-1, True, 1.5):
            with self.assertRaises(ValueError): footprint(bad)
        with self.assertRaises(ValueError): mlp([1, 0, 2])

    def test_no_arbitrary_size_evaluation(self):
        self.assertEqual(integer(ast.parse('64*2**20', mode='eval').body), 67108864)
        for bad in ('int("64")', '2**100000', 'True', '-1', 'capacity', '1/2'):
            with self.assertRaises(ValueError): integer(ast.parse(bad, mode='eval').body)

    def test_missing_size_policy_rejected(self):
        with self.assertRaises(ValueError): limits('def read(p): return p.read_bytes()', 'pass')


if __name__ == '__main__': unittest.main()
