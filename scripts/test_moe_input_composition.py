"""Exact no-solver controls, usable with python -I -S."""
from fractions import Fraction as Q
import importlib.util
import json
from pathlib import Path
import sys
import unittest

spec = importlib.util.spec_from_file_location('composition', Path(__file__).with_name('audit_moe_input_composition.py'))
a = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a)


class Composition(unittest.TestCase):
    def test_no_numerical_or_model_import(self):
        self.assertFalse(set(('act', 'torch', 'scipy', 'numpy', 'highspy')) & set(sys.modules))

    def test_equal_and_degenerate(self):
        for box in ((0, 1), (Q(1, 2), Q(1, 2))):
            self.assertEqual(a.endpoint_gaps(box, box), ((0, 0), (0, 0)))

    def test_containment_is_not_strict_subset(self):
        self.assertEqual(a.endpoint_gaps((0, 1), (Q(1, 4), Q(5, 4))),
                         ((Q(1, 4), 0), (0, Q(1, 4))))

    def test_two_failed_edges_do_not_imply_failed_composition(self):
        requested, box, hz = (0, 1), (Q(1, 4), 2), (0, Q(3, 2))
        self.assertTrue(any(a.endpoint_gaps(requested, box)[0]))
        self.assertTrue(any(a.endpoint_gaps(box, hz)[0]))
        self.assertFalse(any(a.endpoint_gaps(requested, hz)[0]))

    def test_reversed_nonfinite_corrupt_materialization(self):
        with self.assertRaises(ValueError): a.endpoint_gaps((1, 0), (0, 1))
        for lo in (float('nan'), .4):
            with self.assertRaises(ValueError): a.coordinate(.5, lo, .5+2/255, 2/255)

    def test_epsilon_semantics_differ(self):
        r = a.coordinate(.5, .5-2/255, .5+2/255, 2/255)
        self.assertNotEqual(r['rational_epsilon'], r['binary64_epsilon'])

    def test_zero_vs_dropped_positive_radius(self):
        self.assertEqual(a.coordinate(.5, .5, .5, 0.)['binary64_epsilon'], ((0, 0), (0, 0)))
        self.assertTrue(any(a.coordinate(0., 0., 1e-13, 1e-13)['binary64_epsilon'][0]))

    def test_wrong_hash_and_outside_reference_rejected(self):
        for path in ('/dev/null', 'scripts/audit_moe_input_composition.py'):
            with self.assertRaises(ValueError): a.pinned_bytes(path, '0'*64)

    def test_summary_schema_and_roundtrip(self):
        b = {'rank': 0, 'dataset_index': 1, 'coordinates': 1, 'representative_saved_request': {}}
        r = a.box_record(b, {'center': [.5], 'lower': [.5-2/255], 'upper': [.5+2/255]}, 2/255)
        self.assertEqual(json.loads(json.dumps(r)), r)
        with self.assertRaises(ValueError): a.box_record(b, {'center': []}, 2/255)


if __name__ == '__main__':
    unittest.main()
