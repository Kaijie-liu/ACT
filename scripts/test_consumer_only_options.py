"""Arithmetic and refusal-label checks, not new source/consumer controls."""
import unittest
from unittest.mock import patch

from scripts import assess_consumer_only_options as a


class StaticOptions(unittest.TestCase):
    def test_access_free_floor_retains_logical_visits(self):
        self.assertEqual(a.body_floor(3072, 252), 4644864)
        self.assertEqual(a.body_floor(3072, 504), 9289728)
        self.assertLess(a.body_floor(3072, 252), 8000000)
        self.assertGreater(a.body_floor(3072, 504), 8000000)
        self.assertEqual(2 * 3072 * (128 + 8 * 256) + a.body_floor(3072, 252), 18014208)

    def test_lossless_zero_role_extent(self):
        native = a.zero_format(2700, 8)
        general = a.zero_format(2700, 1040)
        self.assertEqual(native['frame_upper'], 190144)
        self.assertEqual(native['stream_upper'], 11981760)
        self.assertEqual(general['frame_upper'], 22481344)
        self.assertEqual(general['stream_upper'], 1416327360)
        self.assertTrue(native['stream_upper_fits'])
        self.assertFalse(general['frame_upper_fits'])
        for item in (native, general):
            self.assertIsNone(item['actual_bytes'])
            self.assertFalse(item['implemented'])

    def test_no_scope_expansion_or_admission(self):
        report = a.analyze()
        self.assertFalse(report['free_access']['algorithm_universal_lower_bound'])
        self.assertFalse(report['free_access']['minimum_endpoint_consumer_alone_refuses'])
        self.assertFalse(report['production_changed'])
        self.assertFalse(report['complete_cost_closed'])
        self.assertFalse(report['execution_admitted'])
        self.assertIsNone(report['new_quota'])
        self.assertIsNone(report['full_work_upper'])
        self.assertIsNone(report['full_rss_upper'])
        self.assertEqual(report['unchanged']['seconds'], 300)
        self.assertEqual(report['unchanged']['work'], 8000000)
        self.assertEqual(sum(report[k] for k in ('source_builds', 'model_runs', 'solver_calls', 'gpu_runs')), 0)

    def test_source_drift_and_invalid_dimensions(self):
        with patch.dict(a.ANCHORS, {'scoped_source/scaled_references.py': '0' * 64}):
            with self.assertRaisesRegex(ValueError, 'source drift'):
                a.analyze()
        for invalid in (0, -1, True, 1.0):
            with self.assertRaises(ValueError):
                a.body_floor(invalid, 252)
            with self.assertRaises(ValueError):
                a.zero_format(invalid, 8)
        with self.assertRaises(ValueError):
            a.zero_format(2700, 4)


if __name__ == '__main__':
    unittest.main()
