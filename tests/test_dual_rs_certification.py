import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from dual_rs_certification_evidence import from_counts, combine, recording_smooth, summarize


class CertificationControls(unittest.TestCase):
    def test_native_sampler_instrumentation_and_partial_failure(self):
        import importlib.util
        import torch
        source = Path('/data1/Kane/MOE/baselines/recent_moe_20260921/dual_rs/code/core.py')
        try:
            import statsmodels
        except ImportError:
            self.skipTest('native sampler differential runs in frozen Dual RS environment')
        spec = importlib.util.spec_from_file_location('native_smooth_control', source)
        native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(native)
        class NoisyControl(torch.nn.Module):
            def forward(self, x, t):
                return torch.randn(len(x), 3) + torch.tensor([2., 0., -1.])
        model = NoisyControl().eval()
        x = torch.zeros(3, 2, 2)
        torch.manual_seed(91)
        expected = native.Smooth(model, 3, 1., 0).certify(x, 10, 50, .0005, 8)
        expected_rng = torch.get_rng_state()
        logged = []
        cls = recording_smooth(native.Smooth, 10, 50, 3, 8, lambda i, c: logged.append((i, c)))
        torch.manual_seed(91)
        instance = cls(model, 3, 1., 0)
        actual = instance.certify(x, 10, 50, .0005, 8)
        self.assertEqual(actual, expected)
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
        self.assertEqual([sum(c) for _, c in logged], [10, 50])
        with self.assertRaises(ValueError):
            instance.certify(x, 10, 50, .0005, 8)
        partial = []
        def deadline(i, counts):
            partial.append(i)
            raise TimeoutError('controlled failure after first completed counts')
        broken = recording_smooth(native.Smooth, 10, 50, 3, 8, deadline)(model, 3, 1., 0)
        with self.assertRaises(TimeoutError):
            broken.certify(x, 10, 50, .0005, 8)
        self.assertEqual(partial, [1])

    def test_abstention_tie_and_selected_not_estimation_winner(self):
        v = from_counts([50, 50, 0], [4000, 6000, 0], 100, 10000, 3, 1., .0005)
        self.assertEqual(v['selected_class'], 0)
        self.assertEqual(v['prediction'], -1)
        self.assertEqual(v['radius_l2'], 0.)
        v = from_counts([1, 0], [0, 10000], 1, 10000, 2, 1., .0005)
        self.assertEqual(v['p_lower'], 0.)

    def test_native_clopper_pearson_differential(self):
        try:
            from statsmodels.stats.proportion import proportion_confint
        except ImportError:
            self.skipTest('native differential is run in the frozen Dual RS environment')
        from scipy.stats import norm
        for na in [0, 1, 4999, 5000, 6000, 9999, 10000]:
            v = from_counts([99, 1], [na, 10000-na], 100, 10000, 2, .5, .0005)
            lower = proportion_confint(na, 10000, alpha=.001, method='beta')[0]
            self.assertAlmostEqual(v['p_lower'], lower, places=14)
            self.assertAlmostEqual(v['radius_l2'], 0. if lower < .5 else .5*norm.ppf(lower), places=14)

    def test_count_pollution_and_missing_reject(self):
        for selection, estimation in [([1], [10000, 0]), ([True, 0], [10000, 0]),
                ([1, 0], [9999, 0]), ([1, 0], [-1, 10001]), ([1, 0], [10000., 0])]:
            with self.assertRaises(ValueError):
                from_counts(selection, estimation, 1, 10000, 2, 1., .0005)

    def test_stage_composition_and_wrong_class(self):
        s = {'prediction': 1, 'radius_l2': .3}
        c = {'prediction': 3, 'radius_l2': .5}
        out = combine(s, c, [.25, .5, 1.], 3)
        self.assertEqual(out['correct_radius_l2'], .3)
        self.assertEqual(out['selected_sigma'], .5)
        self.assertEqual(combine(s, c, [.25, .5, 1.], 2)['correct_radius_l2'], 0.)
        self.assertEqual(combine({'prediction': -1, 'radius_l2': 0}, None, [.25, .5, 1], 3)['radius_l2'], 0.)
        for left, right in [(s, None), ({'prediction': -1}, c)]:
            with self.assertRaises(ValueError):
                combine(left, right, [.25, .5, 1.], 3)

    def test_partial_or_late_evidence_never_promoted(self):
        r = {'status': 'COMPLETED', 'execution_including_preflight_seconds': 1.,
             'total_with_postflight_seconds': 1.1}
        inner = {'status': 'CERTIFICATION_PILOT_AUDITED'}
        self.assertTrue(summarize(r, inner)['accepted'])
        self.assertFalse(summarize(r, None)['accepted'])
        for status in ['ERROR', 'TIMEOUT', 'SOURCE_CHANGED']:
            self.assertFalse(summarize({**r, 'status': status}, inner)['accepted'])


if __name__ == '__main__':
    unittest.main()
