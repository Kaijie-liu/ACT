"""Specification controls only; no external numerical imports or execution."""
import importlib.util
import math
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('guard_semantics',
    Path(__file__).resolve().parents[1]/'scripts/genbab_guard_semantics.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)


class GuardSemantics(unittest.TestCase):
    def test_max_is_conjunction_of_closed_unsafe_atoms(self):
        for values in ((0, 0, 0), (-1, 0, 0), (-1, 1, 0), (1, -1, -1), (-1, -1, -1)):
            f, a, b = values
            maximum = f+max(0, a-f)
            maximum += max(0, b-maximum)
            self.assertEqual(maximum <= 0, all(v <= 0 for v in values))
            self.assertEqual(maximum > 0, not all(v <= 0 for v in values))

    def test_fixed_source_ties_and_full_box_minimum(self):
        for x in (-.5, .5):
            f, a, b = module.values(x)
            self.assertTrue(a <= 0 and b <= 0)
            self.assertGreater(f, .25)
            self.assertEqual(max(a,b), 0)
        self.assertAlmostEqual(module.values(1)[0], .5-math.tanh(1))
        self.assertAlmostEqual(module.values(1)[0], -.2615941559557649)
        self.assertGreater(max(module.values(1)), 0)

    def test_engines_and_budgets_not_conflated(self):
        a = module.configuration('input_split_and')
        b = module.configuration('nonlinear_scalar_max')
        self.assertTrue(a['bab.branching.input_split.enable'])
        self.assertEqual(a['attack.pgd_order'], 'before')
        self.assertFalse(b['bab.branching.input_split.enable'])
        self.assertEqual(b, module.old.CONFIG)
        self.assertEqual(len(module.MODES), 2)
        self.assertLess(2*module.PER_MODE_SECONDS, 300)


if __name__ == '__main__':
    unittest.main()
