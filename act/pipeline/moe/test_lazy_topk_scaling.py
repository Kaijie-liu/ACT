import json
from pathlib import Path
import unittest

import numpy as np

from act.pipeline.moe.lazy_topk_scaling import _condition_order, _router


class LazyTopKScalingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        path = Path("/data1/Kane/MOE/ACT/act/pipeline/moe/configs/lazy_topk_scaling_r1.json")
        cls.config = json.loads(path.read_text(encoding="utf-8"))

    def test_pair_order_alternates_without_dropping_a_condition(self) -> None:
        self.assertEqual(_condition_order(0, 0), (False, True))
        self.assertEqual(_condition_order(0, 1), (True, False))
        self.assertEqual(set(_condition_order(3, 2)), {False, True})

    def test_families_are_deterministic_and_distinct(self) -> None:
        tied, tied_weight, tied_bias = _router(self.config, 8, "all_tied_worst_case", 1)
        tied2, tied_weight2, tied_bias2 = _router(self.config, 8, "all_tied_worst_case", 2)
        self.assertTrue(np.array_equal(tied_weight, tied_weight2))
        self.assertTrue(np.array_equal(tied_bias, tied_bias2))
        self.assertEqual(tied.n_out, 8)
        stable, stable_weight, stable_bias = _router(self.config, 8, "strictly_stable", 3)
        self.assertEqual(stable.n_out, 8)
        self.assertTrue(np.all(stable_bias[:-1] > stable_bias[1:]))
        random1, weight1, bias1 = _router(self.config, 16, "random_affine_box", 4)
        random2, weight2, bias2 = _router(self.config, 16, "random_affine_box", 5)
        self.assertTrue(np.array_equal(weight1, weight2))
        self.assertTrue(np.array_equal(bias1, bias2))
        self.assertEqual(random1.n_out, random2.n_out)
        self.assertFalse(np.array_equal(stable_weight, tied_weight))


if __name__ == "__main__":
    unittest.main()
