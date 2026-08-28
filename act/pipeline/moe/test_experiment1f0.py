# ===- act/pipeline/moe/test_experiment1f0.py - F0 Runner Tests ------====#

import unittest

import torch

from act.back_end.core import Bounds
from act.back_end.solver.solver_hz import HZSupportBoundsResult
from act.pipeline.moe.experiment1f0 import _support_status


class Experiment1F0Tests(unittest.TestCase):
    def test_support_metadata_uses_public_result_fields(self):
        result = HZSupportBoundsResult(
            rows=(0,),
            bounds=Bounds(torch.tensor([[0.0]]), torch.tensor([[1.0]])),
            lower_status=("lp_optimal",),
            upper_status=("fast_fallback",),
            solver_gap=(0.0,),
            elapsed=0.25,
            solves=2,
            exact=False,
        )
        record = _support_status(result)
        self.assertFalse(record["complete_exact"])
        self.assertEqual(record["gaps"], [0.0])
        self.assertEqual(record["lower_status"], ["lp_optimal"])
        self.assertEqual(record["upper_status"], ["fast_fallback"])


if __name__ == "__main__":
    unittest.main()
