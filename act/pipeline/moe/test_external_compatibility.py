import math
from pathlib import Path
import unittest
from unittest.mock import patch
import torch
from act.pipeline.moe.external_compatibility import toy_model, run, CASES


class CompatibilityTests(unittest.TestCase):
    def test_variable_weights_not_frozen_and_both_experts_present(self):
        x=torch.tensor([[.1,-.1]],dtype=torch.float64)
        # Experts happen to agree at this point; second point differentiates weights.
        self.assertAlmostEqual(float(toy_model('static_pair')(x)),1.1)
        x=torch.tensor([[.1,0.]],dtype=torch.float64)
        expected=(math.exp(.1)*1.1+1)/(math.exp(.1)+1)
        self.assertAlmostEqual(float(toy_model('static_pair')(x)),expected)
        self.assertNotAlmostEqual(expected,1.05,places=5)

    def test_dynamic_routes_and_fixed_protocol(self):
        x=torch.tensor([[.1,0.],[-.1,0.]],dtype=torch.float64)
        out=toy_model('dynamic_top2')(x)
        self.assertGreater(float(out[0]),float(out[1]))
        self.assertEqual(CASES,('dynamic_top2','static_pair','input_polytope'))
        with patch('act.pipeline.moe.external_compatibility.git',return_value='main'):
            with self.assertRaises(RuntimeError):run(Path('/data1/Kane/MOE/ACT/data/moe/results/never-created'))


if __name__=='__main__':unittest.main()
