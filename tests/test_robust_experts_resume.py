import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from robust_experts_resume_control import rng_state, restore_rng
from audit_dual_rs_training_control import canonical


class RNGControls(unittest.TestCase):
    def test_exact_rng_restore(self):
        import random
        import numpy as np
        import torch
        state = rng_state()
        expected = (random.random(), np.random.randn(4).tolist(), torch.rand(5))
        restore_rng(state)
        observed = (random.random(), np.random.randn(4).tolist(), torch.rand(5))
        self.assertEqual(canonical(expected), canonical(observed))

    def test_missing_state_rejected(self):
        with self.assertRaises(KeyError):
            restore_rng({})

    def test_optimizer_change_detected(self):
        import torch
        self.assertNotEqual(canonical({'momentum_buffer': torch.tensor([1.])}),
                            canonical({'momentum_buffer': torch.tensor([2.])}))


if __name__ == '__main__':
    unittest.main()
