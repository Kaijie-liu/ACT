import sys
from pathlib import Path
import unittest
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from metamoe_paired_model import InvariantObligations


class PairedSemantics(unittest.TestCase):
    def make(self, scores, logits, label=0):
        router = nn.Linear(1, 2, dtype=torch.float64)
        expert = nn.Linear(1, 2, dtype=torch.float64)
        with torch.no_grad():
            for m in [router, expert]:
                m.weight.zero_()
            router.bias.copy_(torch.tensor(scores))
            expert.bias.copy_(torch.tensor(logits))
        return InvariantObligations(router, expert, [2, 2], 0, label,
                                    1 if scores[0] > 0 else -1, 0.)

    def test_zero_blocks_are_properties(self):
        out = self.make([2., 1.], [-1., -2.])(torch.zeros(1, 1, dtype=torch.float64))
        self.assertTrue((out[:, 1:3] > 0).all())
        self.assertFalse((out[:, 3:] > 0).all())

    def test_negative_selected_score_can_be_defined(self):
        out = self.make([-1., -2.], [2., 1.])(torch.zeros(1, 1, dtype=torch.float64))
        self.assertTrue((out[:, 1:] > 0).all())

    def test_tie_or_zero_not_invariant_positive(self):
        for scores in [[1., 1.], [0., -1.]]:
            out = self.make(scores, [2., 1.])(torch.zeros(1, 1, dtype=torch.float64))
            self.assertFalse((out[:, 1:] > 0).all())


if __name__ == '__main__':
    unittest.main()
