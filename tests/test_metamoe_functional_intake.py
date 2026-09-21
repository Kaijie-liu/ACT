import sys
from pathlib import Path
import unittest
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from metamoe_functional_intake import moduleize_relu, center_witness


class Functional(nn.Module):
    def __init__(self, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.affine = nn.Linear(2, 2, dtype=torch.float64)

    def forward(self, x):
        return F.relu(self.affine(x), inplace=self.inplace)


class TestIntake(unittest.TestCase):
    def test_exact_forward_gradient_and_source_unchanged(self):
        m = Functional().eval()
        before = {k: v.clone() for k, v in m.state_dict().items()}
        converted, records = moduleize_relu(m)
        self.assertEqual(len(records), 1)
        x = torch.tensor([[-2., 3.], [0., 1.]], dtype=torch.float64, requires_grad=True)
        a, b = m(x), converted(x)
        self.assertTrue(torch.equal(a, b))
        self.assertTrue(torch.equal(torch.autograd.grad(a.sum(), x)[0], torch.autograd.grad(b.sum(), x)[0]))
        self.assertEqual(set(before), set(m.state_dict()))
        self.assertTrue(all(torch.equal(v, m.state_dict()[k]) for k, v in before.items()))

    def test_inplace_and_train_reject(self):
        for m in [Functional().train(), Functional(True).eval()]:
            with self.assertRaises(ValueError):
                moduleize_relu(m)

    def test_common_center_not_negative_bound(self):
        class Full(nn.Module):
            def forward(self, x):
                return torch.cat([x, -x], dim=1), x
        x = torch.ones(1, 1, dtype=torch.float64)
        self.assertIsNone(center_witness(Full(), x, x-.1, x+.1, 0, 1e-7))
        self.assertEqual(center_witness(Full(), x, x-.1, x+.1, 1, 1e-7)['status'], 'UNSAFE_REPLAYED')
        with self.assertRaises(ValueError):
            center_witness(Full(), x+1, x-.1, x+.1, 1, 1e-7)


if __name__ == '__main__':
    unittest.main()
