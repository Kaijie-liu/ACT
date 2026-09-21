import sys
from pathlib import Path
import unittest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from dual_rs_log_domain_consistency import consistency_loss


def reference(logits):
    p = sum(F.softmax(x, 1) for x in logits) / len(logits)
    kl = sum(F.kl_div(F.log_softmax(x, 1), p, reduction="none").sum(1) for x in logits) / len(logits)
    ent = -(p * p.clamp(min=1e-20).log()).sum(1)
    return (40 * kl + .5 * ent).clamp(min=1e-10)


class LogDomainControls(unittest.TestCase):
    def test_normal_domain_loss_and_target_gradients_match(self):
        torch.manual_seed(124)
        for dtype, tol in [(torch.float64, 1e-11), (torch.float32, 5e-5)]:
            xs = [torch.randn(9, 3, dtype=dtype, requires_grad=True) for _ in range(2)]
            a, b = reference(xs), consistency_loss(xs, 40)[0]
            torch.testing.assert_close(a, b, atol=tol, rtol=tol)
            ga = torch.autograd.grad(a.sum(), xs)
            gb = torch.autograd.grad(b.sum(), xs)
            for x, y in zip(ga, gb):
                torch.testing.assert_close(x, y, atol=tol, rtol=tol)

    def test_zero_probability_negative_control_and_finite_compatibility(self):
        xs = [torch.tensor([[-150., 150., 0.]], requires_grad=True),
              torch.tensor([[-140., 140., 0.]], requires_grad=True)]
        a = reference(xs)
        ga = torch.autograd.grad(a.sum(), xs)
        self.assertTrue(any(not torch.isfinite(g).all() for g in ga))
        b = consistency_loss(xs, 40)[0]
        gb = torch.autograd.grad(b.sum(), xs)
        self.assertTrue(torch.isfinite(b).all())
        self.assertTrue(all(torch.isfinite(g).all() for g in gb))

    def test_double_gradient_check_and_identical_logits(self):
        torch.manual_seed(125)
        xs = tuple(torch.randn(2, 3, dtype=torch.float64, requires_grad=True) for _ in range(2))
        self.assertTrue(torch.autograd.gradcheck(lambda a, b: consistency_loss([a, b], 40)[0], xs))
        x = torch.zeros(3, 3, dtype=torch.float64, requires_grad=True)
        value, kl, _ = consistency_loss([x, x], 40)
        self.assertAlmostEqual(kl, 0.)
        self.assertTrue(torch.isfinite(torch.autograd.grad(value.sum(), x)[0]).all())

    def test_nonfinite_or_unsupported_not_silently_repaired(self):
        with self.assertRaises(ValueError):
            consistency_loss([torch.tensor([[float("nan"), 1.]])], 40)
        with self.assertRaises(ValueError):
            consistency_loss([torch.zeros(2, 3)], 40, loss="mse")


if __name__ == "__main__":
    unittest.main()
