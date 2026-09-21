import unittest
from unittest.mock import patch

import torch
from torch import nn

from act.back_end.moe.class_separated_top1 import (
    ClassSeparatedTop1, classification_rows, validate_replay, verify_class_separated_box,
)


def affine(weights, biases):
    model = nn.Linear(1, len(biases), dtype=torch.float64)
    with torch.no_grad():
        model.weight.copy_(torch.tensor(weights, dtype=torch.float64).reshape(-1, 1))
        model.bias.copy_(torch.tensor(biases, dtype=torch.float64))
    return model.eval()


class SeparatedTop1Controls(unittest.TestCase):
    def setUp(self):
        self.previous_dtype = torch.get_default_dtype()
        torch.set_num_threads(1)
        self.x = torch.zeros(1, 1, dtype=torch.float64)
        self.lo, self.hi = self.x - .1, self.x + .1

    def tearDown(self):
        # ACT auto-initialization must not change unrelated float32 controls.
        torch.set_default_dtype(self.previous_dtype)

    def model(self, scores=(1., -1.), a=(2., 1.), b=(3.,)):
        return ClassSeparatedTop1(affine([0, 0], scores),
            [affine([0] * len(a), a), affine([0] * len(b), b)], (len(a), len(b)))

    def verify(self, model, q=None):
        if q is None:
            q = classification_rows(model.total_classes, 0)
        return verify_class_separated_box(model, center=self.x, lower=self.lo, upper=self.hi,
            rows=q, thresholds=torch.full((len(q),), 1e-7, dtype=torch.float64), total_seconds=20)

    def test_zero_fill_and_negative_selected_scores(self):
        m = self.model(scores=(-1., -2.))
        self.assertTrue(torch.equal(m(self.x), torch.tensor([[2., 1., 0.]], dtype=torch.float64)))
        r = self.verify(m)
        self.assertEqual(r["status"], "POSITIVE")
        self.assertLess(r["nonzero_obligations"][0]["upper"], 0)
        self.assertFalse(r["source_complete"])

    def test_zero_selected_score_not_hard_one(self):
        m = self.model(scores=(0., -1.))
        self.assertFalse(torch.isfinite(m(self.x)).all())
        self.assertEqual(self.verify(m)["status"], "UNKNOWN")

    def test_classification_needs_positive_global_logit(self):
        m = self.model(a=(-1., -2.))  # expert correct internally, full model not correct
        self.assertEqual(m(self.x).argmax(1).item(), 2)
        self.assertEqual(self.verify(m)["status"], "UNSAFE_REPLAYED")

    def test_all_tie_legal_routes_and_changed_dimensions(self):
        m = self.model(scores=(1., 1.), a=(2., 1., .5), b=(3., 2.))
        q = torch.ones(1, 5, dtype=torch.float64)
        r = self.verify(m, q)
        self.assertEqual(r["status"], "POSITIVE")
        self.assertEqual(r["candidates"], [0, 1])
        self.assertEqual(len(r["nonzero_obligations"]), 2)

    def test_wrong_route_cannot_prove_global_label(self):
        m = self.model(scores=(1., 1.))
        r = self.verify(m)
        self.assertNotEqual(r["status"], "POSITIVE")

    def test_score_crossing_zero_not_simplified(self):
        m = ClassSeparatedTop1(affine([1., 0.], [0., -1.]),
            [affine([0, 0], [2, 1]), affine([0], [1])], (2, 1))
        self.assertEqual(self.verify(m)["status"], "UNKNOWN")

    def test_original_replay_domain_and_nonfinite(self):
        m = self.model(a=(-1., -2.))
        q = classification_rows(3, 0)
        d = torch.zeros(2, dtype=torch.float64)
        self.assertFalse(validate_replay(m, self.x + 1, self.lo, self.hi, q, d))
        self.assertFalse(validate_replay(m, self.x * float('nan'), self.lo, self.hi, q, d))
        self.assertFalse(validate_replay(self.model(scores=(0., -1.)), self.x, self.lo, self.hi, q, d))

    def test_explicit_scope_no_automatic_dtype_or_mode_change(self):
        with self.assertRaises(ValueError):
            ClassSeparatedTop1(nn.Linear(1, 2), [affine([0], [1]), affine([0], [1])], (1, 1))
        m = self.model().float()
        with self.assertRaises(ValueError):
            self.verify(m)
        with self.assertRaises(ValueError):
            ClassSeparatedTop1.from_metamoe(self.model())

    def test_late_complete_record_not_accepted(self):
        m = self.model(a=(-1., -2.))
        with patch('act.back_end.moe.class_separated_top1.time.monotonic', side_effect=[0., 21., 21.]):
            self.assertEqual(self.verify(m)["status"], "TIMEOUT")


if __name__ == '__main__':
    unittest.main()
