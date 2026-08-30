"""Regression tests for requested-versus-represented input-set identity."""

import unittest

import torch

from act.pipeline.moe.certified_artifact_identity import represented_linf_box


class RepresentedSetIdentityTests(unittest.TestCase):
    def test_positive_request_can_collapse_to_a_point(self):
        center = torch.tensor([0.25], dtype=torch.float32)
        lower, upper, identity = represented_linf_box(center, 1e-9)
        self.assertGreater(identity.requested_radius, 0.0)
        self.assertTrue(torch.equal(lower, center))
        self.assertTrue(torch.equal(upper, center))
        self.assertEqual(identity.zero_box_width_coordinates, 1)
        self.assertEqual(identity.maximum_box_width, 0.0)

        # A toy identity network sees a singleton even though the requested
        # real-valued perturbation radius is positive.
        network = torch.nn.Linear(1, 1, bias=False, dtype=torch.float32)
        with torch.no_grad():
            network.weight.fill_(1.0)
        self.assertTrue(torch.equal(network(lower), network(upper)))

    def test_ordinary_radius_is_not_ulp_degenerate(self):
        center = torch.tensor([0.25, 0.5, 0.75], dtype=torch.float32)
        _lower, _upper, identity = represented_linf_box(center, 0.5 / 255.0)
        self.assertEqual(identity.zero_box_width_coordinates, 0)
        self.assertGreater(identity.minimum_box_width, 0.0)
        self.assertGreater(identity.effective_lower_linf, 0.0)
        self.assertGreater(identity.effective_upper_linf, 0.0)

    def test_identity_changes_at_next_float32_box(self):
        center = torch.tensor([0.25], dtype=torch.float32)
        _l0, _u0, collapsed = represented_linf_box(center, 1e-9)
        _l1, _u1, expanded = represented_linf_box(center, 2e-8)
        self.assertNotEqual(collapsed.lower_sha256, expanded.lower_sha256)
        self.assertNotEqual(collapsed.upper_sha256, expanded.upper_sha256)
        self.assertEqual(collapsed.zero_box_width_coordinates, 1)
        self.assertEqual(expanded.zero_box_width_coordinates, 0)

    def test_clipping_is_part_of_represented_identity(self):
        center = torch.tensor([0.0, 1.0], dtype=torch.float32)
        _lower, _upper, identity = represented_linf_box(center, 0.1)
        self.assertEqual(identity.unchanged_lower_coordinates, 1)
        self.assertEqual(identity.unchanged_upper_coordinates, 1)
        self.assertGreater(identity.maximum_box_width, 0.0)


if __name__ == "__main__":
    unittest.main()
