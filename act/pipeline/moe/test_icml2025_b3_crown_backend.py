"""Executable bound smoke for the B3 external-normalization backend path."""

from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.crown_adapter_cohort import _crown_bounds
from act.pipeline.moe.icml2025_b3 import normalize_unit_pixel_box


@unittest.skipUnless(
    importlib.util.find_spec("auto_LiRPA") is not None,
    "auto_LiRPA is tested in its isolated backend environment",
)
class B3CrownBackendTest(unittest.TestCase):
    def test_external_normalization_allows_real_crown_bound(self) -> None:
        lower = torch.zeros(1, 3, 2, 2, dtype=torch.float32)
        upper = torch.full_like(lower, 0.01)
        center, normalized_lower, normalized_upper = normalize_unit_pixel_box(
            lower, upper
        )
        torch.manual_seed(0)
        module = nn.Sequential(nn.Flatten(), nn.Linear(12, 2))
        kwargs = dict(
            center=center,
            lower=normalized_lower,
            upper=normalized_upper,
            property_rows=((np.asarray([1.0, -1.0]), 0.0),),
            device="cpu",
            tolerance=1e-7,
            method="CROWN",
        )
        tracked = _crown_bounds(module, track_gradients=True, **kwargs)
        result = _crown_bounds(
            module,
            track_gradients=False,
            **kwargs,
        )
        self.assertNotEqual(result["status"], "ERROR")
        self.assertTrue(result["complete"])
        self.assertEqual(result["property_rows"], 1)
        self.assertFalse(result["gradient_tracking_enabled"])
        np.testing.assert_allclose(result["lower_bounds"], tracked["lower_bounds"])
        np.testing.assert_allclose(result["upper_bounds"], tracked["upper_bounds"])


if __name__ == "__main__":
    unittest.main()
