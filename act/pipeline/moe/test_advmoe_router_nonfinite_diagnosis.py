"""Tests for first-nonfinite AdvMoE diagnosis helpers."""

from __future__ import annotations

import unittest

import torch

from act.pipeline.moe.advmoe_router_nonfinite_diagnosis import (
    _named_details,
    all_finite,
)


class AdvMoeRouterNonfiniteDiagnosisTests(unittest.TestCase):
    def test_all_finite_accepts_empty_and_finite_states(self) -> None:
        self.assertTrue(all_finite([]))
        self.assertTrue(all_finite([torch.tensor([1.0, -2.0])]))

    def test_named_details_identifies_exact_bad_entry(self) -> None:
        details = _named_details(
            [
                ("good", torch.tensor([1.0, 2.0])),
                ("bad", torch.tensor([float("nan"), float("inf")])),
            ]
        )
        self.assertFalse(details["all_finite"])
        self.assertEqual(details["finite_elements"], 2)
        self.assertEqual(details["nonfinite_entries"][0]["name"], "bad")
        self.assertEqual(details["nonfinite_entries"][0]["nan_elements"], 1)
        self.assertEqual(details["nonfinite_entries"][0]["inf_elements"], 1)

    def test_named_details_accepts_finite_and_empty_collections(self) -> None:
        empty = _named_details([])
        finite = _named_details([("finite", torch.tensor([0.0, -1.0]))])
        self.assertTrue(empty["all_finite"])
        self.assertEqual(empty["elements"], 0)
        self.assertTrue(finite["all_finite"])
        self.assertEqual(finite["finite_elements"], 2)


if __name__ == "__main__":
    unittest.main()
