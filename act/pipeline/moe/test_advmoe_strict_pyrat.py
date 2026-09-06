"""Unit tests for the strict AdvMoE PyRAT bridge."""

from __future__ import annotations

import unittest

import numpy as np

from act.pipeline.moe.advmoe_strict_pyrat import (
    aggregate_strict_paths,
    classification_vnnlib,
    parse_pyrat_status,
)


class AdvMoeStrictPyratTests(unittest.TestCase):
    def test_vnnlib_has_closed_box_and_all_class_margins(self) -> None:
        text = classification_vnnlib(
            np.asarray([[[0.0, 0.25]]]),
            np.asarray([[[0.5, 1.0]]]),
            1,
            classes=3,
        )
        self.assertIn("(assert (>= X_0 0))", text)
        self.assertIn("(assert (<= X_1 1))", text)
        self.assertIn("(assert (>= Y_1 Y_0))", text)
        self.assertIn("(assert (>= Y_1 Y_2))", text)
        self.assertNotIn("(>= Y_1 Y_1)", text)

    def test_only_two_complete_safe_paths_form_safe_endpoint(self) -> None:
        self.assertEqual(
            aggregate_strict_paths(["SAFE", "SAFE"]),
            "SAFE_ALL_PATHS_DIRECTED_ROUNDING",
        )
        self.assertEqual(
            aggregate_strict_paths(["SAFE"]),
            "UNKNOWN_INCOMPLETE_PATH_COVERAGE",
        )
        self.assertEqual(
            aggregate_strict_paths(["SAFE", "UNSAFE"]),
            "UNKNOWN_STATIC_PATH_COUNTEREXAMPLE_NOT_LIFTED",
        )
        self.assertEqual(aggregate_strict_paths(["SAFE", "TIMEOUT"]), "TIMEOUT")

    def test_pyrat_status_parser_fails_closed(self) -> None:
        self.assertEqual(
            parse_pyrat_status("Result = SAFE", returncode=0, timed_out=False),
            "SAFE",
        )
        self.assertEqual(
            parse_pyrat_status("Result = SAFE", returncode=1, timed_out=False),
            "ERROR",
        )
        self.assertEqual(
            parse_pyrat_status("no result", returncode=0, timed_out=False),
            "ERROR",
        )
        self.assertEqual(
            parse_pyrat_status("Result = SAFE\nResult = UNKNOWN", returncode=0, timed_out=False),
            "ERROR",
        )
        self.assertEqual(
            parse_pyrat_status("", returncode=-1, timed_out=True),
            "TIMEOUT",
        )


if __name__ == "__main__":
    unittest.main()
