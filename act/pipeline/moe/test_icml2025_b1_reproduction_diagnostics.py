from pathlib import Path
import unittest

from act.pipeline.moe.icml2025_b1_reproduction_diagnostics import (
    _source_anchor,
    _trajectory_summary,
)


class B1ReproductionDiagnosticsTests(unittest.TestCase):
    def test_source_anchor_requires_unique_match(self) -> None:
        observed = _source_anchor(["alpha", "needle", "omega"], "needle")
        self.assertEqual(observed["line"], 2)
        with self.assertRaises(RuntimeError):
            _source_anchor(["needle", "needle"], "needle")

    def test_trajectory_summary_preserves_ratio_and_gap(self) -> None:
        validation = []
        for epoch in range(10, 131, 10):
            validation.append(
                {
                    "epoch": epoch,
                    "validation_standard_accuracy_percent": 34.22,
                    "validation_robust_accuracy_percent": 32.70,
                    "validation_ra_over_sa": 32.70 / 34.22,
                }
            )
        observed = _trajectory_summary(validation, [{"epoch": value} for value in range(1, 131)])
        self.assertAlmostEqual(
            observed["endpoint"]["robust_to_standard_ratio"], 32.70 / 34.22
        )
        self.assertAlmostEqual(
            observed["endpoint"]["paper_standard_gap_percentage_points"], 43.59
        )
        self.assertEqual(observed["training_epoch_rows"], 130)


if __name__ == "__main__":
    unittest.main()
