"""Tests for endpoint-blind common-cohort selection."""

from __future__ import annotations

import json
import unittest

from act.pipeline.moe.experiment1 import PROJECT_ROOT
from act.pipeline.moe.freeze_staged_multimodel_bundle import (
    select_common_clean_correct,
)
from act.pipeline.moe.run_staged_verifier_confirmatory import _validate_registration


class CommonSelectionTests(unittest.TestCase):
    def test_selects_only_common_clean_correct_in_order(self) -> None:
        labels = [0, 1, 2, 3, 4, 5]
        predictions = {
            "seed0": [0, 1, 9, 3, 4, 5],
            "seed1": [0, 9, 2, 3, 4, 5],
            "seed2": [0, 1, 2, 3, 9, 5],
        }
        selected = select_common_clean_correct(
            predictions,
            labels,
            start_index=0,
            sample_count=2,
            excluded_indices={0},
        )
        self.assertEqual([row["dataset_index"] for row in selected], [3, 5])
        self.assertEqual([row["sample_rank"] for row in selected], [0, 1])
        self.assertEqual(
            selected[0]["clean_predictions"],
            {"seed0": 3, "seed1": 3, "seed2": 3},
        )

    def test_rejects_mismatched_prediction_lengths(self) -> None:
        with self.assertRaisesRegex(ValueError, "lengths differ"):
            select_common_clean_correct(
                {"seed0": [0], "seed1": [0, 1]},
                [0, 1],
                start_index=0,
                sample_count=1,
                excluded_indices=set(),
            )

    def test_each_verdict_config_is_bound_to_its_registered_model(self) -> None:
        for model_id in ("seed0", "seed1", "seed2"):
            path = (
                PROJECT_ROOT
                / f"act/pipeline/moe/configs/staged_verifier_multimodel_{model_id}_fixed2_r1.json"
            )
            config = json.loads(path.read_text(encoding="utf-8"))
            selection = _validate_registration(config)
            self.assertEqual(
                selection["models"][model_id]["checkpoint_sha256"],
                config["checkpoint_sha256"],
            )


if __name__ == "__main__":
    unittest.main()
