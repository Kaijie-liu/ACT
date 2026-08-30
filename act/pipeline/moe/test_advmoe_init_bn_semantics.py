"""Tests for explicit AdvMoE initialization BN semantics."""

from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from act.pipeline.moe.advmoe_init_bn_semantics import (
    _forward_semantics,
    aggregate_seed_rows,
    summarize_scores,
)


class _BnRouter(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(1)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        normalized = self.bn(value).mean(dim=(1, 2, 3))
        return torch.stack((normalized + 0.25, -normalized), dim=1)


class AdvMoeInitBnSemanticsTests(unittest.TestCase):
    def test_eval_and_train_batch_semantics_are_distinct_and_recorded(self) -> None:
        inputs = np.asarray([0.0, 1.0, 2.0, 3.0], dtype=np.float32).reshape(4, 1, 1, 1)
        eval_scores, eval_before, eval_after = _forward_semantics(
            _BnRouter(), inputs, device=torch.device("cpu"), batch_size=2, train_mode=False
        )
        train_scores, train_before, train_after = _forward_semantics(
            _BnRouter(), inputs, device=torch.device("cpu"), batch_size=2, train_mode=True
        )
        self.assertFalse(np.array_equal(eval_scores, train_scores))
        self.assertEqual(eval_before["rows"][0]["num_batches_tracked"], 0)
        self.assertEqual(eval_after["rows"][0]["num_batches_tracked"], 0)
        self.assertEqual(train_before["rows"][0]["num_batches_tracked"], 0)
        self.assertEqual(train_after["rows"][0]["num_batches_tracked"], 2)

    def test_summary_and_aggregate_preserve_collapse_target(self) -> None:
        collapsed = summarize_scores(np.asarray([[2.0, 1.0], [3.0, 0.0]]))
        balanced = summarize_scores(np.asarray([[2.0, 1.0], [0.0, 3.0]]))
        rows = [
            {
                "EVAL_DEFAULT_RUNNING_STATS": collapsed,
                "TRAIN_ORDERED_TEST_BATCH_STATS": balanced,
            }
        ]
        aggregate = aggregate_seed_rows(rows, [1234])
        self.assertEqual(aggregate["EVAL_DEFAULT_RUNNING_STATS"]["collapsed_seed_count"], 1)
        self.assertEqual(aggregate["EVAL_DEFAULT_RUNNING_STATS"]["collapse_target_counts"], {"0": 1})
        self.assertEqual(aggregate["TRAIN_ORDERED_TEST_BATCH_STATS"]["collapsed_seed_count"], 0)


if __name__ == "__main__":
    unittest.main()
