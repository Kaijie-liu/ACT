"""Regression test for the tf_conv2d non-square-input bug.

History: collins_rul_cnn_2022 single-instance smoke (2026-05-25) raised
``RuntimeError: Calculated padded input size per channel: (21 x 8).
Kernel size: (6 x 20). Kernel size can't be greater than actual input
size`` at the model's fc_1 Conv. Root cause: tf_conv2d ignored the
converter-stamped ``L.params['input_shape']`` and instead re-derived
spatial dims from numel via a square-first / descending-factor search.
For the collins layer chain (H=16,12,8,6; W=20 throughout) the very
first non-square layer (conv_2) got TRANSPOSED to (20,16); the
corruption cascaded and at fc_1 landed on (21,8) — the first
factorization of 168 the loop reached.

Fix: prefer L.params['input_shape'] when its per-sample numel matches
``Bin.lb[0].numel()``; only fall back to inference when metadata is
genuinely missing or stale.

This test pins the fix by:
  (a) constructing a non-square Conv2d chain whose numel admits a wrong
      factorization the old loop would have picked, and
  (b) running tf_conv2d on a hand-built ACT Layer and asserting the
      output shape matches the metadata, with no RuntimeError.

Symptom asymmetry test: H != W, kernel (k_h, 1) chain. If tf_conv2d
ever re-introduces shape inference, this test catches it.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from act.back_end.core import Layer, Bounds
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_cnn import tf_conv2d


def _make_layer(in_shape, out_shape, weight):
    n_in = in_shape[1] * in_shape[2] * in_shape[3]
    n_out = out_shape[1] * out_shape[2] * out_shape[3]
    in_vars = list(range(n_in))
    out_vars = list(range(n_in, n_in + n_out))
    return Layer(
        id=1,
        kind=LayerKind.CONV2D.value,
        in_vars=in_vars,
        out_vars=out_vars,
        params={
            "weight": weight,
            "in_channels": in_shape[1],
            "out_channels": out_shape[1],
            "kernel_size": (weight.shape[2], weight.shape[3]),
            "stride": (1, 1),
            "padding": (0, 0),
            "dilation": (1, 1),
            "groups": 1,
            "input_shape": in_shape,
            "output_shape": out_shape,
        },
    )


class TestTfConv2dNonSquareInput(unittest.TestCase):
    def test_collins_style_kx1_chain_preserves_orientation(self):
        """A chain of (k_h, 1) kernels on a non-square input must NOT
        transpose H/W when tf_conv2d re-derives spatial dims. The collins
        model exhibits exactly this pattern: (1,1,20,20) → (1,5,16,20) →
        (1,5,12,20) → (1,5,8,20) → (1,5,6,20). The fix is verified by
        building the conv_2 layer (whose input numel = 1600 = 5*16*20
        admits both 16x20 AND the wrong 20x16) and asserting the output
        shape matches (1, 10, 12, 20)."""
        torch.manual_seed(0)
        weight = torch.randn(10, 5, 5, 1, dtype=torch.float64)
        in_shape = (1, 5, 16, 20)
        out_shape = (1, 10, 12, 20)
        L = _make_layer(in_shape, out_shape, weight)

        n_in = 5 * 16 * 20
        Bin = Bounds(
            lb=torch.zeros(1, n_in, dtype=torch.float64),
            ub=torch.zeros(1, n_in, dtype=torch.float64),
        )

        fact = tf_conv2d(L, Bin)
        n_out_expected = out_shape[1] * out_shape[2] * out_shape[3]
        self.assertEqual(
            fact.bounds.lb.shape, (1, n_out_expected),
            f"output flat shape must be (B, {n_out_expected}); got {tuple(fact.bounds.lb.shape)}. "
            f"A regression to the square-first heuristic would TRANSPOSE H/W and produce a "
            f"different output size."
        )

    def test_collins_fc1_pinch_layer(self):
        """The fc_1 Conv has kernel (6, 20) matching the remaining spatial
        (6, 20). The square-first heuristic on 600 / 5 = 120 would have
        picked (20, 6) first (in_h=20, in_w=6, the largest factor of 120
        not exceeding sqrt+10=20), making F.conv2d raise. After the fix
        tf_conv2d trusts metadata and the call succeeds."""
        torch.manual_seed(1)
        weight = torch.randn(100, 5, 6, 20, dtype=torch.float64)
        in_shape = (1, 5, 6, 20)
        out_shape = (1, 100, 1, 1)
        L = _make_layer(in_shape, out_shape, weight)

        n_in = 5 * 6 * 20
        Bin = Bounds(
            lb=torch.zeros(1, n_in, dtype=torch.float64),
            ub=torch.zeros(1, n_in, dtype=torch.float64),
        )

        try:
            fact = tf_conv2d(L, Bin)
        except RuntimeError as e:
            self.fail(
                f"tf_conv2d raised on collins fc_1 layer — fix has regressed: {e}"
            )
        self.assertEqual(fact.bounds.lb.shape, (1, 100))

    def test_metadata_mismatch_falls_back(self):
        """When metadata is stale (per-sample numel mismatches the actual
        Bin), tf_conv2d MUST fall back to the inference loop rather than
        crash or silently use wrong dims. Test: pass a square input where
        metadata claims non-square mismatch."""
        torch.manual_seed(2)
        # Real input: (1, 1, 4, 4) with numel 16
        weight = torch.randn(2, 1, 2, 2, dtype=torch.float64)
        # Stale metadata claiming (1, 1, 8, 8) -> numel 64, won't match
        L = _make_layer(
            in_shape=(1, 1, 8, 8),
            out_shape=(1, 2, 3, 3),
            weight=weight,
        )

        Bin = Bounds(
            lb=torch.zeros(1, 16, dtype=torch.float64),
            ub=torch.zeros(1, 16, dtype=torch.float64),
        )
        # numel mismatch (16 vs metadata's 64) -> fall through to inference,
        # which picks (1, 1, 4, 4) perfect-square. F.conv2d returns (1, 2, 3, 3).
        fact = tf_conv2d(L, Bin)
        self.assertEqual(fact.bounds.lb.shape, (1, 18))


if __name__ == "__main__":
    unittest.main(verbosity=2)
