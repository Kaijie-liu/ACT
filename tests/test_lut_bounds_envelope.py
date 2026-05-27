"""Soundness containment for LUT_BOUNDS dynamic-Slice envelope (Option A).

Background
==========
cctsdb_yolo_2023 surfaces a `Slice(initializer, starts, ends, axes, steps)`
where the initializer is a static tensor T, the spatial window size
``end - start`` is constant, but `start` itself is derived from a
bounded VNNLIB input variable. After the R12 fail-closed gate, this
pattern can no longer be miscompiled via sample substitution.

The sound representation (`CCTSDB_DYNAMIC_SLICE_DESIGN.md`, Option A)
is a `LUT_BOUNDS` zero-indegree source layer whose `params['lb']`
and `params['ub']` are the precomputed per-output-position envelopes:

    lb_tensor[j] = min over candidate windows of T[window_at(start, j)]
    ub_tensor[j] = max over candidate windows of T[window_at(start, j)]

The candidate set is exactly the integer lattice of `start` values
within `[starts_lb, starts_ub]`. Each runtime crop is therefore
contained in `[lb_tensor, ub_tensor]` element-wise.

Tests pin two invariants:

1. `precompute_lut_envelope(T, window_size, starts_lb, starts_ub)`
   produces lb <= ub element-wise, and matches a brute-force
   per-position min/max ground truth.

2. `tf_lut_bounds` honours the sealed bounds: any concrete runtime
   crop (random `start` in box, sampled per coord) falls inside the
   layer's output bounds.

A failure in either is a soundness bug in the dynamic-Slice path.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import torch

from act.back_end.core import Bounds, Layer
from act.back_end.layer_schema import LayerKind
from act.back_end.interval_tf.tf_mlp import (
    tf_lut_bounds,
    precompute_lut_envelope,
)


class _DeviceIsolated(unittest.TestCase):
    def setUp(self):
        super().setUp()
        self._dev = torch.get_default_device() if hasattr(torch, "get_default_device") else None
        self._dt = torch.get_default_dtype()
        try:
            torch.set_default_device("cpu")
        except Exception:
            pass
        torch.set_default_dtype(torch.float64)

    def tearDown(self):
        try:
            torch.set_default_device(self._dev or "cpu")
        except Exception:
            pass
        torch.set_default_dtype(self._dt)
        super().tearDown()


def _brute_force_envelope(T, window_size, starts_lb, starts_ub, steps=None):
    """Reference implementation: ranges over the integer lattice of
    starts, slices T, accumulates min/max. Used only to cross-check the
    production `precompute_lut_envelope`."""
    from itertools import product
    rank = len(window_size)
    if steps is None:
        steps = (1,) * rank
    cand_per_axis = [list(range(int(starts_lb[i]), int(starts_ub[i]) + 1))
                     for i in range(rank)]
    out_lb = torch.full(window_size, float("inf"), dtype=torch.float64)
    out_ub = torch.full(window_size, float("-inf"), dtype=torch.float64)
    for s in product(*cand_per_axis):
        slices = []
        valid = True
        for i, si in enumerate(s):
            end = si + window_size[i] * steps[i]
            if si < 0 or end > T.shape[i]:
                valid = False
                break
            slices.append(slice(si, end, steps[i]))
        if not valid:
            continue
        w = T[tuple(slices)]
        out_lb = torch.minimum(out_lb, w)
        out_ub = torch.maximum(out_ub, w)
    return out_lb, out_ub


class TestPrecomputeLutEnvelope(_DeviceIsolated):
    def test_envelope_matches_brute_force_1d(self):
        torch.manual_seed(0)
        T = torch.randn(16, dtype=torch.float64)
        for window in [(4,), (8,)]:
            for s_lb, s_ub in [(0, 5), (2, 10)]:
                lb, ub = precompute_lut_envelope(
                    T, window_size=window, starts_lb=(s_lb,), starts_ub=(s_ub,),
                )
                ref_lb, ref_ub = _brute_force_envelope(
                    T, window_size=window, starts_lb=(s_lb,), starts_ub=(s_ub,),
                )
                self.assertTrue(torch.allclose(lb, ref_lb))
                self.assertTrue(torch.allclose(ub, ref_ub))
                # Invariant: lb <= ub everywhere.
                self.assertTrue(torch.all(lb <= ub),
                                f"envelope lb>ub at some position; window={window} s=[{s_lb},{s_ub}]")

    def test_envelope_matches_brute_force_2d(self):
        torch.manual_seed(1)
        T = torch.randn(12, 10, dtype=torch.float64)
        lb, ub = precompute_lut_envelope(
            T, window_size=(3, 4), starts_lb=(0, 0), starts_ub=(5, 4),
        )
        ref_lb, ref_ub = _brute_force_envelope(
            T, window_size=(3, 4), starts_lb=(0, 0), starts_ub=(5, 4),
        )
        self.assertTrue(torch.allclose(lb, ref_lb))
        self.assertTrue(torch.allclose(ub, ref_ub))
        self.assertEqual(tuple(lb.shape), (3, 4))

    def test_singleton_start_recovers_constant_slice(self):
        """starts_lb == starts_ub means only one runtime choice. The
        envelope must collapse to a singleton: lb == ub == T[that slice]."""
        T = torch.arange(20, dtype=torch.float64)
        lb, ub = precompute_lut_envelope(
            T, window_size=(4,), starts_lb=(3,), starts_ub=(3,),
        )
        expected = T[3:7]
        self.assertTrue(torch.equal(lb, expected))
        self.assertTrue(torch.equal(ub, expected))

    def test_oob_starts_skipped(self):
        """Starts whose window extends past T's edge are skipped (runtime
        would raise). At least one valid start must remain, else the
        function raises."""
        T = torch.arange(10, dtype=torch.float64)
        # window=5, starts in [3, 8]: 3,4,5 are valid (3+5=8<=10, 4+5=9<=10, 5+5=10<=10);
        # 6,7,8 push past 10 (max 8+5=13). 3 valid starts.
        lb, ub = precompute_lut_envelope(
            T, window_size=(5,), starts_lb=(3,), starts_ub=(8,),
        )
        ref_lb, ref_ub = _brute_force_envelope(
            T, window_size=(5,), starts_lb=(3,), starts_ub=(8,),
        )
        self.assertTrue(torch.equal(lb, ref_lb))
        self.assertTrue(torch.equal(ub, ref_ub))


class TestTfLutBoundsContainment(_DeviceIsolated):
    def _make_lut_layer(self, lb_t, ub_t):
        n = lb_t.numel()
        return Layer(
            id=0, kind=LayerKind.LUT_BOUNDS.value,
            in_vars=[], out_vars=list(range(n)),
            params={
                "lb": lb_t, "ub": ub_t,
                "input_shape": tuple(lb_t.shape), "output_shape": tuple(lb_t.shape),
            },
        )

    def test_runtime_crops_inside_envelope(self):
        """For many random starts in [s_lb, s_ub], the resulting T[crop]
        must be contained in the layer's output bounds. Probes the
        soundness contract end-to-end."""
        torch.manual_seed(0)
        T = torch.randn(20, 15, dtype=torch.float64)
        window = (4, 3)
        s_lb = (1, 2)
        s_ub = (12, 8)

        lb_t, ub_t = precompute_lut_envelope(
            T, window_size=window, starts_lb=s_lb, starts_ub=s_ub,
        )
        L = self._make_lut_layer(lb_t, ub_t)
        # Dummy Bin to set batch dim
        dummy = Bounds(torch.zeros(1, 0, dtype=torch.float64),
                       torch.zeros(1, 0, dtype=torch.float64))
        fact = tf_lut_bounds(L, dummy)
        out_lb = fact.bounds.lb.view(window)
        out_ub = fact.bounds.ub.view(window)

        # Sample 200 random integer starts in [s_lb, s_ub] and verify
        # the resulting crops are contained.
        rng = torch.Generator()
        rng.manual_seed(42)
        for _ in range(200):
            s0 = int(torch.randint(s_lb[0], s_ub[0] + 1, (1,), generator=rng).item())
            s1 = int(torch.randint(s_lb[1], s_ub[1] + 1, (1,), generator=rng).item())
            crop = T[s0:s0 + window[0], s1:s1 + window[1]]
            self.assertEqual(crop.shape, window)
            self.assertTrue(
                torch.all(crop >= out_lb - 1e-12),
                f"crop at ({s0},{s1}) has value below LUT lb"
            )
            self.assertTrue(
                torch.all(crop <= out_ub + 1e-12),
                f"crop at ({s0},{s1}) has value above LUT ub"
            )

    def test_tf_rejects_lb_gt_ub_params(self):
        """The tf must reject malformed params (lb > ub somewhere) so a
        bug in the precompute helper fails loudly rather than silently
        feeding an unsound box downstream."""
        bad_lb = torch.tensor([1.0, 2.0, 3.0])
        bad_ub = torch.tensor([0.5, 1.5, 2.5])  # all < lb
        L = self._make_lut_layer(bad_lb, bad_ub)
        dummy = Bounds(torch.zeros(1, 0, dtype=torch.float64),
                       torch.zeros(1, 0, dtype=torch.float64))
        with self.assertRaisesRegex(ValueError, "lb > ub"):
            tf_lut_bounds(L, dummy)


if __name__ == "__main__":
    unittest.main(verbosity=2)
