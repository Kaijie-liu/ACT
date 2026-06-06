"""S2 unit tests: chunked Conv vs dense Conv parity + tail soundness.

Acceptance criteria for these tests (Day 1 deliverable):
  1. apply_conv2d_chunked produces NUMERICALLY IDENTICAL output to
      apply_conv2d (no rounding-order difference; Conv is deterministic).
  2. chunk_size does not affect the result (1, 4, 256, 1024 all equal).
  3. Stride / padding / groups all handled identically.
  4. tail_radius propagation matches: |W| @ tail under chunked == dense.
  5. Memory profile estimator returns sensible values.

These tests pin the contract that chunked Conv is a memory-only
optimization, never a precision-or-soundness regression.
"""
from __future__ import annotations

import unittest

import numpy as np

from research.sc_hz.conv_chunked import (
    apply_conv2d_chunked, estimate_chunk_memory, adaptive_chunk_size,
)
from research.sc_hz.ops import apply_conv2d
from research.sc_hz.prune import PrunedState


def _make_state(n_in: int, K: int, seed: int = 20260605) -> PrunedState:
    rng = np.random.default_rng(seed)
    c = rng.normal(scale=0.5, size=n_in).astype(np.float64)
    G = rng.normal(scale=0.1, size=(n_in, K)).astype(np.float64)
    return PrunedState(c=c, G_kept=G, tail_radius=None,
                        metadata={"input_coord_origin": np.arange(K, dtype=np.int64)})


def _make_state_with_tail(n_in: int, K: int, seed: int = 20260605) -> PrunedState:
    rng = np.random.default_rng(seed)
    c = rng.normal(scale=0.5, size=n_in).astype(np.float64)
    G = rng.normal(scale=0.1, size=(n_in, K)).astype(np.float64)
    tail = np.abs(rng.normal(scale=0.05, size=n_in)).astype(np.float64)
    return PrunedState(c=c, G_kept=G, tail_radius=tail,
                        metadata={"input_coord_origin": np.arange(K, dtype=np.int64)})


def _make_conv_w(Co: int, Ci: int, kH: int, kW: int,
                  seed: int = 20260605) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    return rng.normal(scale=1.0 / np.sqrt(Ci * kH * kW),
                        size=(Co, Ci, kH, kW)).astype(np.float64)


class TestConvChunkedParityDenseVsChunked(unittest.TestCase):
    """Bit-equal output between chunked and dense apply_conv2d."""

    def setUp(self):
        self.Ci, self.Hi, self.Wi = 4, 8, 8
        self.Co, self.kH, self.kW = 6, 3, 3
        self.n_in = self.Ci * self.Hi * self.Wi
        self.K = 20
        self.state = _make_state(self.n_in, self.K)
        self.W = _make_conv_w(self.Co, self.Ci, self.kH, self.kW)
        self.b = np.array([0.01 * i for i in range(self.Co)], dtype=np.float64)

    def test_chunked_equals_dense_basic(self):
        ref, ref_shape = apply_conv2d(
            self.state, self.W, self.b,
            input_shape=(self.Ci, self.Hi, self.Wi),
            stride=1, padding=1, groups=1,
        )
        chunked, chunked_shape = apply_conv2d_chunked(
            self.state, self.W, self.b,
            input_shape=(self.Ci, self.Hi, self.Wi),
            stride=1, padding=1, groups=1,
            chunk_size=4,
        )
        self.assertEqual(ref_shape, chunked_shape)
        np.testing.assert_array_equal(ref.c, chunked.c)
        np.testing.assert_array_equal(ref.G_kept, chunked.G_kept)

    def test_chunked_equals_dense_no_bias(self):
        ref, _ = apply_conv2d(
            self.state, self.W, None,
            input_shape=(self.Ci, self.Hi, self.Wi),
            stride=1, padding=0, groups=1,
        )
        chunked, _ = apply_conv2d_chunked(
            self.state, self.W, None,
            input_shape=(self.Ci, self.Hi, self.Wi),
            stride=1, padding=0, groups=1,
            chunk_size=4,
        )
        np.testing.assert_array_equal(ref.c, chunked.c)
        np.testing.assert_array_equal(ref.G_kept, chunked.G_kept)


class TestConvChunkedChunkSizeIndependence(unittest.TestCase):
    """Output independent of chunk_size choice."""

    def test_chunk_sizes_match(self):
        Ci, Hi, Wi = 2, 6, 6
        Co, kH, kW = 4, 3, 3
        n_in = Ci * Hi * Wi
        K = 12
        state = _make_state(n_in, K)
        W = _make_conv_w(Co, Ci, kH, kW)
        b = np.zeros(Co)

        results = []
        for cs in [1, 2, 4, 5, 8, 16, 32, K, K + 5]:
            s, _ = apply_conv2d_chunked(
                state, W, b, input_shape=(Ci, Hi, Wi),
                stride=1, padding=1, groups=1, chunk_size=cs,
            )
            results.append((cs, s.c, s.G_kept))

        c0 = results[0][1]
        G0 = results[0][2]
        for cs, c, G in results[1:]:
            np.testing.assert_array_equal(c, c0,
                err_msg=f"chunk_size={cs} center diverges from baseline")
            np.testing.assert_array_equal(G, G0,
                err_msg=f"chunk_size={cs} generators diverge from baseline")


class TestConvChunkedStridePadGroups(unittest.TestCase):
    """Stride/padding/groups all match dense."""

    def _check(self, Ci, Hi, Wi, Co, kH, kW, stride, padding, groups, K=10):
        n_in = Ci * Hi * Wi
        state = _make_state(n_in, K)
        W = _make_conv_w(Co, Ci // groups, kH, kW)
        ref, ref_shape = apply_conv2d(
            state, W, None, input_shape=(Ci, Hi, Wi),
            stride=stride, padding=padding, groups=groups,
        )
        chunked, ch_shape = apply_conv2d_chunked(
            state, W, None, input_shape=(Ci, Hi, Wi),
            stride=stride, padding=padding, groups=groups, chunk_size=4,
        )
        self.assertEqual(ref_shape, ch_shape)
        np.testing.assert_array_equal(ref.c, chunked.c)
        np.testing.assert_array_equal(ref.G_kept, chunked.G_kept)

    def test_stride_2(self):
        self._check(4, 8, 8, 6, 3, 3, stride=2, padding=1, groups=1)

    def test_padding_0(self):
        self._check(4, 8, 8, 6, 3, 3, stride=1, padding=0, groups=1)

    def test_grouped_conv(self):
        # 4 input channels, 4 output channels, 4 groups (depthwise-ish)
        self._check(4, 8, 8, 4, 3, 3, stride=1, padding=1, groups=4)


class TestConvChunkedTailPropagation(unittest.TestCase):
    """Tail radius propagates identically and remains sound."""

    def test_tail_matches(self):
        Ci, Hi, Wi = 3, 8, 8
        Co, kH, kW = 6, 3, 3
        n_in = Ci * Hi * Wi
        K = 8
        state = _make_state_with_tail(n_in, K)
        W = _make_conv_w(Co, Ci, kH, kW)
        ref, _ = apply_conv2d(
            state, W, None, input_shape=(Ci, Hi, Wi),
            stride=1, padding=1, groups=1,
        )
        chunked, _ = apply_conv2d_chunked(
            state, W, None, input_shape=(Ci, Hi, Wi),
            stride=1, padding=1, groups=1, chunk_size=3,
        )
        np.testing.assert_array_equal(ref.tail_radius, chunked.tail_radius)


class TestConvChunkedMetadataLineage(unittest.TestCase):
    """input_coord_origin metadata is preserved unchanged (Conv is linear)."""

    def test_metadata_preserved(self):
        Ci, Hi, Wi = 3, 8, 8
        Co, kH, kW = 6, 3, 3
        n_in = Ci * Hi * Wi
        K = 10
        state = _make_state(n_in, K)
        # mark some columns as relu-slack
        origin = np.arange(K, dtype=np.int64)
        origin[7:] = -1
        state.metadata["input_coord_origin"] = origin
        W = _make_conv_w(Co, Ci, kH, kW)
        chunked, _ = apply_conv2d_chunked(
            state, W, None, input_shape=(Ci, Hi, Wi),
            stride=1, padding=1, groups=1, chunk_size=4,
        )
        np.testing.assert_array_equal(
            chunked.metadata["input_coord_origin"], origin,
        )


class TestEstimateChunkMemory(unittest.TestCase):
    def test_estimate_chunk_memory_basic(self):
        p = estimate_chunk_memory((3, 32, 32), (64, 32, 32), 256)
        # input: 256 * 3 * 32 * 32 * 8 = 6.29 MB
        self.assertEqual(p.transient_input_bytes, 256 * 3 * 32 * 32 * 8)
        # output: 256 * 64 * 32 * 32 * 8 = 134 MB
        self.assertEqual(p.transient_output_bytes, 256 * 64 * 32 * 32 * 8)
        # total < 200 MB
        self.assertLess(p.total_transient_gb, 0.2)

    def test_adaptive_chunk_size_budget(self):
        # 4 GB budget on a typical cifar mid-layer (C=64, H=W=16)
        cs = adaptive_chunk_size((64, 16, 16), (128, 16, 16), budget_gb=4.0)
        self.assertGreaterEqual(cs, 16)
        self.assertLessEqual(cs, 1024)

    def test_adaptive_chunk_size_tight_budget(self):
        # Tiny budget (1 KB) — should floor to min_chunk=16
        cs = adaptive_chunk_size((64, 16, 16), (128, 16, 16), budget_gb=1e-6)
        self.assertEqual(cs, 16)  # floor to min_chunk

    def test_adaptive_chunk_size_loose_budget(self):
        # Generous budget — should cap at max_chunk=1024
        cs = adaptive_chunk_size((3, 8, 8), (16, 8, 8), budget_gb=100.0)
        self.assertEqual(cs, 1024)


if __name__ == "__main__":
    unittest.main()
