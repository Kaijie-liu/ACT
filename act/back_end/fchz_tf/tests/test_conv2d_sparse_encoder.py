# ===- tests/test_conv2d_sparse_encoder.py — Goal 1 Week 1 Day 1 =====#
"""Unit tests for CONV2D sparse affine encoder.

Verifies: A @ x + b == torch.nn.functional.conv2d(x, W, bias)
"""
import unittest
import numpy as np


class TestConv2DSparseEncoder(unittest.TestCase):

    def test_output_shape_basic(self):
        from act.back_end.fchz_tf.conv2d_sparse_encoder import conv2d_output_shape
        # 3-ch 8x8 input, 3x3 kernel, stride=1, padding=0 → 6x6 output
        out = conv2d_output_shape((1, 3, 8, 8), (16, 3, 3, 3), stride=(1, 1), padding=(0, 0))
        self.assertEqual(out, (1, 16, 6, 6))

    def test_output_shape_stride_padding(self):
        from act.back_end.fchz_tf.conv2d_sparse_encoder import conv2d_output_shape
        # CIFAR-like: 3x32x32, 3x3 kernel, stride=2, padding=1 → 16x16
        out = conv2d_output_shape((1, 3, 32, 32), (16, 3, 3, 3), stride=(2, 2), padding=(1, 1))
        self.assertEqual(out, (1, 16, 16, 16))

    def test_sparse_affine_matches_torch_small(self):
        """3-ch 8x8 input, 8-ch 3x3 kernel, no stride/padding."""
        from act.back_end.fchz_tf.conv2d_sparse_encoder import (
            conv2d_to_sparse_affine, verify_sparse_affine_against_torch)
        rng = np.random.default_rng(42)
        W = rng.standard_normal((8, 3, 3, 3)).astype(np.float64)
        b = rng.standard_normal((8,)).astype(np.float64)
        input_shape = (1, 3, 8, 8)
        A, b_flat, out_shape = conv2d_to_sparse_affine(W, b, input_shape,
                                                                                  stride=(1, 1), padding=(0, 0))
        self.assertEqual(out_shape, (1, 8, 6, 6))
        self.assertEqual(A.shape, (8 * 6 * 6, 3 * 8 * 8))
        ok, diff = verify_sparse_affine_against_torch(A, b_flat, W, b, input_shape,
                                                                                  stride=(1, 1), padding=(0, 0),
                                                                                  n_samples=5)
        self.assertTrue(ok, f"Sparse affine differs from torch by {diff}")

    def test_sparse_affine_with_stride(self):
        """Stride=2 conv."""
        from act.back_end.fchz_tf.conv2d_sparse_encoder import (
            conv2d_to_sparse_affine, verify_sparse_affine_against_torch)
        rng = np.random.default_rng(7)
        W = rng.standard_normal((4, 3, 3, 3)).astype(np.float64)
        b = None
        input_shape = (1, 3, 8, 8)
        A, b_flat, out_shape = conv2d_to_sparse_affine(W, b, input_shape,
                                                                                  stride=(2, 2), padding=(0, 0))
        self.assertEqual(out_shape, (1, 4, 3, 3))
        ok, diff = verify_sparse_affine_against_torch(A, b_flat, W, b, input_shape,
                                                                                  stride=(2, 2), padding=(0, 0),
                                                                                  n_samples=5)
        self.assertTrue(ok)

    def test_sparse_affine_with_padding(self):
        """Padding=1 conv."""
        from act.back_end.fchz_tf.conv2d_sparse_encoder import (
            conv2d_to_sparse_affine, verify_sparse_affine_against_torch)
        rng = np.random.default_rng(11)
        W = rng.standard_normal((4, 2, 3, 3)).astype(np.float64)
        b = rng.standard_normal((4,)).astype(np.float64)
        input_shape = (1, 2, 6, 6)
        A, b_flat, out_shape = conv2d_to_sparse_affine(W, b, input_shape,
                                                                                  stride=(1, 1), padding=(1, 1))
        self.assertEqual(out_shape, (1, 4, 6, 6))
        ok, diff = verify_sparse_affine_against_torch(A, b_flat, W, b, input_shape,
                                                                                  stride=(1, 1), padding=(1, 1),
                                                                                  n_samples=5)
        self.assertTrue(ok)

    def test_sparse_affine_cifar_first_conv(self):
        """CIFAR-style 3x32x32 → 16x16x16 (stride=2, padding=1)."""
        from act.back_end.fchz_tf.conv2d_sparse_encoder import (
            conv2d_to_sparse_affine, verify_sparse_affine_against_torch)
        rng = np.random.default_rng(42)
        W = rng.standard_normal((16, 3, 3, 3)).astype(np.float64)
        b = rng.standard_normal((16,)).astype(np.float64)
        input_shape = (1, 3, 32, 32)
        A, b_flat, out_shape = conv2d_to_sparse_affine(W, b, input_shape,
                                                                                  stride=(2, 2), padding=(1, 1))
        self.assertEqual(out_shape, (1, 16, 16, 16))
        self.assertEqual(A.shape, (16 * 16 * 16, 3 * 32 * 32))
        # Should be sparse
        density = A.nnz / (A.shape[0] * A.shape[1])
        self.assertLess(density, 0.01, f"density {density} too high — encoding bug?")
        # Verify correctness
        ok, diff = verify_sparse_affine_against_torch(A, b_flat, W, b, input_shape,
                                                                                  stride=(2, 2), padding=(1, 1),
                                                                                  n_samples=3)
        self.assertTrue(ok)

    def test_sparse_affine_relusplitter_typical(self):
        """Relusplitter typical layer: 16x32x32 → 32x16x16."""
        from act.back_end.fchz_tf.conv2d_sparse_encoder import (
            conv2d_to_sparse_affine, verify_sparse_affine_against_torch)
        rng = np.random.default_rng(99)
        W = rng.standard_normal((32, 16, 3, 3)).astype(np.float64)
        b = rng.standard_normal((32,)).astype(np.float64)
        input_shape = (1, 16, 32, 32)
        A, b_flat, out_shape = conv2d_to_sparse_affine(W, b, input_shape,
                                                                                  stride=(2, 2), padding=(1, 1))
        self.assertEqual(out_shape, (1, 32, 16, 16))
        ok, diff = verify_sparse_affine_against_torch(A, b_flat, W, b, input_shape,
                                                                                  stride=(2, 2), padding=(1, 1),
                                                                                  n_samples=3)
        self.assertTrue(ok)

    def test_zero_weight_omitted(self):
        """Verify that exact zero weights aren't added to sparse matrix (saves memory)."""
        from act.back_end.fchz_tf.conv2d_sparse_encoder import conv2d_to_sparse_affine
        W = np.zeros((4, 3, 3, 3), dtype=np.float64)
        W[0, 0, 0, 0] = 1.0   # only one nonzero
        b = None
        input_shape = (1, 3, 8, 8)
        A, _, _ = conv2d_to_sparse_affine(W, b, input_shape, stride=(1, 1), padding=(0, 0))
        # Should have many fewer than n_out * (3*3*3) = 27*36 = 972 nonzeros
        # Actually with single weight=1, we get 6*6 = 36 nonzeros (one per output position for co=0)
        self.assertLess(A.nnz, 50)


if __name__ == '__main__':
    unittest.main()
