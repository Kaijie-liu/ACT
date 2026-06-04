"""Regression test for `hz_dense` batched-matmul block-diagonal helper.

mscn_128d in VNN-COMP 2025 nn4sys uses ONNX MatMul patterns of the
form `(K, in) @ (in, out)` where K is the per-branch sample count
(2 / 3 / 6 for join / sample / predicate). The HZ representation
flattens the `K * in` data dim, so `hz_dense` must detect the
batching and expand the weight to a block-diagonal `(K*out, K*in)`
matrix. Mathematically identical to per-sample matmul; sound.

HARDENED detection (2026-06-02 morning user review): K is decided
SOLELY by the ACT DENSE layer's explicit shape metadata. Plain
dim-divisibility is NEVER enough to trigger expansion — a normal
DENSE with happen-to-divide dims must pass through unchanged.

These tests pin:
  1. No-op when no metadata is provided (hardened guard).
  2. No-op when shapes already match (plain DENSE with metadata).
  3. Correct block-diagonal expansion for K=2, 3, 6 when
     input_shape[-1] == in_features AND prod(input_shape) == hz_dim.
  4. Output of the expanded matmul equals concatenating K
     independent matmuls of W on each K-block.
  5. Bias also expands (repeated K times) so per-sample bias is
     applied independently.
  6. Negative: divisible dims without explicit input_shape do NOT
     trigger expansion (defends against false positives on plain
     DENSE where hz.dim happens to equal a multiple of in_features).
  7. Negative: input_shape last dim != in_features → no expansion
     (inconsistent metadata → fail-closed).
  8. Negative: prod(input_shape) != hz_dim → no expansion
     (front-end metadata stale → fail-closed).
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")

from act.back_end.hybridz_tf.hz_routing import (
    _maybe_batched_block_diag,
    _batched_matmul_K_from_metadata,
)


# ─── No-op cases ─────────────────────────────────────────────────────────


def test_no_metadata_never_expands():
    """Without input_shape, K must be 1 even when dims look batched."""
    W = torch.randn(128, 6, dtype=torch.float64)
    b = torch.randn(128, dtype=torch.float64)
    W2, b2, K = _maybe_batched_block_diag(
        W, b, hz_dim=18, input_shape=None, in_features=None
    )
    assert K == 1
    assert torch.equal(W2, W)
    assert torch.equal(b2, b)


def test_no_op_when_plain_dense_with_metadata():
    """input_shape (1, 6), in_features 6 → K=1, no expansion."""
    W = torch.randn(128, 6, dtype=torch.float64)
    b = torch.randn(128, dtype=torch.float64)
    W2, b2, K = _maybe_batched_block_diag(
        W, b, hz_dim=6, input_shape=(1, 6), in_features=6,
    )
    assert K == 1
    assert torch.equal(W2, W)
    assert torch.equal(b2, b)


# ─── Positive cases (batched MatMul) ─────────────────────────────────────


def test_k3_sample_branch_expansion():
    """L11 / L14 pattern: K=3 (sample branch). W (128, 6) -> (384, 18)."""
    W = torch.randn(128, 6, dtype=torch.float64)
    b = torch.randn(128, dtype=torch.float64)
    W2, b2, K = _maybe_batched_block_diag(
        W, b, hz_dim=18, input_shape=(1, 3, 6), in_features=6,
    )
    assert K == 3, f"expected K=3, got {K}"
    assert W2.shape == (384, 18), f"shape {W2.shape}"
    assert b2.shape == (384,), f"bias shape {b2.shape}"
    # Each diagonal block equals W, off-diagonal zero.
    for k in range(3):
        block = W2[k*128:(k+1)*128, k*6:(k+1)*6]
        assert torch.equal(block, W), f"block {k} mismatch"
    for i in range(3):
        for j in range(3):
            if i == j: continue
            off = W2[i*128:(i+1)*128, j*6:(j+1)*6]
            assert torch.all(off == 0), f"off-diag ({i},{j}) nonzero"
    # Bias repeats K times
    for k in range(3):
        assert torch.equal(b2[k*128:(k+1)*128], b)


def test_k6_predicate_branch_expansion():
    """L23 / L26 pattern: K=6 (predicate branch). W (128, 13) -> (768, 78)."""
    W = torch.randn(128, 13, dtype=torch.float64)
    W2, b2, K = _maybe_batched_block_diag(
        W, None, hz_dim=78, input_shape=(1, 6, 13), in_features=13,
    )
    assert K == 6
    assert W2.shape == (768, 78)
    assert b2 is None
    for k in range(6):
        block = W2[k*128:(k+1)*128, k*13:(k+1)*13]
        assert torch.equal(block, W)


def test_k2_join_branch_expansion():
    """L35 / L38 pattern: K=2 (join branch). W (128, 6) -> (256, 12)."""
    W = torch.randn(128, 6, dtype=torch.float64)
    W2, _, K = _maybe_batched_block_diag(
        W, None, hz_dim=12, input_shape=(1, 2, 6), in_features=6,
    )
    assert K == 2
    assert W2.shape == (256, 12)


def test_block_diag_matches_per_sample_matmul():
    """Apply W2 @ x_flat equals concat of W @ x_k for each K-block."""
    torch.manual_seed(0)
    W = torch.randn(5, 3, dtype=torch.float64)
    K_true = 4
    x_batched = torch.randn(K_true, 3, dtype=torch.float64)
    x_flat = x_batched.reshape(-1)
    y_ref = (x_batched @ W.t()).reshape(-1)
    W2, _, K_actual = _maybe_batched_block_diag(
        W, None, hz_dim=12, input_shape=(K_true, 3), in_features=3,
    )
    assert K_actual == K_true
    y_test = W2 @ x_flat
    assert y_test.shape == y_ref.shape
    assert torch.allclose(y_test, y_ref), (
        f"block-diag matmul disagrees with per-sample reference: "
        f"max_diff={(y_test - y_ref).abs().max().item():.3e}"
    )


def test_block_diag_with_bias_matches_per_sample():
    torch.manual_seed(1)
    W = torch.randn(5, 3, dtype=torch.float64)
    b = torch.randn(5, dtype=torch.float64)
    K_true = 4
    x_batched = torch.randn(K_true, 3, dtype=torch.float64)
    x_flat = x_batched.reshape(-1)
    y_ref = (x_batched @ W.t() + b).reshape(-1)
    W2, b2, _ = _maybe_batched_block_diag(
        W, b, hz_dim=12, input_shape=(K_true, 3), in_features=3,
    )
    y_test = W2 @ x_flat + b2
    assert torch.allclose(y_test, y_ref)


# ─── Negative cases (regression guards on the harden) ───────────────────


def test_divisible_dims_without_metadata_do_not_expand():
    """REGRESSION GUARD: a plain DENSE whose flat-input dim happens to
    equal `K * in_features` must NOT silently get expanded. This catches
    the earlier dim-divisibility heuristic if it ever creeps back."""
    # Suppose we have a DENSE with weight (128, 6). If a caller forgets
    # to pass input_shape and hz_dim happens to be 18 (e.g. from a
    # concat of three 6-dim sources), the legacy heuristic would have
    # incorrectly produced K=3. We must return K=1.
    W = torch.randn(128, 6, dtype=torch.float64)
    W2, _, K = _maybe_batched_block_diag(
        W, None, hz_dim=18, input_shape=None, in_features=6,
    )
    assert K == 1, f"divisible dims without input_shape must not expand (got K={K})"
    assert torch.equal(W2, W)


def test_input_shape_last_dim_mismatch_no_expand():
    """input_shape says (1, 3, 7) but in_features=6 → metadata inconsistent
    → fail-closed (K=1)."""
    W = torch.randn(128, 6, dtype=torch.float64)
    W2, _, K = _maybe_batched_block_diag(
        W, None, hz_dim=21, input_shape=(1, 3, 7), in_features=6,
    )
    assert K == 1


def test_prod_input_shape_mismatch_no_expand():
    """input_shape (1, 3, 6) implies prod=18 but hz_dim=24 — front-end
    drifted or there's an upstream reshape → fail-closed."""
    W = torch.randn(128, 6, dtype=torch.float64)
    W2, _, K = _maybe_batched_block_diag(
        W, None, hz_dim=24, input_shape=(1, 3, 6), in_features=6,
    )
    assert K == 1


def test_metadata_helper_directly():
    """Direct sanity on the metadata-only decision helper."""
    # Genuine batched
    assert _batched_matmul_K_from_metadata((1, 3, 6), 6, 18) == 3
    assert _batched_matmul_K_from_metadata((2, 5), 5, 10) == 2
    # Plain
    assert _batched_matmul_K_from_metadata((1, 6), 6, 6) == 1
    # Missing data
    assert _batched_matmul_K_from_metadata(None, 6, 18) == 1
    assert _batched_matmul_K_from_metadata((1, 3, 6), None, 18) == 1
    # Inconsistent
    assert _batched_matmul_K_from_metadata((1, 3, 6), 6, 24) == 1
    assert _batched_matmul_K_from_metadata((1, 3, 7), 6, 21) == 1


if __name__ == "__main__":
    tests = [
        test_no_metadata_never_expands,
        test_no_op_when_plain_dense_with_metadata,
        test_k3_sample_branch_expansion,
        test_k6_predicate_branch_expansion,
        test_k2_join_branch_expansion,
        test_block_diag_matches_per_sample_matmul,
        test_block_diag_with_bias_matches_per_sample,
        test_divisible_dims_without_metadata_do_not_expand,
        test_input_shape_last_dim_mismatch_no_expand,
        test_prod_input_shape_mismatch_no_expand,
        test_metadata_helper_directly,
    ]
    n_pass = n_fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            n_pass += 1
        except AssertionError as e:
            print(f"  FAIL  {t.__name__}: {e}")
            n_fail += 1
        except Exception as e:
            print(f"  ERR   {t.__name__}: {type(e).__name__}: {e}")
            n_fail += 1
    print(f"\nResult: {n_pass}/{len(tests)} passed")
    sys.exit(1 if n_fail else 0)
