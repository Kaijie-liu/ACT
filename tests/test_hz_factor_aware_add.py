"""Stage 3d-1b soundness tests for `hz_factor_aware_sum`.

The factor-aware ADD merges columns whose factor IDs match, treating
them as references to the SAME latent ξ. This is sound iff the
factor-ID semantics are respected. The two soundness-critical cases:

  1. SAME ID = SAME factor: math must collapse correctly (exact).
  2. DIFFERENT ID = DIFFERENT factor: must NOT be merged even when
     the column values happen to be identical (e.g. sibling branches
     A = ReLU₁(C), B = ReLU₂(C) where both ReLUs happen to add aux
     with the same numerical signature but different latent factors).

The advisor's invariant:
  "每个 HZ 内，同一个 factor_id 最多出现一次。如果出现重复，必须先
  exact-normalize：把同 ID 的 Gc columns 相加。"

Tests cover:
  - normalize_unique_ids: exact column-summation, deterministic order
  - factor_aware_sum: shared/only-x/only-y semantics
  - sibling-branch-distinct-aux: identical column VALUES with different
    IDs must NOT be merged (counter-example to a careless "value-equal
    = same factor" heuristic)
  - sample containment: explicit sample of the JOINT (xi_shared, alpha)
    space is contained in the factor-aware sum's set
  - precondition failures: ng_x > 0 with nc > 0 raises ValueError
"""
from __future__ import annotations

import sys
import torch

sys.path.insert(0, "/data1/Kane/ACT")

from act.back_end.solver.solver_hz import (
    HZono,
    hz_factor_aware_sum,
    hz_mixed_factor_aware_sum,
    hz_sparse_factor_aware_sum,
    _hz_factor_normalize_unique_ids,
)
from act.back_end.hybridz_tf.representations import SparseGcZ


def _hz_dense(c, Gc):
    n = c.shape[0]
    ng = Gc.shape[1]
    return HZono(
        c=c.clone(), Gc=Gc.clone(),
        Gb=torch.zeros(n, 0, dtype=c.dtype),
        Ac=torch.zeros(0, ng, dtype=c.dtype),
        Ab=torch.zeros(0, 0, dtype=c.dtype),
        b=torch.zeros(0, 1, dtype=c.dtype),
        eq_mask=None,
    )


def _hz_sparse(c, Gc):
    n, ng = Gc.shape
    idx = torch.nonzero(Gc != 0, as_tuple=False).t()
    if idx.numel() == 0:
        val = torch.zeros(0, dtype=Gc.dtype)
        idx = torch.zeros((2, 0), dtype=torch.long)
    else:
        val = Gc[idx[0], idx[1]]
    Gs = torch.sparse_coo_tensor(
        idx,
        val,
        (n, ng),
        dtype=Gc.dtype,
    ).coalesce()
    return SparseGcZ(c=c.flatten(), Gc_sparse=Gs, dtype=Gc.dtype, device=Gc.device)


# ─── normalize_unique_ids ─────────────────────────────────────────────────


def test_normalize_no_op_when_unique():
    c = torch.randn(3, 1, dtype=torch.float64)
    Gc = torch.randn(3, 4, dtype=torch.float64)
    ids = [10, 20, 30, 40]
    hz = _hz_dense(c, Gc)
    hz_out, ids_out = _hz_factor_normalize_unique_ids(hz, ids)
    assert ids_out == ids
    assert torch.equal(hz_out.Gc, Gc)


def test_normalize_sums_duplicate_columns():
    c = torch.randn(3, 1, dtype=torch.float64)
    Gc = torch.randn(3, 5, dtype=torch.float64)
    # ID 10 appears at cols 0 and 2; ID 20 at cols 1 and 4; ID 30 at col 3
    ids = [10, 20, 10, 30, 20]
    hz = _hz_dense(c, Gc)
    hz_out, ids_out = _hz_factor_normalize_unique_ids(hz, ids)
    # Deterministic: first-appearance order
    assert ids_out == [10, 20, 30]
    expected_col_10 = Gc[:, 0] + Gc[:, 2]
    expected_col_20 = Gc[:, 1] + Gc[:, 4]
    expected_col_30 = Gc[:, 3]
    assert torch.allclose(hz_out.Gc[:, 0], expected_col_10, rtol=1e-12)
    assert torch.allclose(hz_out.Gc[:, 1], expected_col_20, rtol=1e-12)
    assert torch.allclose(hz_out.Gc[:, 2], expected_col_30, rtol=1e-12)


def test_normalize_rejects_constraints():
    c = torch.randn(2, 1, dtype=torch.float64)
    Gc = torch.randn(2, 2, dtype=torch.float64)
    Ac = torch.randn(1, 2, dtype=torch.float64)  # nc > 0
    hz = HZono(c=c, Gc=Gc,
               Gb=torch.zeros(2, 0, dtype=torch.float64),
               Ac=Ac, Ab=torch.zeros(1, 0, dtype=torch.float64),
               b=torch.zeros(1, 1, dtype=torch.float64),
               eq_mask=None)
    # Should raise if duplicates require normalization
    try:
        _hz_factor_normalize_unique_ids(hz, [10, 10])
    except ValueError as e:
        assert "nc=" in str(e) or "Ac" in str(e)
    else:
        raise AssertionError("expected ValueError on nc>0 with duplicates")


def test_normalize_no_op_with_constraints_when_unique():
    """If no duplicates, constraints are passed through (no normalize needed)."""
    c = torch.randn(2, 1, dtype=torch.float64)
    Gc = torch.randn(2, 2, dtype=torch.float64)
    Ac = torch.randn(1, 2, dtype=torch.float64)
    hz = HZono(c=c, Gc=Gc,
               Gb=torch.zeros(2, 0, dtype=torch.float64),
               Ac=Ac, Ab=torch.zeros(1, 0, dtype=torch.float64),
               b=torch.zeros(1, 1, dtype=torch.float64),
               eq_mask=None)
    hz_out, ids_out = _hz_factor_normalize_unique_ids(hz, [10, 20])
    assert ids_out == [10, 20]
    assert torch.equal(hz_out.Gc, Gc)


# ─── factor_aware_sum ─────────────────────────────────────────────────────


def test_factor_aware_sum_shared_factors_collapse():
    """If both operands have the SAME factor IDs, the result is
    column-wise addition with NO duplication."""
    torch.manual_seed(0)
    n, ng = 3, 4
    G = torch.randn(n, ng, dtype=torch.float64)
    c1 = torch.randn(n, 1, dtype=torch.float64)
    c2 = torch.randn(n, 1, dtype=torch.float64)
    hz1 = _hz_dense(c1, G)
    hz2 = _hz_dense(c2, G)
    ids = [100, 200, 300, 400]
    hz_out, ids_out = hz_factor_aware_sum(hz1, hz2, ids, ids)
    assert ids_out == ids
    assert torch.allclose(hz_out.Gc, 2 * G, rtol=1e-12)
    assert torch.allclose(hz_out.c, c1 + c2, rtol=1e-12)


def test_factor_aware_sum_only_y_appended():
    """IDs only in y get appended in y's order; shared IDs sum."""
    n = 2
    c1 = torch.zeros(n, 1, dtype=torch.float64)
    c2 = torch.zeros(n, 1, dtype=torch.float64)
    G1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    G2 = torch.tensor([[7., 8.], [9., 10.]], dtype=torch.float64)
    hz1 = _hz_dense(c1, G1)
    hz2 = _hz_dense(c2, G2)
    # ids1 = [A, B, C]; ids2 = [B, D]  -> output [A, B, C, D]
    hz_out, ids_out = hz_factor_aware_sum(hz1, hz2, [11, 22, 33], [22, 44])
    assert ids_out == [11, 22, 33, 44]
    # Col A: from x only (= G1[:, 0])
    assert torch.allclose(hz_out.Gc[:, 0], G1[:, 0])
    # Col B: x + y (G1[:, 1] + G2[:, 0])
    assert torch.allclose(hz_out.Gc[:, 1], G1[:, 1] + G2[:, 0])
    # Col C: x only
    assert torch.allclose(hz_out.Gc[:, 2], G1[:, 2])
    # Col D: y only
    assert torch.allclose(hz_out.Gc[:, 3], G2[:, 1])


def test_factor_aware_sum_normalizes_duplicates_first():
    """When an operand has self-duplicates (same ID twice), they're
    summed BEFORE merging across operands. Verifies the advisor's
    invariant."""
    n = 2
    c1 = torch.zeros(n, 1, dtype=torch.float64)
    c2 = torch.zeros(n, 1, dtype=torch.float64)
    # ids1 = [A, B, A] (A duplicates inside)
    G1 = torch.tensor([[1., 2., 3.], [4., 5., 6.]], dtype=torch.float64)
    G2 = torch.tensor([[7.], [8.]], dtype=torch.float64)
    hz1 = _hz_dense(c1, G1)
    hz2 = _hz_dense(c2, G2)
    hz_out, ids_out = hz_factor_aware_sum(hz1, hz2, [11, 22, 11], [22])
    assert ids_out == [11, 22]
    # After normalize: hz1's A col = G1[:, 0] + G1[:, 2] = (4, 10)
    # Then add hz2's B (no A in y), so col A = (4, 10).
    expected_A = G1[:, 0] + G1[:, 2]
    assert torch.allclose(hz_out.Gc[:, 0], expected_A)
    # Col B: hz1's B + hz2's B
    expected_B = G1[:, 1] + G2[:, 0]
    assert torch.allclose(hz_out.Gc[:, 1], expected_B)


def test_sibling_branch_distinct_aux_must_NOT_merge():
    """SOUNDNESS GUARD. Two HZs derived from a common ancestor by
    SEPARATE ReLUs introduce aux columns whose VALUES might coincide
    by accident, but whose latent factors are DIFFERENT. Factor-
    aware sum must NOT merge them based on column values — only on
    factor IDs.

    Scenario:
      branch_A_ReLU adds an aux column [1, 0]^T with fresh ID 1000.
      branch_B_ReLU adds an aux column [1, 0]^T with fresh ID 2000.
    Numerically identical, but DIFFERENT factors.

    Block-diag (or factor-aware with separate IDs) keeps them
    distinct → 2 cols. Misguided "value-equal = same factor" would
    sum them → 1 col → UNSOUND.
    """
    n = 2
    c1 = torch.zeros(n, 1, dtype=torch.float64)
    c2 = torch.zeros(n, 1, dtype=torch.float64)
    G1 = torch.tensor([[1.], [0.]], dtype=torch.float64)
    G2 = torch.tensor([[1.], [0.]], dtype=torch.float64)
    hz1 = _hz_dense(c1, G1)
    hz2 = _hz_dense(c2, G2)
    # Different IDs even though values are equal
    hz_out, ids_out = hz_factor_aware_sum(hz1, hz2, [1000], [2000])
    assert ids_out == [1000, 2000]
    assert hz_out.Gc.shape[1] == 2, (
        "distinct factor IDs must remain in separate columns even if "
        "their numerical values are identical"
    )
    # Output set must equal {(x + y, 0) : x ∈ [-1,1], y ∈ [-1,1]}
    # which has range [-2, 2] on row 0. If we wrongly merged, range
    # would be [-1, 1] — strictly smaller (unsound).
    rad0 = float(hz_out.Gc[0].abs().sum())
    assert abs(rad0 - 2.0) < 1e-12, (
        f"expected row-0 radius 2.0 (unmerged), got {rad0}"
    )


def test_sample_containment_in_factor_aware_sum():
    """Joint sample (xi_shared, alpha_branch) is contained in the
    factor-aware sum's set."""
    torch.manual_seed(1)
    n = 3
    root_ng = 4
    aux_ng = 2
    G_root = torch.randn(n, root_ng, dtype=torch.float64)
    G_aux_x = torch.randn(n, 1, dtype=torch.float64)  # x has 1 aux
    G_aux_y = torch.randn(n, 1, dtype=torch.float64)  # y has 1 aux (DIFFERENT id)
    c1 = torch.randn(n, 1, dtype=torch.float64)
    c2 = torch.randn(n, 1, dtype=torch.float64)
    hz1 = _hz_dense(c1, torch.cat([G_root, G_aux_x], dim=1))
    hz2 = _hz_dense(c2, torch.cat([G_root, G_aux_y], dim=1))
    # ids: x = [0,1,2,3, 100], y = [0,1,2,3, 200]
    ids1 = [0, 1, 2, 3, 100]
    ids2 = [0, 1, 2, 3, 200]
    hz_out, ids_out = hz_factor_aware_sum(hz1, hz2, ids1, ids2)
    assert ids_out == [0, 1, 2, 3, 100, 200]
    # Sample: pick (xi_root, alpha_x, alpha_y) and verify membership
    torch.manual_seed(42)
    n_samples = 50
    for _ in range(n_samples):
        xi_root = torch.empty(root_ng).uniform_(-1, 1).to(torch.float64)
        alpha_x = float(torch.empty(1).uniform_(-1, 1))
        alpha_y = float(torch.empty(1).uniform_(-1, 1))
        # True value
        x_val = hz1.c.squeeze(-1) + hz1.Gc @ torch.cat([xi_root, torch.tensor([alpha_x], dtype=torch.float64)])
        y_val = hz2.c.squeeze(-1) + hz2.Gc @ torch.cat([xi_root, torch.tensor([alpha_y], dtype=torch.float64)])
        true_sum = x_val + y_val
        # Recover from hz_out: pick the same xi_root for shared, alpha_x for col 100, alpha_y for col 200
        eta = torch.tensor([xi_root[0], xi_root[1], xi_root[2], xi_root[3], alpha_x, alpha_y], dtype=torch.float64)
        approx_sum = hz_out.c.squeeze(-1) + hz_out.Gc @ eta
        diff = (true_sum - approx_sum).abs().max().item()
        assert diff < 1e-9, f"sample mismatch {diff:.3e}"


def test_preconditions_constraints_rejected():
    """ValueError if nc > 0 on either operand."""
    n = 2
    c1 = torch.zeros(n, 1, dtype=torch.float64)
    c2 = torch.zeros(n, 1, dtype=torch.float64)
    Gc1 = torch.randn(n, 2, dtype=torch.float64)
    Gc2 = torch.randn(n, 2, dtype=torch.float64)
    hz1 = HZono(c=c1, Gc=Gc1,
                Gb=torch.zeros(n, 0, dtype=torch.float64),
                Ac=torch.randn(1, 2, dtype=torch.float64),  # nc=1
                Ab=torch.zeros(1, 0, dtype=torch.float64),
                b=torch.zeros(1, 1, dtype=torch.float64),
                eq_mask=None)
    hz2 = _hz_dense(c2, Gc2)
    try:
        hz_factor_aware_sum(hz1, hz2, [10, 11], [10, 11])
    except ValueError as e:
        assert "nc" in str(e)
    else:
        raise AssertionError("expected ValueError on nc > 0")


def test_preconditions_binary_generators_rejected():
    """ValueError if nb > 0 on either operand."""
    n = 2
    c1 = torch.zeros(n, 1, dtype=torch.float64)
    c2 = torch.zeros(n, 1, dtype=torch.float64)
    Gc1 = torch.randn(n, 2, dtype=torch.float64)
    Gc2 = torch.randn(n, 2, dtype=torch.float64)
    hz1 = HZono(c=c1, Gc=Gc1,
                Gb=torch.randn(n, 1, dtype=torch.float64),  # nb=1
                Ac=torch.zeros(0, 2, dtype=torch.float64),
                Ab=torch.zeros(0, 1, dtype=torch.float64),
                b=torch.zeros(0, 1, dtype=torch.float64),
                eq_mask=None)
    hz2 = _hz_dense(c2, Gc2)
    try:
        hz_factor_aware_sum(hz1, hz2, [10, 11], [10, 11])
    except ValueError as e:
        assert "nb" in str(e)
    else:
        raise AssertionError("expected ValueError on nb > 0")


def test_preconditions_id_length_mismatch_rejected():
    """ValueError if len(ids) != ng."""
    n = 2
    c = torch.zeros(n, 1, dtype=torch.float64)
    G = torch.randn(n, 3, dtype=torch.float64)
    hz = _hz_dense(c, G)
    try:
        hz_factor_aware_sum(hz, hz, [10, 11], [10, 11, 12])  # length wrong on x
    except ValueError as e:
        assert "ids_x" in str(e) or "ids_y" in str(e)
    else:
        raise AssertionError("expected ValueError on id-length mismatch")


def test_sparse_factor_aware_sum_matches_dense_result():
    """SparseGcZ factor-aware ADD is the same exact operation as dense
    HZono factor-aware ADD after densification."""
    n = 4
    c1 = torch.randn(n, 1, dtype=torch.float64)
    c2 = torch.randn(n, 1, dtype=torch.float64)
    G1 = torch.tensor(
        [[1., 0., 2.],
         [0., 3., 0.],
         [4., 0., 0.],
         [0., 0., 5.]],
        dtype=torch.float64,
    )
    G2 = torch.tensor(
        [[0., 7.],
         [8., 0.],
         [0., 0.],
         [9., 10.]],
        dtype=torch.float64,
    )
    ids1 = [11, 22, 33]
    ids2 = [22, 44]
    dense_out, dense_ids = hz_factor_aware_sum(
        _hz_dense(c1, G1), _hz_dense(c2, G2), ids1, ids2
    )
    sparse_out, sparse_ids = hz_sparse_factor_aware_sum(
        _hz_sparse(c1, G1), _hz_sparse(c2, G2), ids1, ids2
    )
    assert sparse_ids == dense_ids
    assert torch.allclose(sparse_out.c.view(-1, 1), dense_out.c, rtol=1e-12)
    assert torch.allclose(
        sparse_out.Gc_sparse.to_dense(), dense_out.Gc, rtol=1e-12
    )


def test_sparse_distinct_ids_not_merged():
    """Same numerical sparse column values with different factor IDs must
    stay independent."""
    c = torch.zeros(2, 1, dtype=torch.float64)
    G = torch.tensor([[1.], [0.]], dtype=torch.float64)
    out, ids = hz_sparse_factor_aware_sum(
        _hz_sparse(c, G),
        _hz_sparse(c, G),
        [1000],
        [2000],
    )
    assert ids == [1000, 2000]
    assert out.ng == 2
    assert abs(float(out.Gc_sparse.to_dense()[0].abs().sum()) - 2.0) < 1e-12


def test_mixed_factor_aware_sum_matches_dense_result_both_orders():
    n = 3
    c1 = torch.randn(n, 1, dtype=torch.float64)
    c2 = torch.randn(n, 1, dtype=torch.float64)
    G1 = torch.tensor(
        [[1., 0., 2.],
         [0., 3., 0.],
         [4., 0., 5.]],
        dtype=torch.float64,
    )
    G2 = torch.tensor(
        [[6., 0.],
         [0., 7.],
         [8., 9.]],
        dtype=torch.float64,
    )
    ids1 = [11, 22, 33]
    ids2 = [22, 44]
    dense_out, dense_ids = hz_factor_aware_sum(
        _hz_dense(c1, G1), _hz_dense(c2, G2), ids1, ids2
    )
    mixed_out, mixed_ids = hz_mixed_factor_aware_sum(
        _hz_dense(c1, G1), _hz_sparse(c2, G2), ids1, ids2
    )
    assert mixed_ids == dense_ids
    assert torch.allclose(mixed_out.c, dense_out.c, rtol=1e-12)
    assert torch.allclose(mixed_out.Gc, dense_out.Gc, rtol=1e-12)

    dense_rev, dense_rev_ids = hz_factor_aware_sum(
        _hz_dense(c2, G2), _hz_dense(c1, G1), ids2, ids1
    )
    mixed_rev, mixed_rev_ids = hz_mixed_factor_aware_sum(
        _hz_sparse(c2, G2), _hz_dense(c1, G1), ids2, ids1
    )
    assert mixed_rev_ids == dense_rev_ids
    assert torch.allclose(mixed_rev.c, dense_rev.c, rtol=1e-12)
    assert torch.allclose(mixed_rev.Gc, dense_rev.Gc, rtol=1e-12)


if __name__ == "__main__":
    tests = [
        test_normalize_no_op_when_unique,
        test_normalize_sums_duplicate_columns,
        test_normalize_rejects_constraints,
        test_normalize_no_op_with_constraints_when_unique,
        test_factor_aware_sum_shared_factors_collapse,
        test_factor_aware_sum_only_y_appended,
        test_factor_aware_sum_normalizes_duplicates_first,
        test_sibling_branch_distinct_aux_must_NOT_merge,
        test_sample_containment_in_factor_aware_sum,
        test_preconditions_constraints_rejected,
        test_preconditions_binary_generators_rejected,
        test_preconditions_id_length_mismatch_rejected,
        test_sparse_factor_aware_sum_matches_dense_result,
        test_sparse_distinct_ids_not_merged,
        test_mixed_factor_aware_sum_matches_dense_result_both_orders,
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
