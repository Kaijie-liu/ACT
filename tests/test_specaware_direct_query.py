"""Soundness + correctness tests for `SpecAwareLP.verify_one_disjunct_from_arrays`
— the direct-disjunct entry that lets ACT skip the O(N²) cli × SpecAware
outer-loop on multi-disjunct VNNLib specs.

The wrapper MUST satisfy:

  1. **Single-disjunct VNNLib equivalence**: when the VNNLib has
     exactly ONE disjunct, `verify_one_disjunct_from_arrays(...)` must
     return the same verdict as `verify(onnx_path, vnnlib_path)` —
     both for `verified` (LP-infeasible spec) and `unknown` (margin
     not closed).
  2. **Two-disjunct same input box**: calling
     `verify_one_disjunct_from_arrays` with disjunct 0 must NOT
     iterate over disjunct 1. Verdict depends ONLY on the disjunct
     passed in. (No silent OR-aggregation.)
  3. **Fail-closed on unbounded input**: infinite lb / ub must return
     `fail(unbounded_input)`, not 'verified'.
  4. **Layer-extract cache hit on repeated calls**: same onnx_path →
     `_LAYER_EXTRACT_CACHE` size stays at 1. Guards against
     accidentally bypassing the cache via Path-vs-str key mismatch.
  5. **Bound cache + direct-query compose**: cache writes for the
     direct path use the same key as `verify()` would, so a direct
     call followed by a `verify()` call with same (model + input box)
     hits cache.
  6. **Stable-affine fast path is conservative**: when every ReLU is
     interval-stable and one unsafe row is individually impossible, it
     may certify; if any ReLU is unstable it must decline and let the
     LP path handle the query.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
import tempfile

import numpy as np

sys.path.insert(0, "/data1/Kane/HyZor")

import SpecAwareLP as sa


def _make_3layer_relu_model_onnx(tmpdir, seed: int = 0):
    """Write a tiny 3-layer ReLU ONNX (2 → 4 → 4 → 2) so SpecAwareLP can
    consume it. Uses ONNX Runtime layer for sanity but the actual file is
    what SpecAwareLP.extract_layers parses."""
    import onnx
    from onnx import helper, TensorProto, numpy_helper
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(2, 4)).astype(np.float32)
    b0 = rng.normal(size=(4,)).astype(np.float32)
    W1 = rng.normal(size=(4, 4)).astype(np.float32)
    b1 = rng.normal(size=(4,)).astype(np.float32)
    W2 = rng.normal(size=(4, 2)).astype(np.float32)
    b2 = rng.normal(size=(2,)).astype(np.float32)
    inits = [
        numpy_helper.from_array(W0, "W0"), numpy_helper.from_array(b0, "b0"),
        numpy_helper.from_array(W1, "W1"), numpy_helper.from_array(b1, "b1"),
        numpy_helper.from_array(W2, "W2"), numpy_helper.from_array(b2, "b2"),
    ]
    nodes = [
        helper.make_node("MatMul", ["X", "W0"], ["t0"]),
        helper.make_node("Add", ["t0", "b0"], ["a0"]),
        helper.make_node("Relu", ["a0"], ["r0"]),
        helper.make_node("MatMul", ["r0", "W1"], ["t1"]),
        helper.make_node("Add", ["t1", "b1"], ["a1"]),
        helper.make_node("Relu", ["a1"], ["r1"]),
        helper.make_node("MatMul", ["r1", "W2"], ["t2"]),
        helper.make_node("Add", ["t2", "b2"], ["Y"]),
    ]
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 2])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 2])
    graph = helper.make_graph(nodes, "g", [X], [Y], initializer=inits)
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 13)]
    )
    model.ir_version = 7
    onnx.checker.check_model(model)
    path = Path(tmpdir) / "tiny_3layer.onnx"
    onnx.save(model, str(path))
    return path


def _write_vnnlib_single_disjunct(tmpdir, lb_x, ub_x, axis, d_val):
    """Write a VNNLib with EXACTLY ONE disjunct: a single Y halfspace
    `Y_axis ≤ d`. Uses the canonical `(<= Y_i const)` form that
    `GlobalTriangleLP.parse_vnnlib` recognizes."""
    path = Path(tmpdir) / "single.vnnlib"
    with open(path, "w") as f:
        f.write("(declare-const X_0 Real)\n(declare-const X_1 Real)\n")
        f.write("(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n")
        f.write(f"(assert (>= X_0 {lb_x[0]}))\n")
        f.write(f"(assert (<= X_0 {ub_x[0]}))\n")
        f.write(f"(assert (>= X_1 {lb_x[1]}))\n")
        f.write(f"(assert (<= X_1 {ub_x[1]}))\n")
        f.write(f"(assert (<= Y_{axis} {d_val}))\n")
    return path


def _write_vnnlib_two_disjunct(tmpdir, lb_x, ub_x, specs):
    """Write a VNNLib with TWO disjuncts sharing the input box.
    Each entry of `specs` is (axis, d) → `Y_axis ≤ d` in that disjunct.
    """
    path = Path(tmpdir) / "two.vnnlib"
    with open(path, "w") as f:
        f.write("(declare-const X_0 Real)\n(declare-const X_1 Real)\n")
        f.write("(declare-const Y_0 Real)\n(declare-const Y_1 Real)\n")
        f.write(f"(assert (>= X_0 {lb_x[0]}))\n")
        f.write(f"(assert (<= X_0 {ub_x[0]}))\n")
        f.write(f"(assert (>= X_1 {lb_x[1]}))\n")
        f.write(f"(assert (<= X_1 {ub_x[1]}))\n")
        f.write("(assert (or\n")
        for axis, d_val in specs:
            f.write(f"  (and (<= Y_{axis} {d_val}))\n")
        f.write("))\n")
    return path


def _baseline_verify(onnx_path, vnnlib_path):
    """Wrapper around `verify()` returning the verdict only."""
    v, _ = sa.verify(
        onnx_path, vnnlib_path,
        time_limit_per_lp=5.0, max_refinement_passes=0,
    )
    return v


# ─── Test 1: single-disjunct equivalence ────────────────────────────────


def test_single_disjunct_equivalence():
    with tempfile.TemporaryDirectory() as td:
        onnx = _make_3layer_relu_model_onnx(td, seed=0)
        lb_x = np.array([-0.2, -0.2], dtype=np.float64)
        ub_x = np.array([0.2, 0.2], dtype=np.float64)
        # Loose: Y_0 ≤ 1e9 → LP feasible → both 'unknown'
        c_loose = np.array([1.0, 0.0], dtype=np.float64); d_loose = 1e9
        vnn = _write_vnnlib_single_disjunct(td, lb_x, ub_x, 0, d_loose)

        sa._layer_extract_cache_reset()
        sa._bound_cache_reset()
        v_full = _baseline_verify(onnx, vnn)
        v_direct, _ = sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [(c_loose, d_loose)],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        assert v_full == v_direct, (
            f"verdict mismatch: full={v_full!r} direct={v_direct!r}"
        )

        # Tight: Y_1 ≤ -1e9 → LP infeasible → both 'verified'
        c_tight = np.array([0.0, 1.0], dtype=np.float64); d_tight = -1e9
        vnn_t = _write_vnnlib_single_disjunct(td, lb_x, ub_x, 1, d_tight)
        v_full_t = _baseline_verify(onnx, vnn_t)
        v_direct_t, _ = sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [(c_tight, d_tight)],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        assert v_full_t == v_direct_t, (
            f"verdict mismatch (tight): full={v_full_t!r} direct={v_direct_t!r}"
        )


# ─── Test 2: two-disjunct same box; direct-call sees only one ──────────


def test_two_disjunct_same_box_direct_isolates_one():
    """With TWO disjuncts on the same input box, the direct entry must
    return the verdict for ONLY the disjunct passed in. The full
    verify() iterates both and returns 'unknown' if either is unknown."""
    with tempfile.TemporaryDirectory() as td:
        onnx = _make_3layer_relu_model_onnx(td, seed=1)
        lb_x = np.array([-0.2, -0.2], dtype=np.float64)
        ub_x = np.array([0.2, 0.2], dtype=np.float64)
        # Two disjuncts on the same input box:
        #   tight: Y_1 ≤ -1e9 (infeasible → 'verified')
        #   loose: Y_0 ≤  1e9 (always feasible → 'unknown')
        spec_tight_c = (np.array([0.0, 1.0], dtype=np.float64), -1e9)
        spec_loose_c = (np.array([1.0, 0.0], dtype=np.float64),  1e9)
        vnn = _write_vnnlib_two_disjunct(
            td, lb_x, ub_x, [(1, -1e9), (0, 1e9)]
        )

        sa._layer_extract_cache_reset()
        sa._bound_cache_reset()
        # full verify() iterates both → at least one is unknown → unknown
        v_full = _baseline_verify(onnx, vnn)
        # direct on tight disjunct → should be 'verified' (independent of loose)
        v_direct_tight, _ = sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [spec_tight_c],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        # direct on loose disjunct → should be 'unknown'
        v_direct_loose, _ = sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [spec_loose_c],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        assert v_direct_tight == 'verified', (
            f"direct-tight should verify, got {v_direct_tight!r}"
        )
        assert v_direct_loose == 'unknown', (
            f"direct-loose should be unknown, got {v_direct_loose!r}"
        )
        # full verify() should match: ANY unknown → unknown
        assert v_full == 'unknown', (
            f"full verify() should be unknown (loose disjunct), got {v_full!r}"
        )


# ─── Test 3: fail-closed on unbounded input ─────────────────────────────


def test_unbounded_input_fails_closed():
    with tempfile.TemporaryDirectory() as td:
        onnx = _make_3layer_relu_model_onnx(td, seed=2)
        lb_x = np.array([-np.inf, -0.2], dtype=np.float64)
        ub_x = np.array([0.2, 0.2], dtype=np.float64)
        c_vec = np.array([1.0, 0.0], dtype=np.float64)
        d_val = 0.0
        sa._layer_extract_cache_reset()
        v_direct, _ = sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [(c_vec, d_val)],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        assert v_direct.startswith('fail'), (
            f"unbounded input must fail-closed, got {v_direct!r}"
        )
        assert 'unbounded' in v_direct, (
            f"fail message should mention 'unbounded', got {v_direct!r}"
        )


# ─── Test 4: layer-extract cache hit on repeated direct calls ──────────


def test_layer_extract_cache_hit_on_repeated_calls():
    with tempfile.TemporaryDirectory() as td:
        onnx = _make_3layer_relu_model_onnx(td, seed=3)
        lb_x = np.array([-0.1, -0.1], dtype=np.float64)
        ub_x = np.array([0.1, 0.1], dtype=np.float64)
        c_vec = np.array([1.0, 0.0], dtype=np.float64)
        d_val = 1e9
        sa._layer_extract_cache_reset()
        # First call populates cache.
        sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [(c_vec, d_val)],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        assert len(sa._LAYER_EXTRACT_CACHE) == 1, (
            f"layer-extract cache should have 1 entry, got "
            f"{len(sa._LAYER_EXTRACT_CACHE)}"
        )
        # Second call must hit the cache, not grow it.
        sa.verify_one_disjunct_from_arrays(
            onnx, lb_x, ub_x, [(c_vec, d_val)],
            time_limit_per_lp=5.0, max_refinement_passes=0,
        )
        assert len(sa._LAYER_EXTRACT_CACHE) == 1, (
            f"cache should still have 1 entry after repeat, got "
            f"{len(sa._LAYER_EXTRACT_CACHE)}"
        )


# ─── Test 5: bound-cache composes with direct path ─────────────────────


def test_bound_cache_composes_with_direct_path():
    """A direct-query call on (model, lb_x, ub_x) writes the SAME cache
    entry that verify() would (same key). A subsequent direct call on
    the same box hits cache (skips the per-neuron bound LP solves)."""
    with tempfile.TemporaryDirectory() as td:
        onnx = _make_3layer_relu_model_onnx(td, seed=4)
        lb_x = np.array([-0.15, -0.15], dtype=np.float64)
        ub_x = np.array([0.15, 0.15], dtype=np.float64)
        spec_a = (np.array([1.0, 0.0], dtype=np.float64), 1e9)
        spec_b = (np.array([0.0, 1.0], dtype=np.float64), 1e9)

        os.environ["ACT_HZ_SPECAWARE_BOUND_CACHE"] = "1"
        try:
            sa._bound_cache_reset()
            sa._layer_extract_cache_reset()
            # First call: miss + write
            sa.verify_one_disjunct_from_arrays(
                onnx, lb_x, ub_x, [spec_a],
                time_limit_per_lp=5.0, max_refinement_passes=0,
            )
            s1 = sa._bound_cache_stats()
            assert s1["writes"] == 1 and s1["hits"] == 0
            # Second call same box different spec: hit
            sa.verify_one_disjunct_from_arrays(
                onnx, lb_x, ub_x, [spec_b],
                time_limit_per_lp=5.0, max_refinement_passes=0,
            )
            s2 = sa._bound_cache_stats()
            assert s2["hits"] == 1, (
                f"second direct call should hit bound cache, got {s2}"
            )
        finally:
            os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)


def test_stable_affine_fastpath_conservative():
    sa._stable_affine_cache_reset()
    # All ReLUs active on [-0.1, 0.1]^2 because pre = x + 2.
    layers = [
        (
            np.eye(2, dtype=np.float64),
            np.array([2.0, 2.0], dtype=np.float64),
        )
    ]
    output_layer = (
        np.array([[1.0], [0.0]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )
    lb_x = np.array([-0.1, -0.1], dtype=np.float64)
    ub_x = np.array([0.1, 0.1], dtype=np.float64)
    # y = x0 + 2, so min y = 1.9 > 1.0. Unsafe row y <= 1 is impossible.
    assert sa._try_stable_affine_verified(
        None, layers, output_layer, lb_x, ub_x,
        [(np.array([1.0], dtype=np.float64), 1.0)]
    )

    # If the same ReLU can cross zero, the fast path must decline.
    unstable_layers = [
        (
            np.eye(2, dtype=np.float64),
            np.array([0.0, 0.0], dtype=np.float64),
        )
    ]
    assert not sa._try_stable_affine_verified(
        None, unstable_layers, output_layer,
        np.array([-1.0, -1.0], dtype=np.float64),
        np.array([1.0, 1.0], dtype=np.float64),
        [(np.array([1.0], dtype=np.float64), 1.0)]
    )


def test_stable_affine_cache_reuses_same_box_across_rows():
    sa._stable_affine_cache_reset()
    layers = [
        (
            np.eye(2, dtype=np.float64),
            np.array([2.0, 2.0], dtype=np.float64),
        )
    ]
    output_layer = (
        np.array([[1.0], [0.0]], dtype=np.float64),
        np.array([0.0], dtype=np.float64),
    )
    lb_x = np.array([-0.1, -0.1], dtype=np.float64)
    ub_x = np.array([0.1, 0.1], dtype=np.float64)
    assert sa._try_stable_affine_verified(
        None, layers, output_layer, lb_x, ub_x,
        [(np.array([1.0], dtype=np.float64), 1.0)]
    )
    s1 = sa._stable_affine_cache_stats()
    assert s1["writes_affine"] == 1 and s1["hits"] == 0

    assert sa._try_stable_affine_verified(
        None, layers, output_layer, lb_x, ub_x,
        [(np.array([1.0], dtype=np.float64), 0.0)]
    )
    s2 = sa._stable_affine_cache_stats()
    assert s2["hits"] == 1 and s2["writes_affine"] == 1, s2


if __name__ == "__main__":
    tests = [
        test_single_disjunct_equivalence,
        test_two_disjunct_same_box_direct_isolates_one,
        test_unbounded_input_fails_closed,
        test_layer_extract_cache_hit_on_repeated_calls,
        test_bound_cache_composes_with_direct_path,
        test_stable_affine_fastpath_conservative,
        test_stable_affine_cache_reuses_same_box_across_rows,
    ]
    n_pass = 0; n_fail = 0
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
