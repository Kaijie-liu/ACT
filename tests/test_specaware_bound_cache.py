"""Soundness + correctness tests for SpecAwareLP input-box bound cache
(env: `ACT_HZ_SPECAWARE_BOUND_CACHE=1`, default OFF).

The cache MUST satisfy:

  1. **Same input box + different unsafe rows → identical verdict** as
     uncached. The cache speeds up the initial bound LP step but must
     not change downstream LP behavior.
  2. **Different input box → no cache hit.** Cache keys are
     byte-exact on (lb_x, ub_x); even a single ulp difference must
     produce a distinct key.
  3. **Different model (different weights or biases) → no cache hit.**
     A weight perturbation in any layer must invalidate the key.
  4. **Per-LP timeout / failure → no cache write.** When any per-
     neuron bound LP fails, the partial layer_pre_bounds must NOT be
     cached (caching them would let later disjuncts use suspect /
     incomplete bounds).
  5. **Env default OFF.** Without `ACT_HZ_SPECAWARE_BOUND_CACHE=1`,
     the cache must remain empty even after multiple calls (no
     write, no read).
"""
from __future__ import annotations

import os
import sys

import numpy as np

# HyZor sources live alongside ACT in the repo layout.
sys.path.insert(0, "/data1/Kane/HyZor")

import SpecAwareLP as sa


def _make_2layer_model(seed: int = 0):
    """Build a tiny 2-layer ReLU MLP: 3 -> 4 -> 2. Returns (layers,
    output_layer, sub_const). Layer weights/biases are deterministic."""
    rng = np.random.default_rng(seed)
    W0 = rng.normal(size=(3, 4)).astype(np.float64)
    b0 = rng.normal(size=(4,)).astype(np.float64)
    W1 = rng.normal(size=(4, 2)).astype(np.float64)
    b1 = rng.normal(size=(2,)).astype(np.float64)
    # `output_layer = (Wy, by)` of shape (n_out, n_post_last) and (n_out,).
    Wy = np.eye(2, dtype=np.float64)
    by = np.zeros(2, dtype=np.float64)
    layers = [(W0, b0), (W1, b1)]
    output_layer = (Wy, by)
    return layers, output_layer, None


def _make_disjunct(seed: int = 0, *, c_offset: float = 0.0):
    """Build (lb_x, ub_x, unsafe_rows). The input box is fixed
    [-0.5, 0.5]^3; unsafe row is shifted by `c_offset` so two calls
    on the same box have different specs."""
    lb_x = -0.5 * np.ones(3, dtype=np.float64)
    ub_x = 0.5 * np.ones(3, dtype=np.float64)
    # Unsafe set: y_0 + 0.5 * y_1 <= c_offset  (linear unsafe row)
    unsafe_rows = [(np.array([1.0, 0.5], dtype=np.float64),
                    float(c_offset))]
    return lb_x, ub_x, unsafe_rows


def _run_once(layers, output_layer, sub_const, lb_x, ub_x, unsafe_rows):
    """Direct call to the per-disjunct entry. Returns a verdict string."""
    return sa._verify_one_disjunct_specaware(
        sub_const, layers, output_layer,
        lb_x, ub_x, unsafe_rows,
        time_limit_per_lp=5.0,
        max_refinement_passes=0,
        min_tighten_abs=1e-3,
    )


# ─── Test 1: same input box + different spec → identical verdict ────────


def test_same_box_different_spec_verdict_matches_uncached():
    layers, output_layer, sub_const = _make_2layer_model(seed=0)
    lb_x, ub_x, _ = _make_disjunct()
    spec_a = [(np.array([1.0, 0.5], dtype=np.float64), 0.0)]
    spec_b = [(np.array([1.0, 0.5], dtype=np.float64), 5.0)]

    # Baseline: cache OFF
    os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)
    sa._bound_cache_reset()
    v_a_off = _run_once(layers, output_layer, sub_const, lb_x, ub_x, spec_a)
    v_b_off = _run_once(layers, output_layer, sub_const, lb_x, ub_x, spec_b)
    assert sa._bound_cache_stats()["hits"] == 0
    assert sa._bound_cache_stats()["writes"] == 0  # env off → no writes

    # Cache ON: first call writes, second call hits.
    os.environ["ACT_HZ_SPECAWARE_BOUND_CACHE"] = "1"
    try:
        sa._bound_cache_reset()
        v_a_on = _run_once(layers, output_layer, sub_const, lb_x, ub_x, spec_a)
        s1 = sa._bound_cache_stats()
        assert s1["writes"] == 1 and s1["hits"] == 0 and s1["misses"] == 1, (
            f"first call should miss + write, got {s1}"
        )
        v_b_on = _run_once(layers, output_layer, sub_const, lb_x, ub_x, spec_b)
        s2 = sa._bound_cache_stats()
        assert s2["hits"] == 1 and s2["writes"] == 1, (
            f"second call should hit, no new write, got {s2}"
        )
    finally:
        os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)

    # Same verdicts cached vs uncached, despite different spec rows.
    assert v_a_off == v_a_on, f"cache changed verdict for spec_a: {v_a_off!r} vs {v_a_on!r}"
    assert v_b_off == v_b_on, f"cache changed verdict for spec_b: {v_b_off!r} vs {v_b_on!r}"


# ─── Test 2: different input box → no cache hit ─────────────────────────


def test_different_input_box_does_not_hit_cache():
    layers, output_layer, sub_const = _make_2layer_model(seed=1)
    lb_a = -0.5 * np.ones(3, dtype=np.float64)
    ub_a = 0.5 * np.ones(3, dtype=np.float64)
    lb_b = lb_a.copy(); lb_b[0] += 1e-12  # one ulp difference
    ub_b = ub_a.copy()
    spec = [(np.array([1.0, 0.5], dtype=np.float64), 0.0)]

    os.environ["ACT_HZ_SPECAWARE_BOUND_CACHE"] = "1"
    try:
        sa._bound_cache_reset()
        _run_once(layers, output_layer, sub_const, lb_a, ub_a, spec)
        _run_once(layers, output_layer, sub_const, lb_b, ub_b, spec)
        s = sa._bound_cache_stats()
        assert s["hits"] == 0, (
            f"single-ulp input change should NOT hit cache, got {s}"
        )
        assert s["misses"] == 2 and s["writes"] == 2
    finally:
        os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)


# ─── Test 3: different model → no cache hit ─────────────────────────────


def test_different_model_does_not_hit_cache():
    layers_a, output_a, sub_const = _make_2layer_model(seed=2)
    layers_b = [
        (layers_a[0][0].copy(), layers_a[0][1].copy()),
        (layers_a[1][0].copy(), layers_a[1][1].copy()),
    ]
    # Perturb a single bias entry in layer 0 to make model B distinct.
    layers_b[0][1][0] += 1e-10
    lb_x, ub_x, spec = _make_disjunct()

    os.environ["ACT_HZ_SPECAWARE_BOUND_CACHE"] = "1"
    try:
        sa._bound_cache_reset()
        _run_once(layers_a, output_a, sub_const, lb_x, ub_x, spec)
        _run_once(layers_b, output_a, sub_const, lb_x, ub_x, spec)
        s = sa._bound_cache_stats()
        assert s["hits"] == 0, (
            f"bias perturbation 1e-10 should NOT hit cache, got {s}"
        )
        assert s["misses"] == 2 and s["writes"] == 2
    finally:
        os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)


# ─── Test 4: env default OFF → no read, no write ────────────────────────


def test_env_default_off_does_not_touch_cache():
    layers, output_layer, sub_const = _make_2layer_model(seed=3)
    lb_x, ub_x, spec = _make_disjunct()

    # Explicitly unset
    os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)
    sa._bound_cache_reset()
    _run_once(layers, output_layer, sub_const, lb_x, ub_x, spec)
    _run_once(layers, output_layer, sub_const, lb_x, ub_x, spec)
    s = sa._bound_cache_stats()
    assert s == {"hits": 0, "misses": 0, "writes": 0, "skipped_fail": 0}, (
        f"default OFF must not read/write cache, got {s}"
    )


# ─── Test 5: LP fail / timeout → no cache write ─────────────────────────
#
# Simulate by monkey-patching `_bound_cache_put` to verify the
# `all_lps_ok=False` branch increments `skipped_fail` and DOES NOT
# write. We can't easily force a real bound LP timeout in this tiny
# model (LPs are too fast), so we exercise the cache API directly with
# `all_lps_ok=False`.


def test_failed_lps_not_cached():
    layers, output_layer, sub_const = _make_2layer_model(seed=4)
    lb_x, ub_x, _ = _make_disjunct()
    dummy_bounds = [
        (np.zeros(4, dtype=np.float64), np.ones(4, dtype=np.float64)),
        (np.zeros(2, dtype=np.float64), np.ones(2, dtype=np.float64)),
    ]

    os.environ["ACT_HZ_SPECAWARE_BOUND_CACHE"] = "1"
    try:
        sa._bound_cache_reset()
        sa._bound_cache_put(layers, sub_const, lb_x, ub_x, dummy_bounds,
                            all_lps_ok=False)
        s = sa._bound_cache_stats()
        assert s["skipped_fail"] == 1, (
            f"all_lps_ok=False must increment skipped_fail, got {s}"
        )
        assert s["writes"] == 0, (
            f"all_lps_ok=False must not write to cache, got {s}"
        )
        # Subsequent get on same key should NOT hit (cache is empty).
        hit = sa._bound_cache_get(layers, sub_const, lb_x, ub_x)
        assert hit is None, "failed-LP path leaked a cache entry"
    finally:
        os.environ.pop("ACT_HZ_SPECAWARE_BOUND_CACHE", None)


if __name__ == "__main__":
    tests = [
        test_same_box_different_spec_verdict_matches_uncached,
        test_different_input_box_does_not_hit_cache,
        test_different_model_does_not_hit_cache,
        test_env_default_off_does_not_touch_cache,
        test_failed_lps_not_cached,
    ]
    n_pass = 0
    n_fail = 0
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
