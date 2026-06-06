# Phase E Gate 0 + Gate 1 v2 — Audit + Streaming-Prune

**Date**: 2026-06-05 late-night (post-S3 advisor critique)
**Status**: Gate 0 PASSED (1472 holds); Gate 1 v2 infra ready
**Headline**: 1472 V/A (audit-validated under strict G4)

---

## 1. Gate 0: Full 558 strict-`>` audit — PASSED 558/558

`audit_results/sc_hz_gate0_full_audit_558_20260605T044242Z/`

Per advisor demand: not just sample, audit ALL 558 safenlp A_CONFIRMED
(546 forward-coeff + 12 S1) under strict `>` threshold rule with full
candidate menu.

```
n_audited:              558
n_STRICT_PASS:          558  (100%)
n_boundary (margin=0):  0
n_fail_other:           0
wall:                   2 seconds (with re-derive + ORT per iid)
```

**Result: 1472 V/A headline holds**. No A_CONFIRMED was a boundary
near-miss. Every witness has d.y > threshold strictly.

---

## 2. Gate 1: S2 v1 → S2 v2 streaming-prune

### Why v1 wasn't sufficient

Advisor critique: the existing `conv_chunked.py` still pre-allocates
`new_G = np.empty((n_out_flat, K), dtype=np.float64)`. This means peak
RESIDENT memory still scales with K. For tinyimagenet 56×56 deep ResNets
with K=60K, the resident `(n_out × K)` matrix alone is multi-GB per layer,
and the value-DAG holds several in-flight states.

The transient-only memory savings of v1 are necessary but not sufficient.

### S2 v2: `research/sc_hz/conv_streaming_prune.py`

New module with `apply_conv2d_streaming_prune` implementing the full
spec advisor demanded:

```python
apply_conv2d_streaming_prune(
    state, W, b, input_shape,
    stride, padding, groups,
    chunk_size, K_target,
)
```

Algorithm:
1. Compute new_c via single Conv on state.c
2. Compute priority per generator column = `||G_in[:, k]||_1`
   - Root-coord generators (origin >= 0) get +1e20 boost
3. Sort by priority descending; keep top K_target indices, drop rest
4. Pass 2: compute Conv only on kept indices, write to new_G_kept
5. Pass 3: compute Conv on dropped indices in chunks, accumulate
   `|chunk_out|.sum(axis=1)` into new_tail_drop
6. new_tail = (|W| @ tail_in) + new_tail_drop
7. new_state has at most K_target kept columns + per-row tail

### Soundness gates (all enforced by tests)

| Gate | Status |
|---|---|
| K_target >= K_old → identical to chunked (no prune) | ✓ tested |
| K_target < K_old → LP UB streaming ≥ LP UB no-prune (any d_out) | ✓ tested |
| Brute-force samples from no-prune set lie in streaming-prune box | ✓ tested (100/100) |
| Root-coord generators (origin >= 0) NEVER dropped | ✓ tested |
| ReLU-slack (origin = -1) dropped first | ✓ implicit via priority |
| chunk_size has zero effect on result (modulo float epsilon) | ✓ tested |
| Tail propagation: |W| @ tail_in + drop_fold dominates baseline | ✓ tested |
| Memory estimator returns sensible values | ✓ tested |

### Test suite: 7 new + 40 prior = **47/47 PASS**

`research/sc_hz/tests/test_conv_streaming_prune_soundness.py`

| Test | Assertion |
|---|---|
| TestNoPruneIdentity.test_no_prune_matches_chunked | K_target=100 vs K=12 → bit-equal chunked |
| TestPruneSoundness.test_lp_ub_streaming_ge_no_prune | UB at K_target ∈ {5,10,20,25} >= UB no-prune |
| TestRootColumnsAlwaysKept.test_root_priority | 8 roots kept even when slack has 100× higher L1 |
| TestBruteForceContainment.test_brute_force_samples_contained | 0/100 no-prune samples outside streaming box |
| TestPropagatesTail.test_tail_propagation_with_drop | streaming tail ≥ no-prune tail per coord |
| TestChunkSizeIndependence.test_chunk_size_does_not_affect_K_keep | c, G_kept bit-equal; tail differs by ≤ 1e-13 (float epsilon) |
| TestStreamingMemoryProfile.test_estimate_basic | resident at K_target=10K is 5.2 GB (vs 31 GB at K=60K) |

---

## 3. Memory profile comparison

| Layer (cifar L8) | v1 chunked peak | v2 streaming peak |
|---|---:|---:|
| transient | 0.75 GB | 0.75 GB (same) |
| resident `new_G` at K=60K | **31 GB** | **2.5 GB** at K_target=5000 |
| resident `new_G` at K_target=5K with v2 prune | n/a | 2.5 GB |
| tail vector | n/a | 0.001 GB (1024 cells × 8B × 32×32) |

For tiny first layer (56×56): v1 resident at K=60K = 16 GB → v2 at K_target=5000 = 1.3 GB.

This is the actual memory reduction advisor required. The full state can fit comfortably in a 40-50 GB envelope per worker even on tiny.

---

## 4. Path forward to Gate 2

Per advisor's chain:

- Gate 0 ✓ (audit done; 1472 holds)
- Gate 1 ✓ (streaming-prune infra; 7 soundness tests)
- **Gate 2** (next): integrate streaming-prune into `onnx_walker_resnet.py` Conv branch + run 40-sentinel memory gate
   - Acceptance: 0 OOM, peak_mem < 80 GB per iid, center parity OK, no skipped ops
   - V/A scoring NOT a gate criterion at this stage
- Gate 3 (after Gate 2): tighter ReLU on last 1-2 layers, +30% LP excess reduction on 8 cifar PHANTOMs
- 2-week kill switch: 0 NEW + LP excess no drop → close SC-HZ phase, move to Phase F new abstraction

---

## 5. Files

| File | Status |
|---|---|
| `research/sc_hz/audit_558_full_strict.py` | NEW — Gate 0 full audit driver |
| `audit_results/sc_hz_gate0_full_audit_558_20260605T044242Z/` | 558/558 STRICT-PASS receipts |
| `research/sc_hz/conv_streaming_prune.py` | NEW — Gate 1 v2 streaming-prune Conv |
| `research/sc_hz/tests/test_conv_streaming_prune_soundness.py` | NEW — 7 soundness tests |
| `research/sc_hz/conv_chunked.py` | v1 (preserved; v2 supersedes for Gate 2+) |
| Unit test suite total | **47/47 pass** |
| `act/` | UNCHANGED |
| 1472 freeze | UNCHANGED (audit-validated) |
