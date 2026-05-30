# Steps 1+2+3 preliminary results (2026-05-27)

## Step 1: T2b longer-wall sweep (resnet_large, in progress)

20 iids ∈ {100, 105, …, 195} × wall=300s × 24 GiB cap × T2b ON.

**Early signal (3/20 done at the time of writing):**

| iid | verdict | wall_s |
|---|---|---|
| 100 | UNKNOWN | 179.2 |
| 105 | UNKNOWN | 157.5 |
| 110 | UNKNOWN | 176.4 |

Key observation: verdicts are **`UNKNOWN`, not `UNKNOWN_TIMEOUT`** —
the analyzer ran to natural algorithmic termination in ~170 s, neither
RSS-bound nor wall-bound. The progression for resnet_large is now:

| budget | failure mode |
|---|---|
| baseline @ 60 s, 24 GiB | `UNKNOWN_RESOURCE_LIMIT` (OOM) |
| T2b @ 90 s, 24 GiB | `UNKNOWN_TIMEOUT` (wall-bound) |
| T2b @ 300 s, 24 GiB | **`UNKNOWN`** (algorithm-bound, ~170 s) |

The failure mode has rotated all the way from RSS-bound (worst) →
wall-bound → algorithm-bound. The remaining gap to CERT is now a
**precision problem**, no longer a resource problem.

## Step 2: Fix #9 — ONNX Flatten(axis≥2) shape assertion

**Bug:** `act/back_end/interval_tf/tf_cnn.py:tf_flatten` asserted
`lb_flat.shape[1] == _prod(output_shape[1:])`, which assumed the
flatten output `(B, rest)` layout. ONNX `Flatten(axis=2)` (used by
the cgan_2023 small_transformer attention block) emits `(BC, rest)`
where dim 0 is `prod(input_shape[:axis])`, not the analyze batch.

**Fix:** compare `lb_flat.shape[1]` against `prod(output_shape) //
max(B_in, 1)` instead (`B_in` is the analyze batch). This handles
any axis without assuming the first output dim is the batch.

### Effect on cgan_2023 ERR cohort (24 GiB cap, 30–60 s wall)

| iid | model | pre-Fix #8/#9 | post-Fix #8 only | post-Fix #8 + #9 |
|---|---|---|---|---|
| 18 | upsample | ERROR_ValueError | UNKNOWN_TIMEOUT | UNKNOWN_TIMEOUT |
| 19 | small_transformer | ERROR_ValueError | ERROR_AssertionError | UNKNOWN_TIMEOUT @ 24 GiB |
| 20 | small_transformer | ERROR_ValueError | ERROR_AssertionError | UNKNOWN_TIMEOUT @ 24 GiB |

cgan_2023 ERROR count: **3 → 0**.

### Soundness gate
Regression pack 8/8 PASS with all edits in place (knobs OFF default
+ knobs ON sweeps).

## Step 3: T2b on tinyimagenet (in progress)

5 iids × baseline + T2b, 24 GiB cap, 120 s wall.

**Baseline (4/5 done):**

| iid | verdict | peak_rss | wall_s |
|---|---|---|---|
| 0 | UNKNOWN_RESOURCE_LIMIT | ~24,000 MiB | 36.4 |
| 1 | UNKNOWN_RESOURCE_LIMIT | ~24,000 MiB | 31.7 |
| 2 | UNKNOWN_RESOURCE_LIMIT | ~24,000 MiB | 33.6 |
| 3 | UNKNOWN_RESOURCE_LIMIT | ~24,000 MiB | 33.0 |

All OOM at the cap, matching r93's `200/200 UNKNOWN_RESOURCE_LIMIT`
on tinyimagenet CPU. T2b mode pending.

## Step 3b: vggnet16 skipped (rationale)

`vggnet16_2022` on CPU r93 was reported as `0/0/0/18/0/0` in the
final table but the per_instance.csv labels say `UNKNOWN_TIMEOUT`
(wall-bound). T2b helps memory-bound cases; wall-bound benchmarks
need an orthogonal lever (longer wall or faster solver). Out of
T2b's scope.

## Net effect on the ACT canonical-source ERROR ledger

| Benchmark | r93 GPU ERR | Post Fix #5/#6/#7 | Post Fix #8/#9 |
|---|---|---|---|
| cgan_2023 | 3 | 3 | **0** |
| (all other canonical sources) | 0 | 0 | 0 |
| **Total canonical ERR** | 3 | 3 | **0** |

This closes Fix #8 candidate from FINAL_RESULTS_TABLE.md §G.

## Files (committed)

| File | Change |
|---|---|
| `act/back_end/hybridz_tf/sparse_gc_t2.py` | T2 + T2b knobs + operators |
| `act/back_end/hybridz_tf/hz_routing.py` | wiring at 3 call sites |
| `act/pipeline/verification/utils.py` | OnnxResize numel filter (Fix #8) |
| `act/back_end/interval_tf/tf_cnn.py` | tf_flatten total-numel assert (Fix #9) |
| `tests/test_sparse_gc_t2.py` | 6 soundness tests |
| `audit_results/.../scripts/regression_pack.sh` | safenlp expected fixed |

## Open items (deferred)

- Step 1 full result (~50 min more compute on background)
- Step 3 T2b mode for tinyimagenet (~10 min more compute)
- T2b longer-wall on cgan iids 19/20 to see if any CERT/FAL
- Precision-side lever (the rotation to algorithm-bound makes
  this the next ROI direction)
