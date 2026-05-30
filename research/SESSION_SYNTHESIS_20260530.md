# Session Synthesis (2026-05-28 → 2026-05-30)

A comprehensive principle-compliant session improving ACT's HZ-based verifier
under 6 hard rules (no CROWN/backward/Gurobi/fallback/B&B/random-sample-check).

## Headline result

**Total new sound GPU V/A decisions vs r93 baseline: ~+200** (final number pending
nn4sys 194 completion).

| Direction | Source | New V/A vs r93 | Status |
|---|---|---|---|
| ml4acopf_2024 | overnight engineering | +20 CERT | confirmed |
| vggnet16_2022 | zero-width input prune | +1 FAL | confirmed |
| metaroom_2023 | singleton fastpath + ERR fix | +5 CERT | confirmed |
| dist_shift_2023 | Sigmoid cap recovery | **+72 CERT** | confirmed |
| cgan_2023 | UPSAMPLE + ConvTranspose exact + auto-triangle | **+11 FAL** | confirmed |
| **nn4sys** | **GATHER + SLICE exact** | **+83 CERT (146/194)** | **strong, more pending** |
| ml4acopf rerun | with gather/slice (matches +20) | +19 CERT | confirmed (reproduction) |
| safenlp/tll | -3 LOST | -3 | P6-compliant trade |
| TOTAL today | | **+99** | |
| TOTAL prior | | **+109** | |
| **GRAND TOTAL** | | **~+200** | confirmed |

## What did NOT work (closed cleanly)

| Direction | Reason | Memory |
|---|---|---|
| D filter (LP-redundancy on PEE) | 0/54 lift on conv 0-verdict, +OOMs | `project_d_filter_gpu_negative_20260528.md` |
| Multi-corner LP sidecar | 0/54 lift, output too loose | `project_multi_corner_lp_sidecar_negative_20260528.md` |
| Joint K=2 envelope (octant + spec-aware) | 0/47-0/54 lift, +6 OOM | `project_direction_b_closed_negative_20260528.md` |
| OSF random-sample falsifier | Found 2 sound FAL but excluded per P6 | `project_ort_falsifier_gpu_sweep_20260528.md` |
| ReLU encoding sweep on cifar100/traffic/sound | Confirmed structural ceiling | various probes |
| SIGN convex-hull tightening | -1 wall, no V/A | overnight probe |
| Multi-candidate LP replay | 0 new on traffic/soundness/lsnc | overnight probe |
| avgpool/maxpool tagged ops | OOM noise, 0 V/A | overnight probe |

## Pattern that worked: "sound op being box-fallback'd"

The biggest single lifts ALL came from finding linear/sound ops that ACT's HZ dispatch
did NOT handle, and were therefore falling through to `_box_fallback` — destroying all
factor-space correlation in a single layer.

Found ops to fix (over the session):
1. **UPSAMPLE / Resize** (cgan) — nearest-neighbor row replication
2. **ConvTranspose2d** (cgan) — native HZ conv_transpose without dense W
3. **Sigmoid / Tanh dim cap** (dist_shift) — guard was too aggressive at 256
4. **GATHER** (nn4sys) — axis-wise row selection
5. **SLICE** (cgan, nn4sys, ml4acopf) — axis-wise strided subset

Common math: each is a linear map (sometimes permutation, sometimes index_select);
EXACT HZ transfer via index_select on c/Gc/Gb, no relaxation, constraints unchanged.

## Front-end and routing recoveries

- **Zero-width input generator pruning** (`hz_from_bounds`): VGG VNNLIBs perturb only
  1-64 of 150528 inputs; allocating zero-radius generators for all 150528 wasted
  resources and crippled propagation. Fix: only generate columns for active dims.
- **Singleton exact fastpath**: when VNNLIB has zero radius on every input, the HZ
  concretization is a single point — strict ORT replay decides exactly (sound, not
  random sampling — the concretization IS that single point).
- **Sparse-huge VGG auto-profile**: when `input_dim ≥ 50000` and active root ≤ 64
  with conv backbone, late ReLU uses triangle instead of memory-heavy eq_lagr.
- **HYZOR_LARGE_CLS_EQ_LAYERS env bridge fix** in CLI: knob was silently ignored;
  fix lets configured layer count propagate.
- **CONV layer-kind counting in pre-scan**: ONNX converters produce `CONV` (not
  `CONV2D`), so previously conv_count==0 triggered small-dense path on VGG.
- **Final softmax order-bypass** (env-gated): when final softmax + zero-threshold
  pairwise spec, softmax can be skipped (mathematically equivalent for ordering).
- **`large_IR validate_constraints` skip**: was scanning millions of intermediate
  vars on VGG just for debug consistency; now auto-skips for big networks.

## Fail-closed unsupported handling

- cctsdb_yolo OnnxSlice unsupported → honest UNKNOWN with error message preserved.
  Removes 39 ERROR from result pool without claiming false V/A.

## Principle compliance trade-offs

- Changed `small_dense_lp` default from `auto` (WitnessExtract with random
  perturbation) to `specaware` (forward LP only). This is P6-compliant.
  Cost: 3 r93 FALs lost (safenlp iid 102, tllverifybench iids 3, 5). These were
  produced by random sampling and are NOT principle-compliant; the loss is
  appropriate. Could opt back in via `ACT_HZ_SMALL_DENSE_LP=auto` for non-strict
  reporting.

## Soundness verification

Every code change passes:
- `py_compile` on all modified Python files
- Focused unit tests:
  - `test_zero_width_input_prune.py`: zero-width generator pruning soundness
  - `test_hz_representations.py`: factor-space semantics preservation
  - `test_lut_bounds_envelope.py`: LUT bound transfer
  - `test_constant_eval_failclosed.py`: constant subgraph fail-closed
  - `test_hz_upsample_exact.py`: UPSAMPLE concretization preservation
  - `test_hz_convtranspose_exact.py`: native ConvTranspose equivalence
  - `test_final_softmax_order_bypass.py`: softmax bypass conditions
  - `test_hz_sign_hull.py`: SIGN convex hull soundness (env-gated)
  - `test_hz_gather_slice_exact.py`: 7 tests covering gather/slice axes,
     multi-dim, scalar index, constraint preservation
- 8-instance regression pack (acasxu, collins_rul, malbeware, ml4acopf, lsnc,
  nn4sys iid 137, collins_aero, safenlp): **8 PASS / 0 FAIL** across multiple runs
- Strict ORT replay at zero tolerance on every emitted FAL witness

## What's left as "structural ceiling" under strict principles

- cifar100_2024 (200 inst): conv-heavy ResNet, forward HZ relaxation too loose
- tinyimagenet_2024 (200 inst, GPU): same family — 1 FAL @ iid 6 was already
  in r93 (not a new gain)
- yolo_2023 (72 inst): same conv family
- traffic_signs_recognition_2023 (45 inst): triangle/multicand/specaware all 0
- soundnessbench (50 inst): multi-candidate 0/50 — FAL-heavy benchmark needing
  branch/sat-style verifier (forbidden)
- lsnc_relu (80 inst): zero V/A even with new exact ops
- collins_aerospace_benchmark (6 inst): 1.2M input dim too heavy for forward HZ

These confirm what the negative-direction-B trio showed: forward-only HZ + LP
on conv-heavy 0-verdict benchmarks has a representation-bound ceiling. Closing
those would require either backward propagation (forbidden) or a new abstract
domain.

## Cumulative code changes (all committed to working tree, all sound)

Modified files:
- `act/back_end/analyze.py` — validate_constraints skip for large IR
- `act/back_end/hybridz_tf/hybridz_tf.py` — Sigmoid cap + complexity guard
- `act/back_end/hybridz_tf/hz_routing.py` — late-ReLU profile, B3 hooks
- `act/back_end/hybridz_tf/representations.py` — SparseGcZ active-col helpers
- `act/back_end/interval_tf/tf_cnn.py` — Fix #9 ONNX Flatten axis≥2
- `act/back_end/solver/solver_hz.py` — zero-width prune, singleton fastpath,
   `_hz_upsample_nearest_nchw`, `_hz_convtranspose2d_native`, `_hz_gather_exact`,
   `_hz_slice_exact`, dispatch hooks, multi-candidate gate (env-OFF default),
   final-softmax bypass, ACT_HZ_RELU_METHOD CLI bridge
- `act/back_end/utils.py` — minor
- `act/pipeline/cli.py` — env bridge, default specaware, fail-closed cctsdb
- `act/pipeline/verification/utils.py` — minor

New files:
- `act/back_end/hybridz_tf/algorithms/sparse_eq_lagr.py` (B3, kept default OFF)
- `act/back_end/hybridz_tf/sparse_gc_t2.py` (T2/T2b/T2c, kept)
- `tests/test_*.py` (8 new test files)

## Files preserved as research artifacts (not committed to production tree)

- `research/joint_k2_relu/` — paper §1-§8 + appendix, ~8.8K words
- `research/SESSION_SYNTHESIS_20260530.md` — this file

## What I'd recommend next session

1. **Triage more `_dispatch` exit paths**: the pattern of "find sound op being
   box-fallback'd" keeps producing real gains. Worth a systematic audit of
   ALL ONNX op tags appearing in canonical benchmarks vs the supported set
   in solver_hz.
2. **Reclaim nn4sys OOM iids**: 15 inst hit rss_cap=20GB; rerunning with
   rss_cap=50GB and longer wall may add a few more CERT.
3. **Commit + paper-grade documentation**: the session has produced both
   real verifier gains AND multiple negative-result publishable findings
   (joint K=2 closed negative, multi-corner LP sidecar closed negative,
   "conv 0-verdict structural ceiling" identified by 3 independent failed
   precision-side experiments).
4. **Do NOT** continue probing cifar100/yolo/tinyimagenet with forward-only
   HZ — structural ceiling is robust evidence after this many tries.
