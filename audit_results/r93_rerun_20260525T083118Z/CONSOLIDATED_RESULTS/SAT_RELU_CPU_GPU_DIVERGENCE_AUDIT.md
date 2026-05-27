# sat_relu CPU/GPU Divergence Audit — 5 stable iid differences

**Date**: 2026-05-26 UTC
**Trigger**: 5 stable per-instance verdict diffs between canonical CPU and GPU
runs on `sat_relu`.

## Setup

| Run | Path | iids covered |
|---|---|---|
| canonical CPU | `sat_relu_cpu100_current_20260525/` | 0..99 |
| canonical GPU | `sat_relu_gpu100_cudafix_20260525/` | 0..99 |
| recheck pass A | `sat_relu_divergent_recheck_20260526T020342Z/passA/` | 34,50,56,86,92 |
| recheck pass B | `sat_relu_divergent_recheck_20260526T020342Z/passB/` | 34,50,56,86,92 |

Recheck flags: `device cpu, dtype float64, OMP=MKL=OPENBLAS=1, wall_s=10`.

## Per-iid verdicts

| iid | canon CPU | canon GPU | recheck A | recheck B |
|---:|---|---|---|---|
| 34 | UNKNOWN | **FALSIFIED** | UNKNOWN | UNKNOWN |
| 50 | UNKNOWN | **FALSIFIED** | UNKNOWN | UNKNOWN |
| 56 | UNKNOWN | **FALSIFIED** | UNKNOWN | UNKNOWN |
| 86 | UNKNOWN | **FALSIFIED** | UNKNOWN | UNKNOWN |
| 92 | **FALSIFIED** | UNKNOWN | **FALSIFIED** | **FALSIFIED** |

## Determinism

- `recheck A == recheck B == canonical CPU` on all 5 iids → **CPU is fully
  deterministic** under single-thread float64 settings.
- The divergence is therefore not CPU noise; it is a real device-level
  behavior difference.

## ORT independent replay of all 5 divergent witnesses

For every divergent iid, the FAL-producing receipt (4 GPU receipts + 1 CPU
receipt for iid 92) was replayed via a fresh `onnxruntime.InferenceSession`
on the witness `.x_star.npy`:

| iid | ACT y_cached | ORT y_replay | diff | spec satisfied? |
|---:|---|---|---|---|
| 34 (GPU FAL) | [1, 0] | [1, 0] | 0 | yes (Y_0 ≥ 1.0 ∧ Y_1 ≤ 0.0) |
| 50 (GPU FAL) | [1, 0] | [1, 0] | 0 | yes |
| 56 (GPU FAL) | [1, 0] | [1, 0] | 0 | yes |
| 86 (GPU FAL) | [1, 0] | [1, 0] | 0 | yes |
| 92 (CPU FAL) | [1, 0] | [1, 0] | 0 | yes |

All 5 receipts also carry: `input_box_holds=True`, `spec_zero_tol_holds=True`,
`spec_small_tol_holds=True`.

## Diagnosis

- **All 5 witnesses are genuine, ORT-confirmed counterexamples.**
- No soundness bug on either device.
- The CPU and GPU LP samplers reach slightly different points in the search
  space due to floating-point summation ordering. On the 5 edge instances,
  these differences land on opposite sides of the FAL/UNK boundary.
- CPU finds witnesses for iid 92 that GPU's LP misses; GPU finds witnesses
  for iids 34/50/56/86 that CPU's LP misses.

## Aggregate counts

- canonical CPU FAL  : 18  (includes iid 92, excludes 34/50/56/86)
- canonical GPU FAL  : 21  (includes 34/50/56/86, excludes 92)
- shared FAL          : 17
- CPU-only FAL        :  1  (iid 92)
- GPU-only FAL        :  4  (iids 34, 50, 56, 86)
- **CPU ∪ GPU FAL** : **22 sound counterexamples** (17 + 1 + 4)

## Reporting recommendation

Three acceptable styles for the paper:

1. **CPU canonical only** (simplest): cite `1 V / 18 A / 81 U`. Footnote that
   GPU finds 3 additional sound FAL on 5 stable edge iids, all ORT-verified.
2. **Side-by-side**: cite CPU `1/18/81` and GPU `1/21/78`. Mark the 5 stable
   diff iids with a divergence flag and add the ORT-replay receipt table.
3. **Union**: cite `CPU ∪ GPU sound FAL = 22` with an explicit note that the
   union is the device-diversity coverage (and that intersection = 17).

Do NOT silently pick one of CPU or GPU — both are sound, and the divergence
is structural to the LP sampler under different float ordering.

## Source links

- `CONSOLIDATED_RESULTS/sat_relu/_source_cpu`           → canonical CPU run
- `CONSOLIDATED_RESULTS/sat_relu/_source_gpu`           → canonical GPU run
- `CONSOLIDATED_RESULTS/sat_relu/_source_cpu_recheck_A` → first reproducibility rerun
- `CONSOLIDATED_RESULTS/sat_relu/_source_cpu_recheck_B` → second reproducibility rerun

Rebuild after future GPU rechecks: `python3 build_csvs.py`.
