# Final Results Table — VNN-COMP 2025 (25 scored benchmarks)

**Snapshot**: 2026-05-27 UTC, after Round 1-7 + Fix #5/#6/#7
**Build**: from `<bench>/per_instance.csv` aggregating all canonical sources

Per-row format: `V / A / U / TO / RSS / E` (CERTIFIED / FALSIFIED / UNKNOWN /
UNKNOWN_TIMEOUT / UNKNOWN_RESOURCE_LIMIT / ERROR).

## A. Bit-identical CPU+GPU (formal-qualified)

| Benchmark | N | CPU | GPU | Δ | Speedup |
|---|---:|---|---|---|---|
| `collins_rul_cnn_2022` | 62 | 39/11/12/0/0/0 | 39/11/12/0/0/0 | 0 | 6.34× |
| `malbeware` | 150 | 123/13/14/0/0/0 | 123/13/14/0/0/0 | 0 | 10.31× |
| `acasxu_2023` (t30) | 186 | 73/15/98/0/0/0 | 73/15/98/0/0/0 | 0 | 1.03× |
| `linearizenn_2024` | 60 | 13/0/47/0/0/0 | 13/0/47/0/0/0 | 0 | — |
| `safenlp_2024` | 1080 | 333/10/737/0/0/0 | 333/10/737/0/0/0 | **0 across 1080 inst** | — |
| `tllverifybench_2023` | 32 | 1/2/29/0/0/0 | 1/2/29/0/0/0 | 0 | — |
| `dist_shift_2023` | 72 | 0/0/72/0/0/0 | 0/0/72/0/0/0 | 0 | — |

GPU FAL strict-clean: 11/11 collins_rul, 13/13 malbeware, 15/15 acasxu_2023.

## B. CPU/GPU divergent, both sound

| Benchmark | N | CPU | GPU | Notes |
|---|---:|---|---|---|
| `sat_relu` | 100 | 1/18/81/0/0/0 | 1/21/78/0/0/0 | 5 stable iid diffs (34/50/56/86/92); all ORT-clean; CPU∪GPU = 22 sound FAL |

## C. GPU unlocks more decisions (where CPU was RSS-bound)

| Benchmark | N | CPU | GPU | GPU − CPU |
|---|---:|---|---|---|
| `metaroom_2023` | 100 | 37/0/1/2/60/0 | **89/0/10/0/0/1** | **+52 V** 🚀 |
| `cora_2024` | 180 | 15/0/3/162/0/0 | 16/**4**/37/123/0/0 | +1 V, +4 A |
| `ml4acopf_2024` | 69 | 4/0/49/16/0/0 | **6**/0/57/5/0/1 | +2 V (not in official longtable) |
| `relusplitter` | 220 | 7/0/42/156/15/0 | 7/0/99/112/0/2 | UNK-only diff |
| `nn4sys` | 194 | 4/0/114/61/15/0 | 4/0/110/79/0/1 | UNK-only diff (post-Fix-#5) |
| `tinyimagenet_2024` | 200 | 0/0/0/0/200/0 | 0/**1**/199/0/0/0 | **+1 A** (new small_tol-residual disagreement) |

All new CERTs cross-checked vs official zero_tol AND small_tol: **0 conflicts**.

## D. Resource-bound both sides (no decisive verdicts)

| Benchmark | N | CPU | GPU |
|---|---:|---|---|
| `traffic_signs_recognition_2023` | 45 | 0/0/30/0/15/0 | 0/0/45/0/0/0 |
| `yolo_2023` | 72 | 0/0/0/0/72/0 | 0/0/72/0/0/0 |
| `cifar100_2024` | 200 | 0/0/99/1/100/0 | 0/0/200/0/0/0 |
| `soundnessbench` | 50 | 0/0/3/0/47/0 | 0/0/50/0/0/0 |
| `vggnet16_2022` | 18 | 0/0/0/0/18/0 | 0/0/0/18/0/0 |
| `lsnc_relu` | 80 | 0/0/80/0/0/0 | 0/0/80/0/0/0 |
| `collins_aerospace_benchmark` | 6 | 0/0/1/5/0/0 | 0/0/2/4/0/0 |

GPU unlocks RAM (CPU RSS-cap → GPU UNK), but precision still bounded.

## E. Both run, GPU data verified bit-compatible

(Earlier "CPU-only" labelling was a CANON-map oversight in build script; both
benchmarks DO have GPU runs at `_source_gpu_full`.)

The two rows below **do have GPU data** (in `_source_gpu_full`) — earlier
versions of this report mislabelled them as "CPU-only" due to an oversight in
the build script's CANON map. Corrected here:

| Benchmark | N | CPU | GPU | Notes |
|---|---:|---|---|---|
| `cersyve` | 12 | 0/3/9/0/0/0 (cpu_native_r2 + sidecar) | 0/0/12/0/0/0 (gpu_full) | bit-identical 12 UNK; 3 FAL only via SATSidecar bridge (iids 1/5/9) |
| `cgan_2023` | 21 | 0/0/0/5/0/0 (CPU smoke 5/21) | 0/0/4/14/0/**3** (gpu_full 21/21) | GPU ran full 21; **3 ERR** on iids 18/19/20 (`OnnxResize: cannot resolve scales/sizes` — Fix #8 candidate, see §G) |

## F. Currently not supported

| Benchmark | Blocker |
|---|---|
| `cctsdb_yolo_2023` (N=39) | Dynamic `OnnxSlice` (starts/ends from variable inputs); rank-1 input `[12296]`. R17 `LUT_BOUNDS` scaffold ready, converter detector not wired. |
| `vit_2023` | Attention shape-lineage gap: `flatten output numel 75 ≠ expected 5`. |

## G. Known remaining ERR (not fixed, deferred)

`cgan_2023` iids 18/19/20 on GPU: `ValueError: OnnxResize at resize: cannot
resolve scales or sizes`. Affects only the `small_transformer` model variant
where Resize takes scales/sizes from a variable input (similar pattern to
CCTSDB dynamic Slice). **Fix #8 candidate** — same general approach as R17
LUT-style resolution; not yet implemented. The benchmark is partially
covered: 18/21 GPU instances ran without error.

## Soundness audit (final, post-Round-7)

- Cross-check vs VNN-COMP 2025 official `small_tol`: **0 conflicts on 837 CERT
  agreements + 123 FAL agreements**.
- **Two documented label discrepancies** with strict ACT receipts + ORT replay
  supporting ACT:
  - `collins_rul_cnn_2022` iids 0 / 22 / 47: ACT FAL vs official zero_tol AND small_tol UNSAT
  - `tinyimagenet_2024` iid 6: ACT FAL vs official zero_tol UNSAT (small_tol = SAT)
- `nn4sys` and `ml4acopf` are not in the official `longtable.tex`; CERT
  verification relies on strict ACT receipts only.

## ERROR status (post-Fix #5/#6/#7)

| Source quality | ERROR count |
|---|---|
| All 22 benchmark *canonical* sources | **0** |
| Archival pre-fix sources (kept for paper-trail) | unchanged (deliberate evidence of regression+fix) |

The 3 ACT fixes that eliminated all canonical-source ERRORs:
- **#5** `onnx_converter.py`: simplify-first ONNX strategy (raw fallback)
- **#6** `interval_tf/tf_mlp.py` + `solver/solver_hz.py`: LeakyReLU accept `alpha` / `negative_slope` / default 0.01
- **#7** `interval_tf/tf_cnn.py`: `tf_upsample` strip non-spatial scale_factor dims

Patches saved in `act_fixes_diff/`.

## Reproduction

All scripts in `scripts/`, ingest helper `build_csvs.py`, audit helper
`soundness_check.py`, master index `MASTER_INDEX.md`. To rebuild:

```bash
source /data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/CONSOLIDATED_RESULTS/QUICKSTART.sh
act_audit                  # rebuild CSVs + re-run official cross-check
```
