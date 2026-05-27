# Scored Benchmark Support Matrix (2026-05-25)

> **Post-snapshot results notice (2026-05-26):** this capability matrix was
> written before the completed CPU Round 2 bounded run.  The authoritative
> per-instance CPU/GPU archive, corrected watchdog-inclusive counts, current
> bounded-smoke statuses, and official-label cross-check now live in
> `CONSOLIDATED_RESULTS/MASTER_INDEX.md` and
> `CONSOLIDATED_RESULTS/SOUNDNESS_VS_VNNCOMP_OFFICIAL.md`.  Do not use the
> pre-Round-2 bounded-smoke rows below as the latest result table.

Source table: `/data1/Kane/HyZor/arXiv-2512.19007v1/benchmarks.tex`,
Table `Overview of all scored benchmarks` (lines 11-70).

This matrix distinguishes:

- **Formal-qualified**: a completed ACT formal run exists with `ERROR=0`.
- **Partial**: a subset runs or strict witnesses exist, but the benchmark is
  not qualified for an ACT full-run result.
- **Blocked**: a concrete current failure or a bounded smoke resource ceiling
  has been observed.

The bounded-smoke classification below used one CPU instance per previously
untested category with an external 25-second wall limit. A timeout is not an
operator failure; it means the category is not yet safe to launch as a full
experiment without resource controls.

## Overview Table Alignment

| Paper benchmark | ACT category | Current status | Evidence / current blocker | Preparation before full run |
|---|---|---|---|---|
| cGAN | `cgan_2023` | Blocked | `.vnnlib.gz` and `ConvTranspose` conversion fixed; post-fix HZ smoke exceeded 1 hour and about 18.6 GB RSS on first instance. | Implement instance wall/RSS guard and scalable `ConvTranspose` HZ treatment; rerun a 1-instance gate. |
| NN4Sys | `nn4sys` | CPU targeted-runnable (NOT formal-qualified) | R11/R12 closed the observed conversion-side blockers: `Gather`/`Split`/`Conv1d`/integer `Pow`/`ReduceSum`/var-var broadcast `Div` (via helper-pred wiring) all land; mixed 5-instance CPU gate is `5 UNK + 0 ERR + run_status=PASSED`. A strict-watchdog ORT falsification probe ran two ACT `CERTIFIED` lindex instances (iids 105/106: zero sampled unsafe across 1000 samples each) and three pensieve `UNKNOWN` instances (iids 0/1/4: zero sampled unsafe across 500 samples each). This sampling evidence is not proof or CPU/GPU equivalence. `lindex_200+` still expands to 400 queries, and strict iid 107 correctly returns `UNKNOWN_TIMEOUT` with `run_status=FAILED`. | Keep behind strict watchdog; address lindex query scaling, then run a broader formal/equivalence audit before any status upgrade. |
| LinearizeNN | `linearizenn_2024` | Formal-qualified | GPU full: `13 CERT + 47 UNKNOWN + 0 ERROR / 60`, matching CPU counts. | Capability improvement requires stronger DAG/relaxation path, not missing operator work. |
| ml4acopf | `ml4acopf_2024` | CPU targeted-runnable (NOT formal-qualified) | R14/R15 add sound `Floor`/`Ceil`/`Round`/`Sin`/`Cos` interval transfers and repair var-constant broadcast helper predecessors. A same-snapshot strict-watchdog gate across all nine model-family representatives (iids 0/14/19/23/37/42/56/60/65) yielded `6 UNKNOWN + 3 UNKNOWN_TIMEOUT + 0 ERROR`; all three nonlinear representatives completed normally (`6.5 s`, `3.0 s`, `34.0 s`). The old immediate `OnnxRound`/`OnnxFunction(cos)` errors are closed; remaining bounded failures are large-family resource/time ceilings. | Run an ORT falsification probe and decide whether scaling work is worthwhile; do not claim formal support until resource-bounded coverage and outcome audit are complete. |
| ViT | `vit_2023` | Blocked | One-instance smoke: `mat1 and mat2 shapes cannot be multiplied (1x240 and 48x144)`. | Fix reshape/transpose/attention shape lineage; validate BatchNorm/Softmax on one model. |
| Collins Aerospace | `collins_aerospace_benchmark` | Blocked by bounded smoke | No first verdict within 25 seconds; structure contains Conv/LeakyReLU/MaxPool/Pow/Resize/Split. | Trace conversion and HZ memory on one instance under wall/RSS cap before any full run. |
| LSNC-ReLU | `lsnc_relu` | Partial/runnable | Wrapper INPUT_SPEC predecessor fix removed the `tf_gather` shape failure; 5-instance CPU gate completes as `5 UNKNOWN + 0 ERROR` in about 2.5 seconds. | Run a larger bounded gate after strict wall-budget enforcement; capability improvement now concerns relaxations, not a conversion blocker. |
| CCTSDB | `cctsdb_yolo_2023` | Blocked | With formal constant evaluation fail-closed, the current first blocker is `OnnxSlice at slice_23: cannot resolve starts/ends`. Its crop indices depend on two variable inputs (`X_12288`, `X_12289`, each spanning `0..62`), so folding from one sample would be unsound. The model also has native rank-1 input `[12296]`, which must not be naively reshaped to `[1,12296]`. | Design sound dynamic-Slice/branch handling while preserving native rank-1 model semantics and logical verification batching; only after that revisit downstream `Shape`/`Where`/`Expand`. |
| Collins RUL CNN | `collins_rul_cnn_2022` | Formal-qualified, GPU-qualified | CPU/GPU both `39 CERT + 11 FAL + 12 UNKNOWN + 0 ERROR / 62`; 0 per-instance verdict diffs; GPU 6.34x faster; 11/11 GPU FAL receipts strict-clean. | Ready for GPU formal reporting. |
| VGGNet16 | `vggnet16_2022` | Blocked by bounded smoke | No first verdict within 25 seconds; operators are otherwise conventional Conv/MaxPool/Gemm/ReLU. | Add memory/time instrumentation and run smallest-instance GPU/CPU gate. |
| Traffic Signs Recognition | `traffic_signs_recognition_2023` | Blocked by bounded smoke | No first verdict within 25 seconds; contains BatchNorm/MaxPool/Sign/Softmax. | Diagnose first layer causing time/memory growth; one-instance formal gate before expansion. |
| cifar100 | `cifar100_2024` | Blocked by bounded smoke | No first verdict within 25 seconds. | Scalable Conv/Residual/BatchNorm path and strict wall/RSS guard; start with one model. |
| tinyimagenet | `tinyimagenet_2024` | Blocked by bounded smoke | No first verdict within 25 seconds. | Same scalability gate as CIFAR100; do not launch full 200 until first-instance bounded success. |
| Metaroom | `metaroom_2023` | Blocked by bounded smoke | No first verdict within 25 seconds. | Measure Conv HZ memory; select a smallest model/property pilot. |
| Yolo | `yolo_2023` | Partial/runnable | Wrapper INPUT_SPEC predecessor fix also removed the prior reshape failure; CPU 5-instance gate completes as `5 UNKNOWN + 0 ERROR`, but takes about 7.8 minutes despite `--timeout 30`. | Fix/enforce aggregate instance/query wall budget before any wider run; then determine whether stronger bounds can produce decisions. |
| SoundnessBench | `soundnessbench` | Blocked by bounded smoke | No first verdict within 25 seconds. | Because soundness is central, require bounded first-instance run plus receipt audit before counting any verdict. |
| Relusplitter | `relusplitter` | Partial/runnable | One-instance smoke returned `UNKNOWN` with no error. | Run 5-instance gate with wall/RSS cap; expand only if decisions or useful coverage emerge. |
| MalBeWare | `malbeware` | Formal-qualified, GPU-qualified | CPU/GPU both `123 CERT + 13 FAL + 14 UNKNOWN + 0 ERROR / 150`; 0 verdict diffs; GPU 10.31x faster; 13/13 GPU FAL receipts strict-clean. | Ready for GPU formal reporting. |
| cersyve | `cersyve` | Partial strict-FAL capability | Native ACT run errors; sidecar bridge emitted 3/3 strict-clean canonical FAL receipts (iids 1, 5, 9). | Wire sidecar route into canonical pipeline/per-instance summary before claiming benchmark full-run support. |
| TLL Verify Bench | `tllverifybench_2023` | Formal-qualified CPU | Completed: `1 CERT + 2 FAL + 29 UNKNOWN + 0 ERROR / 32`. | GPU is optional; low priority unless a timing study is required. |
| Acas XU | `acasxu_2023` | Formal-qualified, GPU-qualified | CPU/GPU (`--timeout 30`) both `73 CERT + 15 FAL + 98 UNKNOWN + 0 ERROR / 186`; 0 per-instance verdict diffs; GPU wall `23.5 min` versus CPU `22.9 min`; 15/15 GPU FAL receipts strict-clean. | Ready for GPU formal reporting, although this LP-bound workload receives no speedup. |
| Dist Shift | `dist_shift_2023` | Formal-runnable, no capability decisions | Completed: `0 CERT + 0 FAL + 72 UNKNOWN + 0 ERROR / 72`. | Diagnosis is precision/capability work, not operator enablement. |
| safeNLP | `safenlp_2024` | Formal-qualified CPU | Completed: `333 CERT + 10 FAL + 737 UNKNOWN + 0 ERROR / 1080`. | Eligible for GPU timing gate if desired; CPU capability already reportable. |
| CORA | `cora_2024` | Blocked by bounded smoke | No first verdict within 25 seconds; model op profile is basic FC arithmetic/ReLU. | Diagnose HZ/LP scale under external timeout before full 180-instance experiment. |
| SAT ReLU | `sat_relu` | Formal-qualified CPU; GPU not equivalent | Current CPU: `1 CERT + 18 FAL + 81 UNKNOWN`; GPU: `1 CERT + 21 FAL + 78 UNKNOWN`. Five stable per-instance differences; all GPU FAL receipts are strict-clean. | Keep CPU result as baseline; analyze deterministic device/search divergence on iids 34, 50, 56, 86, 92 before using GPU counts. |

## Immediate Execution Order

1. Use the implemented process-level wall/RSS watchdog with
   `--strict-bounded-failure` for qualification gates. A real
   `nn4sys` iid 107 gate now terminates as `UNKNOWN_TIMEOUT` and correctly
   writes `run_status=FAILED`; bounded termination is evidence, not qualification.
2. Integrate the bridged `cersyve` strict FAL path into the canonical pipeline
   summary so the three witnesses are directly reportable.
3. Extend the completed `nn4sys` ORT falsification probe only after addressing
   lindex query explosion; keep all such gates behind strict watchdog.
4. Diagnose the five deterministic CPU/GPU `sat_relu` differences; do not
   use GPU totals for soundness comparisons until resolved.
5. On a frozen code snapshot, optionally GPU-qualify the remaining completed
   CPU baselines (`tllverifybench_2023`, `dist_shift_2023`, then
   `safenlp_2024`); keep result fields blank until each full run is audited.
6. For unqualified table categories, proceed by bounded one-instance gates.
   `CCTSDB` first requires sound dynamic-Slice/rank-1-input design; then
   prioritize `ViT` shape lineage and broader bounded `ml4acopf` coverage
   before large resource-bound CNN categories.
