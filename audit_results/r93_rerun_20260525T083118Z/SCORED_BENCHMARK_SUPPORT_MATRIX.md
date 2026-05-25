# Scored Benchmark Support Matrix (2026-05-25)

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
| NN4Sys | `nn4sys` | Partial | `Gather`/`Split`/`Conv1d`/integer `Pow` fixes landed; `pensieve_small_simple` runs 5/5 as `UNKNOWN`; parallel variant still errors at `ReduceSum` / broadcast `Div`; `lindex_200` expands to 400 queries and exceeds practical pilot time. | Add `ReduceSum` and broadcast `Div`; add per-instance/query aggregate timeout; then rerun targeted 5-instance gates by sub-family. |
| LinearizeNN | `linearizenn_2024` | Formal-qualified | GPU full: `13 CERT + 47 UNKNOWN + 0 ERROR / 60`, matching CPU counts. | Capability improvement requires stronger DAG/relaxation path, not missing operator work. |
| ml4acopf | `ml4acopf_2024` | Blocked | One-instance smoke: `Unsupported onnx2torch module OnnxRound at floor_cos_x_62`; models also contain `Sin`/`Cos`/`Floor`. | Implement sound nonlinear transfer/conversion for round/trigonometric path; then conversion and 1-instance gate. |
| ViT | `vit_2023` | Blocked | One-instance smoke: `mat1 and mat2 shapes cannot be multiplied (1x240 and 48x144)`. | Fix reshape/transpose/attention shape lineage; validate BatchNorm/Softmax on one model. |
| Collins Aerospace | `collins_aerospace_benchmark` | Blocked by bounded smoke | No first verdict within 25 seconds; structure contains Conv/LeakyReLU/MaxPool/Pow/Resize/Split. | Trace conversion and HZ memory on one instance under wall/RSS cap before any full run. |
| LSNC-ReLU | `lsnc_relu` | Blocked | One-instance smoke: `shape '[1, 1, 6]' is invalid for input of size 2`. | Repair input/shape propagation, then test `ReduceSum`/arithmetic path on a 5-instance gate. |
| CCTSDB | `cctsdb_yolo_2023` | Blocked | One-instance smoke: `OnnxExpand at expand_94: cannot resolve target shape`. | Resolve constant/dynamic `Expand` shape, then gate `Resize`/`ScatterND`/`Where` pipeline. |
| Collins RUL CNN | `collins_rul_cnn_2022` | Formal-qualified, GPU-qualified | CPU/GPU both `39 CERT + 11 FAL + 12 UNKNOWN + 0 ERROR / 62`; 0 per-instance verdict diffs; GPU 6.34x faster; 11/11 GPU FAL receipts strict-clean. | Ready for GPU formal reporting. |
| VGGNet16 | `vggnet16_2022` | Blocked by bounded smoke | No first verdict within 25 seconds; operators are otherwise conventional Conv/MaxPool/Gemm/ReLU. | Add memory/time instrumentation and run smallest-instance GPU/CPU gate. |
| Traffic Signs Recognition | `traffic_signs_recognition_2023` | Blocked by bounded smoke | No first verdict within 25 seconds; contains BatchNorm/MaxPool/Sign/Softmax. | Diagnose first layer causing time/memory growth; one-instance formal gate before expansion. |
| cifar100 | `cifar100_2024` | Blocked by bounded smoke | No first verdict within 25 seconds. | Scalable Conv/Residual/BatchNorm path and strict wall/RSS guard; start with one model. |
| tinyimagenet | `tinyimagenet_2024` | Blocked by bounded smoke | No first verdict within 25 seconds. | Same scalability gate as CIFAR100; do not launch full 200 until first-instance bounded success. |
| Metaroom | `metaroom_2023` | Blocked by bounded smoke | No first verdict within 25 seconds. | Measure Conv HZ memory; select a smallest model/property pilot. |
| Yolo | `yolo_2023` | Blocked | One-instance smoke: `shape '[1, 3, 1, 14421]' is invalid for input of size 43264`. | Fix output/reshape shape lineage, then verify Pad/AveragePool path. |
| SoundnessBench | `soundnessbench` | Blocked by bounded smoke | No first verdict within 25 seconds. | Because soundness is central, require bounded first-instance run plus receipt audit before counting any verdict. |
| Relusplitter | `relusplitter` | Partial/runnable | One-instance smoke returned `UNKNOWN` with no error. | Run 5-instance gate with wall/RSS cap; expand only if decisions or useful coverage emerge. |
| MalBeWare | `malbeware` | Formal-qualified, GPU-qualified | CPU/GPU both `123 CERT + 13 FAL + 14 UNKNOWN + 0 ERROR / 150`; 0 verdict diffs; GPU 10.31x faster; 13/13 GPU FAL receipts strict-clean. | Ready for GPU formal reporting. |
| cersyve | `cersyve` | Partial strict-FAL capability | Native ACT run errors; sidecar bridge emitted 3/3 strict-clean canonical FAL receipts (iids 1, 5, 9). | Wire sidecar route into canonical pipeline/per-instance summary before claiming benchmark full-run support. |
| TLL Verify Bench | `tllverifybench_2023` | Formal-qualified CPU | Completed: `1 CERT + 2 FAL + 29 UNKNOWN + 0 ERROR / 32`. | GPU is optional; low priority unless a timing study is required. |
| Acas XU | `acasxu_2023` | Formal-qualified CPU; GPU revalidation pending | CPU full: `73 CERT + 15 FAL + 98 UNKNOWN + 0 ERROR / 186`. A GPU run was terminated because it incorrectly used a 60-second timeout instead of the CPU-comparable 30 seconds and stalled in the tail. | Rerun GPU with identical 30-second configuration and audit per-instance equivalence/receipts. |
| Dist Shift | `dist_shift_2023` | Formal-runnable, no capability decisions | Completed: `0 CERT + 0 FAL + 72 UNKNOWN + 0 ERROR / 72`. | Diagnosis is precision/capability work, not operator enablement. |
| safeNLP | `safenlp_2024` | Formal-qualified CPU | Completed: `333 CERT + 10 FAL + 737 UNKNOWN + 0 ERROR / 1080`. | Eligible for GPU timing gate if desired; CPU capability already reportable. |
| CORA | `cora_2024` | Blocked by bounded smoke | No first verdict within 25 seconds; model op profile is basic FC arithmetic/ReLU. | Diagnose HZ/LP scale under external timeout before full 180-instance experiment. |
| SAT ReLU | `sat_relu` | Formal-qualified CPU; GPU not equivalent | Current CPU: `1 CERT + 18 FAL + 81 UNKNOWN`; GPU: `1 CERT + 21 FAL + 78 UNKNOWN`. Five stable per-instance differences; all GPU FAL receipts are strict-clean. | Keep CPU result as baseline; analyze deterministic device/search divergence on iids 34, 50, 56, 86, 92 before using GPU counts. |

## Immediate Execution Order

1. Rerun `acasxu_2023` on GPU with the CPU-comparable 30-second timeout;
   only accept it after per-instance equality and FAL receipt audit.
2. Integrate the bridged `cersyve` strict FAL path into the canonical pipeline
   summary so the three witnesses are directly reportable.
3. Complete `nn4sys` arithmetic support (`ReduceSum`, broadcast `Div`) and
   add an aggregate query/wall timeout for `lindex`.
4. Diagnose the five deterministic CPU/GPU `sat_relu` differences; do not
   use GPU totals for soundness comparisons until resolved.
5. For unqualified table categories, proceed by bounded one-instance gates,
   prioritizing explicit code fixes (`LSNC-ReLU`, `CCTSDB`, `Yolo`,
   `ml4acopf`, `ViT`) before large resource-bound CNN categories.
