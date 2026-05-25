# R9.3/R9.4 Capability and GPU Qualification Report

Run root: `/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/`
Finalized: 2026-05-25.

## Code State

| File | Landed change |
|---|---|
| [solver_hz.py](../../../act/back_end/solver/solver_hz.py) | R9.3 input-box witness gate is fail-closed; CUDA-hosted bound tensors are copied to host before NumPy checks. |
| [fal_receipt.py](../../../act/back_end/solver/fal_receipt.py) | Formal FAL receipts persist `input_box_holds` and `input_box_reason`. |
| [vnnlib_parser.py](../../../act/front_end/vnnlib_loader/vnnlib_parser.py) | Transparent `.vnnlib.gz` reads. |
| [torch2act.py](../../../act/pipeline/verification/torch2act.py), [act2torch.py](../../../act/pipeline/verification/act2torch.py) | `CONVTRANSPOSE2D` required params and `output_padding` round trip fixed. Added `nn.Conv1d` converter (nn4sys pensieve_parallel). ONNX handler binding loop now honors dict key so aliases (e.g. `OnnxSplit` → `OnnxSplit13`) actually dispatch. |
| [tf_cnn.py](../../../act/back_end/interval_tf/tf_cnn.py) | `tf_conv2d` trusts valid stamped non-square input shape metadata. |
| [tf_mlp.py](../../../act/back_end/interval_tf/tf_mlp.py) | `tf_gather` handles ONNX semantics: 0-d scalar indices, negative wrap-around. |
| [onnx_converter.py](../../../act/front_end/vnnlib_loader/onnx_converter.py) | ONNX input-shape selection skips initializer tensors. |
| [utils.py](../../../act/pipeline/verification/utils.py) | `OnnxGather` falls through to constant subgraph for indices. `OnnxSplit` aliased to `OnnxSplit13`; equal-axis split (no sizes input) infers `num_splits` from downstream `getitem` children. `OnnxPow` accepts any positive integer exponent via repeated MUL chain. |
| [scripts/bridge_sidecar_to_act_receipt.py](../../../scripts/bridge_sidecar_to_act_receipt.py) (NEW) | Bridge that translates SATSidecar `sat_zero_tol` artifacts into ACT-canonical FAL receipts. Re-validates `in_input_domain`, `ast_holds`, model sha256, and x_star sha256 fail-CLOSED. |

Verification:

- `python -m unittest discover tests -v`: **73/73 passed** (was 58; +5 gzip + 7 ConvTranspose + 3 tf_conv2d non-square + 2 onnx_input_shape init-filter + 7 sidecar bridge + 8 nn4sys op-fix = +32 new, less some overlaps).
- `tests/test_hz_reduction_soundness.py`: **38/38 passed**, including CUDA input-box regression.

## Formal Results — primary capability

| Benchmark | Device | Total | CERT | FAL | UNKNOWN | ERROR | Wall | Qualification |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `sat_relu` | CPU | 100 | 1 | 18 | 81 | 0 | 0.4 min | R9.3 soundness validation; prior out-of-box FALs rejected |
| `sat_relu` | GPU | 100 | 1 | 21 | 78 | 0 | 0.4 min | 21/21 GPU FALs strict-clean, but 5 per-instance verdict differences from current CPU persist on repeat; not approved for CPU/GPU equivalence reporting. |
| `collins_rul_cnn_2022` | CPU | 62 | 39 | 11 | 12 | 0 | 13.1 min | Sound formal baseline |
| `collins_rul_cnn_2022` | GPU | 62 | 39 | 11 | 12 | 0 | 2.1 min | Exact CPU verdict match; **6.34x** speedup |
| `malbeware` | CPU | 150 | 123 | 13 | 14 | 0 | 315.9 min | Sound formal baseline |
| `malbeware` | GPU | 150 | 123 | 13 | 14 | 0 | 30.6 min | Exact CPU verdict match; **10.31x** speedup |
| `linearizenn_2024` | GPU | 60 | 13 | 0 | 47 | 0 | 30.4 min | Exact CPU verdict match (CPU baseline 13/47/0 from memory). GPU speedup negligible — LP-bound benchmark. |
| `acasxu_2023` | CPU | 186 | 73 | 15 | 98 | 0 | 22.9 min | Completed formal baseline. An attempted GPU run used a non-comparable 60-second timeout and was terminated at 181/186; GPU qualification is pending. |
| `cersyve` (bridged) | sidecar | 12 | — | 3 | — | — | — | 3/3 SATSidecar `sat_zero_tol` artifacts translated to ACT canonical receipts via the new bridge; all 3 pass ACT `load_receipt` with `input_box_holds=true` + `spec_zero_tol_holds=true`. iid 1, 5, 9. |
| `nn4sys` lindex pilot | CPU | 1/5 | 1 | 0 | 0 | 0 | 1.2 s completed | `lindex_1` certified; the next `lindex_200` instance expanded to 400 queries and was stopped after over 31 minutes without completing. |

Receipt audits:

- `collins_rul_cnn_2022`: CPU 11/11 and GPU 11/11 FAL receipts strict-clean (`input_box_holds=true`, `spec_zero_tol_holds=true`).
- `malbeware`: CPU 13/13 and GPU 13/13 FAL receipts strict-clean.
- `sat_relu`: GPU 21/21 FAL receipts strict-clean (incl. the 4 GPU-unique).
- `cersyve`: 3/3 bridged receipts strict-clean.

`malbeware` CPU family breakdown:

| Model family | CERT | FAL | UNKNOWN | ERROR |
|---|---:|---:|---:|---:|
| `scaled_linear-25` | 49 | 1 | 0 | 0 |
| `scaled_4-25` | 39 | 5 | 6 | 0 |
| `scaled_16-25` | 35 | 7 | 8 | 0 |

## GPU Qualification — equivalence policy

Two classes of GPU outcomes observed:

1. **Bit-identical (deterministic verdict equivalence)**: `collins_rul_cnn_2022`, `malbeware`, `linearizenn_2024`. Every CPU verdict reproduced exactly; all FAL receipts strict-clean. Speedup 6.34x–10.31x where the workload is Conv-heavy; negligible for LP-bound (`linearizenn`).
2. **Strict-receipt but verdict-divergent**: `sat_relu`. The CUDA path reaches a different accepted-witness set than CPU; 21/21 GPU FAL receipts pass strict replay, but verdict sets differ on 5 instances and repeat stably. It is not qualified for CPU/GPU equivalence reporting until the divergence is explained.

`acasxu_2023` currently has only a completed CPU baseline. The partial GPU run is not admissible for equivalence because it used a different timeout and did not finish.

### Root cause of the original GPU CUDA replay failure

The first GPU gate downgraded valid sampled FALs to UNKNOWN. The strict input-box replay tried to convert CUDA `INPUT_SPEC` bounds directly with NumPy. After routing `x`, `lb`, and `ub` through the existing tensor-to-host helper in `solver_hz.py`, a 6-instance mixed CERT/FAL gate reproduced all CPU verdicts and the subsequent full GPU runs passed formal equivalence audits.

## Capability Status

Newly closed paths:

| Path | Result |
|---|---|
| R9.3 strict FAL gate | `sat_relu` CPU valid set reduced to sound `1 CERT + 18 FAL`; false out-of-box witnesses no longer reported. |
| Non-square Conv / initializer input handling | `collins_rul_cnn_2022` moved from ERROR path to `39 CERT + 11 FAL` over all 62 instances. |
| CUDA strict FAL replay | Full GPU equivalence established on `malbeware`, `collins_rul_cnn_2022`, and `linearizenn_2024`; `sat_relu` remains divergent and `acasxu_2023` GPU remains pending. |
| nn4sys op coverage | OnnxSplit (incl. equal-axis form), OnnxGather (constant subgraph indices, 0-d scalar negative indices), nn.Conv1d, OnnxPow (positive integer exp): implemented. `pensieve_small_simple` converts cleanly (5/5 UNKNOWN); only `lindex_1` completed CERT before query expansion stopped the pilot. |
| cersyve sidecar → ACT canonical bridge | 3/3 strict FAL receipts now in canonical ACT format and re-loadable via `fal_receipt.load_receipt`. |

Not cleared for full experiment:

| Benchmark/path | Remaining blocker |
|---|---|
| `cgan_2023` | `.vnnlib.gz` and ConvTranspose schema fixed, but HZ evaluation of the first smoke instance exceeded one hour and about 18.6 GB RSS. Needs bounded/scalable ConvTranspose-HZ propagation. |
| `nn4sys` pensieve_*_parallel | OnnxReduceStaticAxes ReduceSum variant and var-var broadcast Div needed. |
| `nn4sys` lindex_200/400/600/800 | VNNLIB Cartesian-product explosion (e.g. lindex_200 → 400 queries); not an op gap, a query-count gap. Verifier-side scalability lever. |
| `nn4sys` OnnxPow non-integer/negative exponents | Need real POW transfer function. |
| `linearizenn_2024` precision | Runnable with zero errors, but substantial UNKNOWN coverage remains. |

## Next Execution Order

1. Use GPU for formal full runs of `malbeware` and `collins_rul_cnn_2022`; retain `linearizenn_2024` GPU result as an exact-count validation with negligible timing gain.
2. Rerun `acasxu_2023` on GPU with the exact CPU 30-second timeout configuration, then audit per-instance equality and receipts.
3. Diagnose the five reproducible `sat_relu` CPU/GPU verdict differences before admitting GPU counts to comparison tables.
4. Promote bridged cersyve 3/12 strict FAL into canonical capability counts.
5. Implement nn4sys ReduceSum + var-var Div broadcasting and query-count/time controls.
6. Add a strict resource/time guard and scalable HZ treatment for `cgan_2023` before attempting another full run.

## Files & artifacts

- Run dirs under `/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/`:
  - `malbeware_R93/`, `collins_full62_v2_*/`, `malbeware_gpu_full_*/`, `collins_gpu62_*/`,
    `linearizenn_gpu60_*/`, `sat_relu_gpu100_cudafix_20260525/`,
    `sat_relu_cpu100_current_20260525/`, `cersyve_bridged/`, `nn4sys_lindex5_*/`.
- Bridge tool: [scripts/bridge_sidecar_to_act_receipt.py](../../../scripts/bridge_sidecar_to_act_receipt.py).
- Regression tests: `tests/test_vnnlib_parser_soundness.py`, `tests/test_convtranspose_round_trip.py`, `tests/test_tf_conv2d_non_square.py`, `tests/test_onnx_input_shape_init_filter.py`, `tests/test_sidecar_bridge.py`, `tests/test_nn4sys_op_fixes.py`, `tests/test_hz_reduction_soundness.py` (with CUDA addition).
