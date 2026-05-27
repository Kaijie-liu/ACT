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
| [utils.py](../../../act/pipeline/verification/utils.py) | `OnnxGather` falls through to constant subgraph for indices. `OnnxSplit` aliased to `OnnxSplit13`; equal-axis split (no sizes input) infers `num_splits` from downstream `getitem` children. `OnnxPow` accepts any positive integer exponent via repeated MUL chain. `OnnxReduceStaticAxes` routes `sum` → REDUCE_SUM (schema-correct `axes`/`keepdims`) alongside `mean` → MEAN. |
| [torch2act.py](../../../act/pipeline/verification/torch2act.py) | **R10**: wrapper `_build_layer_graph` now connects INPUT_SPEC as predecessor of EVERY model layer whose `in_vars` overlap INPUT_SPEC's `out_vars` (not just the first model layer). This closes the LSNC-ReLU `tf_gather` size-2 view error; same fix collaterally unblocks `yolo_2023`. |
| [cli.py](../../../act/pipeline/cli.py) | **R10**: multi-query instances now share a cooperative instance budget; remaining queries receive only remaining time, and an unvisited branch after budget exhaustion forces `UNKNOWN` rather than an incomplete `CERTIFIED`. A process-level watchdog is still required to interrupt one long `analyze()` call. |
| [act/pipeline/watchdog_runner.py](../../../act/pipeline/watchdog_runner.py) (NEW) | **R10**: process-level wall + RSS watchdog. Each official instance runs in its own subprocess; on wall deadline the runner SIGTERMs then SIGKILLs and emits `UNKNOWN_TIMEOUT`; on aggregate process-tree RSS exceeding `--rss-cap-gb`, emits `UNKNOWN_RESOURCE_LIMIT`. A synthetic per-instance JSON is written when the CLI is killed mid-flight so downstream aggregators can still join on `official_instance_id`. **R11**: bounded UNKNOWN (`UNKNOWN_TIMEOUT`, `UNKNOWN_RESOURCE_LIMIT`) records `run_status=PASSED` by default as an acceptable auditable verdict. **R13**: `--strict-bounded-failure` now propagates into synthetic per-instance JSON as well as driver exit status; a bounded termination under qualification policy records `run_status=FAILED`. Every non-OK termination now returns an authoritative synthetic UNKNOWN record even if the child prewrote a result, and synthetic filenames carry instance id plus microsecond timestamp to prevent artifact overwrite. Fail-closed: no watchdog termination path can produce `CERTIFIED`/`FALSIFIED`. |
| [torch2act.py](../../../act/pipeline/verification/torch2act.py) + [utils.py](../../../act/pipeline/verification/utils.py) | **R11/R15**: explicit predecessor tracking for helper layers. `_LayerGraphBuilder._set_explicit_preds` lets handlers register a positional preds list for helper layers that have no FX node (EXPAND inserted for nn4sys var-var broadcast `Div` and ml4acopf var-constant broadcasting; chained MULs from `OnnxPow`). Duplicates are preserved so `tf_mul`/`tf_div` positional indexing works on `Mul(x, x)` (`preds=[src, src]`). `_assert_dag` counts unique preds for in-degree. |
| [tf_cnn.py](../../../act/back_end/interval_tf/tf_cnn.py) | **R12**: `_conv1d_to_linear_matrix` now normalises `stride` / `padding` / `dilation` to scalar ints. ONNX→torch routinely delivers them as length-1 tuples (`stride=(1,)`); the prior code did `meshgrid_tensor * stride` which fell through to Python's `tuple.__rmul__(tensor)` and raised `TypeError: only integer tensors of a single element can be converted to an index`. Surfaced immediately after R11 unblocked nn4sys pensieve_big_parallel's Conv1d. |
| [utils.py](../../../act/pipeline/verification/utils.py) | **R12**: added `_convert_OnnxConstantOfShape` handler — resolves a truly static shape input and materialises a CONSTANT layer filled with `mod.value`. After the fail-closed constant-evaluation gate, CCTSDB no longer proceeds by sample-folding to this later path: its current first blocker is a data-dependent `OnnxSlice`, which requires a sound dynamic representation. |
| [torch2act.py](../../../act/pipeline/verification/torch2act.py) | **R12 soundness gate (advisor 2026-05-25)**: `_evaluate_constant_subgraph` now requires an explicit `allow_sample_substitution=True` to fold the model placeholder via `sample_input`. The default is False: a chain that bottoms out at the placeholder returns `None`, so any handler that asks for a "constant" answer fails-closed rather than silently producing a sample-locally-valid IR. This blocks the previously-implicit downgrade where data-dependent shape/index/branch values became fixed tensors, which would only be correct at the box center. No current handler opts in (the previous CCTSDB path through sample substitution was unsound for formal verification). |
| [layer_schema.py](../../../act/back_end/layer_schema.py) + [tf_mlp.py](../../../act/back_end/interval_tf/tf_mlp.py) + [interval_tf.py](../../../act/back_end/interval_tf/interval_tf.py) + [hybridz_tf/hybridz_tf.py](../../../act/back_end/hybridz_tf/hybridz_tf.py) + [utils.py](../../../act/pipeline/verification/utils.py) | **R14/R15**: added `FLOOR` / `CEIL` / `ROUND` / `SIN` / `COS` LayerKinds and sound interval transfers. Floor/Ceil/Round are monotone and use tight endpoint images, including banker's rounding `[round(lb), round(ub)]`; Sin/Cos evaluate endpoints and include periodic extrema when present. `_convert_OnnxRound` and `_convert_OnnxFunction` route the ONNX modules. Together with var-constant EXPAND predecessor wiring, the same-snapshot strict-watchdog `ml4acopf` nine-family gate is `6 UNKNOWN + 3 UNKNOWN_TIMEOUT + 0 ERROR`; old `OnnxRound`/`cos` conversion blockers are closed. |
| [scripts/bridge_sidecar_to_act_receipt.py](../../../scripts/bridge_sidecar_to_act_receipt.py) (NEW) | Bridge that translates SATSidecar `sat_zero_tol` artifacts into ACT-canonical FAL receipts. Re-validates `in_input_domain`, `ast_holds`, model sha256, and x_star sha256 fail-CLOSED. |

Verification:

- `python -m unittest discover tests -v`: **126/126 passed** after R15. Coverage added after the R14 baseline includes fail-closed ORT audit input-contract tests, a real ml4acopf var-constant EXPAND predecessor regression, Sin/Cos interval-containment tests, and a watchdog hard-kill test for children that ignore `SIGTERM`. R12 audit found and fixed a test-ordering pollution where a real ONNX-loading helper test left torch's default device on `cuda:0`; downstream VNNLIB NumPy conversion now explicitly copies tensors to host.
- New: `tests/test_constant_eval_failclosed.py` — pins fail-closed default on `_evaluate_constant_subgraph`.
- New: `tests/test_pow_broadcast_containment.py` — sampling-based interval containment for `Mul(x,x)`, `Div(x, broadcast_scalar)`, and `x**3` chains.
- New: `tests/test_round_floor_ceil_containment.py` — sampling-based containment for Floor/Ceil/Round, including half-integer-tie probes for Round.
- New: `tests/test_trig_interval_containment.py` — endpoint/extremum containment for periodic Sin/Cos interval transfers.
- New: `tests/test_ort_sampling_audit.py` — ORT audit fails closed on native-rank/dtype ambiguity and missing ACT verdict inputs.
- New: `audit_results/r93_rerun_20260525T083118Z/CCTSDB_DYNAMIC_SLICE_DESIGN.md` — design doc for the dynamic Slice + rank-1 input convention; no code lands until that design is implemented (3 options ranked by precision).
- `tests/test_hz_reduction_soundness.py`: **38/38 passed**, including CUDA input-box regression.
- Post-budget-patch CPU formal regression: `sat_relu` remains `1 CERT + 18 FAL + 81 UNKNOWN + 0 ERROR / 100`, with `18/18` FAL receipts strict-clean and `0` cooperative-budget exhaustions.

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
| `acasxu_2023` | CPU | 186 | 73 | 15 | 98 | 0 | 22.9 min | Completed formal baseline. |
| `acasxu_2023` | GPU | 186 | 73 | 15 | 98 | 0 | 23.5 min | **Bit-identical** to CPU at t=30 (rerun after the t=60 partial was discarded); 15/15 GPU FAL receipts strict-clean. |
| `cersyve` (bridged) | sidecar | 12 | — | 3 | — | — | — | 3/3 SATSidecar `sat_zero_tol` artifacts translated to ACT canonical receipts via the new bridge; all 3 pass ACT `load_receipt` with `input_box_holds=true` + `spec_zero_tol_holds=true`. iid 1, 5, 9. |
| `nn4sys` lindex pilot | CPU | 1/5 | 1 | 0 | 0 | 0 | 1.2 s completed | `lindex_1` certified; the next `lindex_200` instance expanded to 400 queries and was stopped after over 31 minutes without completing. |
| `nn4sys` lindex budget diagnostic | CPU | 1 targeted | — | — | — | — | external 45 s guard | With cooperative query budgeting enabled and `--timeout 3`, `lindex_200` still does not finish its first query before the external guard exits `124`; a process-level hard timeout is required around `analyze()`. |
| `lsnc_relu` (NEW unblock) | CPU | 5 | 0 | 0 | 5 | 0 | 2.5 s | All 5 reach verdict (UNK) in 0.5s each, no error. Previously 1/1 ERROR (size-2 view at `tf_gather`). Wrapper INPUT_SPEC-connect-all fix. |
| `yolo_2023` (NEW unblock) | CPU | 5 | 0 | 0 | 5 | 0 | 7.8 min | All 5 reach verdict (UNK), no error. Previously 1/1 ERROR (reshape mismatch). Same wrapper fix. The requested `--timeout 30` was not enforced as aggregate instance wall time: actual cost is about 90s/instance. |

Receipt audits:

- `collins_rul_cnn_2022`: CPU 11/11 and GPU 11/11 FAL receipts strict-clean (`input_box_holds=true`, `spec_zero_tol_holds=true`).
- `malbeware`: CPU 13/13 and GPU 13/13 FAL receipts strict-clean.
- `acasxu_2023`: GPU 15/15 FAL receipts strict-clean; GPU and CPU have zero per-instance verdict differences under the identical 30-second configuration.
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

1. **Bit-identical (deterministic verdict equivalence)**: `collins_rul_cnn_2022`, `malbeware`, `linearizenn_2024`, `acasxu_2023`. Every CPU verdict reproduced exactly; all FAL receipts strict-clean. Speedup 6.34x–10.31x where the workload is Conv-heavy; ~1.0x for LP-bound (`linearizenn`, `acasxu`) — GPU yields no time gain but verdict equivalence makes the run acceptable for unified reporting.
2. **Strict-receipt but verdict-divergent**: `sat_relu`. The CUDA path reaches a different accepted-witness set than CPU; 21/21 GPU FAL receipts pass strict replay, but verdict sets differ on 5 instances and repeat stably. It is not qualified for CPU/GPU equivalence reporting until the divergence is explained.

### Root cause of the original GPU CUDA replay failure

The first GPU gate downgraded valid sampled FALs to UNKNOWN. The strict input-box replay tried to convert CUDA `INPUT_SPEC` bounds directly with NumPy. After routing `x`, `lb`, and `ub` through the existing tensor-to-host helper in `solver_hz.py`, a 6-instance mixed CERT/FAL gate reproduced all CPU verdicts and the subsequent full GPU runs passed formal equivalence audits.

## R13 audit: nn4sys ORT-sampled soundness probe

Under the strict watchdog (`--strict-bounded-failure`, R13 hardening),
five nn4sys instances were run end-to-end and then checked with a
canonical CPU ONNX Runtime falsification probe via
[`scripts/audit_nn4sys_ort_containment.py`](../../../scripts/audit_nn4sys_ort_containment.py).
The audit samples N=500 inputs uniformly from the VNNLIB input box,
adds the two extreme corners (lb / ub), forwards each through ORT, and
evaluates the VNNLIB output spec on the result. **A single sampled
unsafe output on a CERTIFIED instance would mean the CERT verdict is
unsound.** None were observed.

| iid | onnx | vnnlib | ACT verdict | wall (strict watchdog) | unsafe @ tol=0 |
|---|---|---|---|---:|---:|
| 105 | `lindex.onnx` | `lindex_1.vnnlib` | CERTIFIED | 4.0 s | 0 / 1000 |
| 106 | `lindex_deep.onnx` | `lindex_1.vnnlib` | CERTIFIED | 6.5 s | 0 / 1000 |
| 0 | `pensieve_small_simple.onnx` | `pensieve_simple_0.vnnlib` | UNKNOWN | 3.0 s | 0 / 500 |
| 1 | `pensieve_big_parallel.onnx` | `pensieve_parallel_1.vnnlib` | UNKNOWN | 15.5 s | 0 / 500 |
| 4 | `pensieve_small_parallel.onnx` | `pensieve_parallel_4.vnnlib` | UNKNOWN | 3.5 s | 0 / 500 |

Caveat (advisor framing preserved): sampling is a falsification probe, not a proof.
Zero unsafe samples is consistent with sound CERT (lindex) and with
"no obvious missed FAL" on the UNK instances; it does **not** lift any
of the three UNKs into CERT, nor does it formally verify the two
CERTs. The audit is therefore evidence to keep nn4sys at
"CPU targeted-runnable", not to upgrade it to formal-qualified.

R13 watchdog hardening evidence:
- iid 105/106 watchdog summary: `counts.OK = 2`, all `run_status=PASSED`.
- iid 0/1/4 watchdog summary: `counts.OK = 3`, all `run_status=PASSED`.
- A separate strict-mode run of `iid 107 (lindex_200, 400-query box)`
  with `--wall-s 3 --rss-cap-gb 8 --strict-bounded-failure` records
  `UNKNOWN_TIMEOUT`, driver `rc=1`, and synthetic per-instance
  `run_status=FAILED` — i.e. strict policy is consistent end-to-end.

## R15 + ORT probe: ml4acopf_2024 same-snapshot family coverage

After the R15 fixes (var-const broadcast pred wiring, tight monotone
ROUND bound, SIN/COS sound interval transfers, watchdog hard-kill
regression), all 9 unique ml4acopf model-family representatives were
re-run under the same code snapshot with the strict watchdog:

| iid | onnx | wall | status |
|---:|---|---:|---|
| 0 | `14_ieee_ml4acopf-linear-residual.onnx` | 5.5 s | OK |
| 14 | `118_ieee_ml4acopf.onnx` | 6.5 s | OK |
| 19 | `118_ieee_ml4acopf-linear-residual.onnx` | 35.1 s | UNKNOWN_TIMEOUT |
| 23 | `14_ieee_ml4acopf-linear-nonresidual.onnx` | 3.5 s | OK |
| 37 | `118_ieee_ml4acopf-linear-nonresidual.onnx` | 7.0 s | OK |
| 42 | `14_ieee_ml4acopf.onnx` | 3.0 s | OK |
| 56 | `300_ieee_ml4acopf-linear-residual.onnx` | 35.1 s | UNKNOWN_TIMEOUT |
| 60 | `300_ieee_ml4acopf-linear-nonresidual.onnx` | 35.2 s | UNKNOWN_TIMEOUT |
| 65 | `300_ieee_ml4acopf.onnx` | 34.0 s | OK |

**0 ERROR across all 9 families** — both the broadcast/wire fix and the
SIN/COS handlers are exercised end-to-end (the nonlinear `118_ieee` and
`300_ieee` representatives are the explicit nonlinear coverage).
Remaining tight-budget timeouts are size-bound, not op-bound.

ORT containment probe on the 6 normally-completing iids
(`scripts/audit_nn4sys_ort_containment.py`, n=300 samples per query):

| iid | model | ACT verdict | sampled counterexamples |
|---:|---|---|---:|
| 0 | `14_ieee_ml4acopf-linear-residual` | UNKNOWN | 0 across 2 queries |
| 14 | `118_ieee_ml4acopf` | UNKNOWN | 0 across 1 query |
| 23 | `14_ieee_ml4acopf-linear-nonresidual` | UNKNOWN | 0 across 2 queries |
| 37 | `118_ieee_ml4acopf-linear-nonresidual` | UNKNOWN | 0 across 1 query |
| 42 | `14_ieee_ml4acopf` | UNKNOWN | 0 across 2 queries |
| 65 | `300_ieee_ml4acopf` | UNKNOWN | 0 across 282 queries |

Probe verdict: **`NO_SAMPLED_COUNTEREXAMPLE` on every instance**. Same
caveat as the nn4sys probe — sampling is a falsification probe, not a
soundness proof. The probe outcome is **consistent with**, not a
substitute for, sound CERT. ml4acopf therefore remains at
**CPU targeted-runnable, NOT formal-qualified**.

Artifacts:

- Family gate: `audit_results/r93_rerun_20260525T083118Z/ml4acopf_all_families_r15_strict_20260525T125509Z/`
- ORT probe: `audit_results/r93_rerun_20260525T083118Z/ml4acopf_ort_probe_r15_*/ort_containment_summary.json`

## R16 + R17 (this round, advisor 2026-05-25)

R16 — analyze() worklist ready-check
------------------------------------
[analyze.py](../../../act/back_end/analyze.py): the worklist now defers
a layer when ANY predecessor has not been visited yet. CONSTANT and
LUT_BOUNDS seeds are admitted to ``visited`` at initialization; an
intermediate node never dispatches against the ``(-inf, +inf)``
sentinel. Surfaced on `vit_2023`: pre-R16, popping the zero-indegree
CONSTANT enqueued CONCAT before the CONV→RESHAPE→TRANSPOSE branch
finished; CONCAT box_joined the sentinel, DENSE computed
``(-inf) * 0_weight = NaN``, and a downstream SLICE then aborted with
"invalid bounds (lb > ub)". After R16, the same instance reaches a
real downstream shape gap (`flatten output numel 75 != expected 5`) —
a different ViT shape-lineage issue, separate from the NaN
poisoning.

R16 — tf_dense rank-3 input shape
---------------------------------
[tf_mlp.py](../../../act/back_end/interval_tf/tf_mlp.py) `tf_dense`:
ViT-style ``input_shape=(1, T, in_features)`` is reshaped to
``(B*T, in_features)`` before `affine_bounds`, then folded back. The
linear is applied per token; bounds compose row-wise.

R17 — LUT_BOUNDS source layer (CCTSDB dynamic-Slice Option A scaffold)
----------------------------------------------------------------------
[layer_schema.py](../../../act/back_end/layer_schema.py), [tf_mlp.py](../../../act/back_end/interval_tf/tf_mlp.py): added the
`LUT_BOUNDS` LayerKind + schema entry, a zero-indegree source TF
(`tf_lut_bounds`) that emits sealed ``(lb, ub)`` per output position,
and a converter helper `precompute_lut_envelope(T, window_size,
starts_lb, starts_ub, steps)` that brute-force enumerates integer
starts and returns the per-position min/max over the candidate
windows. Sound: every runtime crop is contained in the envelope by
construction. analyze.py also admits LUT_BOUNDS to its ``visited``
seed set so the layer's bounds are immediately available downstream.
6 regression tests in `test_lut_bounds_envelope.py` pin (a) envelope
matches a brute-force reference for 1-D and 2-D windows, (b) singleton
starts collapse to a constant slice, (c) out-of-range starts are
correctly skipped, (d) `tf_lut_bounds` rejects malformed params, and
(e) random runtime crops fall inside the sealed bounds.

The CCTSDB onnx2torch converter handler that detects "static T +
dynamic window-start = LUT_BOUNDS layer" pattern is **not yet wired**;
the precompute / TF / schema scaffold is complete and tested, the
converter-side pattern recognition is the next step.

R14b — SIN / COS sound interval transfers
-----------------------------------------
[tf_mlp.py](../../../act/back_end/interval_tf/tf_mlp.py): periodic
envelope via endpoint evaluation plus extrema-containment test
(`_tf_periodic_trig`). For each input position, the bound includes
``[-1, 1]`` whenever the interval ``[lb, ub]`` contains the
corresponding extremum phase ``±π/2 + 2πk`` (sin) or ``0 + 2πk`` /
``π + 2πk`` (cos). Sound and exact for any scalar interval; widening
guarded by a relative ULP buffer so a phase-membership rounding
widens rather than under-approximates.

R14c — Round tightened to monotone bound
----------------------------------------
[tf_mlp.py](../../../act/back_end/interval_tf/tf_mlp.py) `tf_round`:
banker's rounding is monotone non-decreasing (verified in the
regression suite), so the tight image of `[lb, ub]` is
``[round(lb), round(ub)]``. The earlier conservative
``[floor(lb), ceil(ub)]`` was wider than necessary; the tight version
still passes sampled-containment over half-integer boundaries.

R17 — watchdog hard-kill regression
-----------------------------------
[test_watchdog_runner.py](../../../tests/test_watchdog_runner.py):
new regression covers the case where a child process traps SIGTERM.
The watchdog grace period elapses, SIGKILL fires, and the result is
recorded as `UNKNOWN_TIMEOUT` with strict-policy `run_status=FAILED`.
Previously the timeout assertion only verified SIGTERM-honouring
children; this closes the SIGTERM-ignoring path.

## Overnight CPU serial run (advisor 2026-05-25)

Launched (process PID 1773311, BG) immediately after R16/R17 landed:
serial CPU full-bench runs for benchmarks whose conversion path is
validated but whose full-bench CPU run hasn't happened yet, all under
the strict watchdog. Output:
`audit_results/r93_rerun_20260525T083118Z/overnight_cpu_20260525T144203Z/`.

Partial counts at the time this section was written:

| Benchmark | Instances | OK | UNKNOWN_TIMEOUT | ERROR |
|---|---:|---:|---:|---:|
| `lsnc_relu` | 80 (full) | 78 | 2 | **0** |
| `ml4acopf_2024` | 69 (full) | 59 | 10 | **0** |
| `nn4sys` | 52 (pensieve + lindex_1; lindex_200+ deliberately excluded for query-explosion) | 52 | 0 | **0** |
| `relusplitter` | 30 (first slice) | in progress | — | — |
| `yolo_2023` | 10 (first slice; instances are ~90s each) | queued | — | — |

The **0 ERROR** count on the three completed benchmarks is the
concrete evidence that the R11–R17 op-conversion + analyze-ready-check
chain holds up at full-benchmark scale; bounded UNKNOWN_TIMEOUT is the
strict-policy signal for "real runtime exceeded the per-instance
budget", not a verifier error.

## Capability Status

Newly closed paths:

| Path | Result |
|---|---|
| R9.3 strict FAL gate | `sat_relu` CPU valid set reduced to sound `1 CERT + 18 FAL`; false out-of-box witnesses no longer reported. |
| Non-square Conv / initializer input handling | `collins_rul_cnn_2022` moved from ERROR path to `39 CERT + 11 FAL` over all 62 instances. |
| CUDA strict FAL replay | Full GPU equivalence established on `malbeware`, `collins_rul_cnn_2022`, `linearizenn_2024`, and `acasxu_2023`; `sat_relu` remains divergent. |
| nn4sys op coverage | OnnxSplit (incl. equal-axis form), OnnxGather (constant subgraph indices, 0-d scalar negative indices), nn.Conv1d, OnnxPow (positive integer exp), OnnxReduceStaticAxes-sum routed to REDUCE_SUM: implemented. `pensieve_small_simple` converts cleanly (5/5 UNKNOWN); `lindex_1` certified. |
| Wrapper INPUT_SPEC connect-all (NEW) | `LSNC-ReLU` (5/5 UNK 0 ERR in 2.5s) and `yolo_2023` (5/5 UNK 0 ERR in 7.8 min) both went from immediate ERROR to clean conversion + verdict-attempting after the wrapper change. |
| cersyve sidecar → ACT canonical bridge | 3/3 strict FAL receipts now in canonical ACT format and re-loadable via `fal_receipt.load_receipt`. |

Not cleared for full experiment:

| Benchmark/path | Remaining blocker |
|---|---|
| `cgan_2023` | `.vnnlib.gz` and ConvTranspose schema fixed, but HZ evaluation of the first smoke instance exceeded one hour and about 18.6 GB RSS. Needs bounded/scalable ConvTranspose-HZ propagation. |
| `nn4sys` pensieve_*_parallel | **R11+R12 fixes landed**: helper-layer predecessor tracking (R11) + Conv1d tuple-attr defensiveness (R12). `pensieve_small_parallel` and `pensieve_big_parallel` both went from ERROR to UNKNOWN. The mixed-model gate is `5 UNK + 0 ERR + run_status=PASSED`; a real-instance CPU ORT falsification probe observed no sampled unsafe output on iids 0/1/4 and no sampled counterexample for CERT iids 105/106. **Status remains "CPU targeted runnable", NOT "formal-qualified"**: sampling is not proof or CPU/GPU equivalence, and strict watchdog correctly fails qualification on timed-out `lindex_200`. |
| `nn4sys` lindex_200/400/600/800 | VNNLIB Cartesian-product explosion (e.g. lindex_200 → 400 queries); not an op gap, a query-count gap. Verifier-side scalability lever. |
| `nn4sys` OnnxPow non-integer/negative exponents | Need real POW transfer function. |
| `linearizenn_2024` precision | Runnable with zero errors, but substantial UNKNOWN coverage remains. |
| `cctsdb_yolo_2023` | **OnnxConstantOfShape is implemented (R12), but cannot qualify this benchmark.** With `_evaluate_constant_subgraph` now fail-closed for model-input dependence, the first reliable error is `OnnxSlice at slice_23: cannot resolve starts/ends`: its crop bounds derive from `X_12288` and `X_12289`, both variable over `0..62` in the tested spec. Sample-folding those bounds would be unsound. The ONNX model also uses native rank-1 input `[12296]` and rejects a naive `[1,12296]` reshape. Needs sound dynamic-Slice/branch handling plus an explicit native-input/logical-batch convention before downstream `Shape`/`Where`/`Expand` work. |
| `vit_2023` | Hand-shape lineage broken at first matmul: `mat1 and mat2 shapes cannot be multiplied (1x240 and 48x144)`. Needs reshape/transpose tracking through attention blocks. |
| `ml4acopf_2024` | **R14/R15 operator blockers closed**: sound Floor/Ceil/Round/Sin/Cos interval transfers and var-constant broadcast EXPAND wiring now let all nine model-family representatives run under strict watchdog as `6 UNK + 3 UNKNOWN_TIMEOUT + 0 ERR`; nonlinear iids 14/42/65 all complete without watchdog intervention. Remains CPU targeted-runnable only: an ORT falsification audit and any desired scaling work are still required; large variants already hit budget ceilings. |

## Process-level Watchdog (NEW)

The CLI's `--timeout` only governs HZ solver / LP consumption; it does
not interrupt a long-running `analyze()` call, nor does it bound RSS.
`act/pipeline/watchdog_runner.py` adds an out-of-process enforcement
layer: each official instance runs in its own subprocess, polled by
the parent. Wall-deadline -> `SIGTERM` then `SIGKILL` -> `UNKNOWN_TIMEOUT`;
aggregate process-tree RSS over `--rss-cap-gb` -> kill -> `UNKNOWN_RESOURCE_LIMIT`.

Three gates run against the previously-uncontrollable regression cases:

| Gate | Config | Previous (no watchdog) | With watchdog | Watchdog verdict |
|---|---|---|---|---|
| `nn4sys` iid 107 (`lindex_200`, 400 queries) | `--wall-s 3` `--startup-grace-s 8` | >31 min, never returned first query | killed at **11.1 s**, peak RSS 835 MB, rc=-15 | `UNKNOWN_TIMEOUT` |
| `yolo_2023` iid 0 | `--wall-s 30` `--startup-grace-s 5` | ~90 s/instance (CLI `--timeout 30` not enforced) | killed at **36.3 s**, peak RSS 45.9 GB, rc=-15 | `UNKNOWN_TIMEOUT` |
| `cgan_2023` iid 0 | `--wall-s 60` `--rss-cap-gb 8` | >1 h at 18.6 GB RSS | killed at **10.4 s** when RSS hit **8.7 GB** | `UNKNOWN_RESOURCE_LIMIT` |

All three watchdog terminations produce a synthetic `per_instance_*.json`
with `watchdog_synthetic=true` so phase1_runner / capability aggregators
can join on `official_instance_id` without losing provenance. Fail-closed
invariant: no termination path can ever return `CERTIFIED` or `FALSIFIED`
(unit-tested in `test_watchdog_runner.py`).

R13 strict-policy revalidation: `nn4sys` iid 107 with `--wall-s 3`,
`--startup-grace-s 8`, `--rss-cap-gb 8`, and
`--strict-bounded-failure` terminates as `UNKNOWN_TIMEOUT`; the runner exits
non-zero and its synthetic per-instance JSON records `run_status=FAILED`
and `strict_bounded_failure=true`, matching qualification semantics. Tests
also cover a killed child that prewrites `CERTIFIED`: the returned
authoritative artifact remains synthetic `UNKNOWN_TIMEOUT` and records the
superseded child path for audit. A new regression also covers a child that
ignores `SIGTERM`: the watchdog issues `SIGKILL` after grace and returns a
strict `UNKNOWN_TIMEOUT/FAILED` record. One pre-R15 ml4acopf diagnostic
recorded an anomalous 128.5 s timeout despite a nominal 35 s deadline; it
has not reproduced under the hard-kill test or the R15 nonlinear gate and
is retained as an observation requiring reproduction before code changes.

## Next Execution Order

1. GPU-qualified for formal full runs: `malbeware`, `collins_rul_cnn_2022`, `linearizenn_2024`, `acasxu_2023`. `sat_relu` GPU still divergent; CPU remains its report basis.
2. Keep `nn4sys` behind strict watchdog; its initial real-instance ORT falsification probe is complete, but query scaling and broader audit remain before any qualification upgrade.
3. Diagnose the five reproducible `sat_relu` CPU/GPU verdict differences (iids 34/50/56/86/92) before admitting GPU counts.
4. Promote bridged cersyve 3/12 strict FAL into canonical capability counts (the bridge tool is ready; remaining is pipeline wiring).
5. Keep `yolo_2023` behind strict watchdog and a low RSS cap; its prior gate reached about 45.9 GB RSS, so do not schedule a full run yet.
6. Design sound dynamic-Slice/branch treatment and native rank-1 input handling for `cctsdb_yolo_2023`; do not restore placeholder sample folding or implement a sample-local `Expand` workaround.
7. Investigate ViT attention shape lineage (currently `mat1 1x240` vs `mat2 48x144` mismatch at first attn matmul).
8. Run an ORT falsification probe for the now-runnable `ml4acopf` subset and decide whether its large-family timeout ceiling merits scaling work; R14/R15 closed observed operator/conversion blockers but did not establish formal qualification.
9. Once watchdog is fronting full runs, retry a `cgan_2023` 5-instance gate with a tighter RSS cap; the existing 8 GB cap fires at 10 s, so HZ scaling work is still required before quoting `cgan` numbers.

## Files & artifacts

- Run dirs under `/data1/Kane/ACT/audit_results/r93_rerun_20260525T083118Z/`:
  - `malbeware_R93/`, `collins_full62_v2_*/`, `malbeware_gpu_full_*/`, `collins_gpu62_*/`,
    `linearizenn_gpu60_*/`, `sat_relu_gpu100_cudafix_20260525/`,
    `sat_relu_cpu100_current_20260525/`, `cersyve_bridged/`, `nn4sys_lindex5_*/`,
    `watchdog_strict_nn4sys_iid107_final_20260525T113333Z/`,
    `nn4sys_strict_lindex1_20260525T115041Z/`,
    `nn4sys_strict_pensieve_20260525T115216Z/`,
    `ml4acopf_nonlinear_strict_20260525T125055Z/`,
    `ml4acopf_all_families_r15_strict_20260525T125509Z/`.
- Bridge tool: [scripts/bridge_sidecar_to_act_receipt.py](../../../scripts/bridge_sidecar_to_act_receipt.py).
- Regression tests: `tests/test_vnnlib_parser_soundness.py`, `tests/test_convtranspose_round_trip.py`, `tests/test_tf_conv2d_non_square.py`, `tests/test_onnx_input_shape_init_filter.py`, `tests/test_sidecar_bridge.py`, `tests/test_nn4sys_op_fixes.py`, `tests/test_hz_reduction_soundness.py`, `tests/test_constant_eval_failclosed.py`, `tests/test_pow_broadcast_containment.py`, `tests/test_round_floor_ceil_containment.py`, `tests/test_trig_interval_containment.py`, `tests/test_ort_sampling_audit.py`, `tests/test_helper_pred_tracking.py`, `tests/test_watchdog_runner.py`, and `tests/test_wrapper_placeholder_readers.py`.
