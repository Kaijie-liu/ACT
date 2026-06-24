# HyZor Stage-II Consolidation Audit 2026-06-24

Scope: productize the strict pure-HybridZ capability that produced the frozen
ICSE artifact into ACT backend code, then shrink/remove experiment scaffolding
without changing soundness or frozen results.

Frozen baseline:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25`

Soundfix headline: `1763/2213 = 977 CERT + 786 ADV`, `P0=0`.

2026-06-25 erratum: the previous 2026-06-24 headline
`1768/2213 = 983 CERT + 785 ADV` is retained only as historical provenance.
The metaroom row was corrected from `100 CERT / 0 ADV` to
`94 CERT / 1 ADV / 5 TIMEOUT` after fixing the package runner's split-disjunct
aggregation rule. Current reporting files are under
`/data1/Kane/ICSE/act_hybridz_soundfix_20260625`.

Current script-to-mainline inventory:

`docs/HYZOR_STAGE2_SCRIPT_MIGRATION_MATRIX_20260624.md`

That matrix is the authoritative current-state view of which `scripts/`
capabilities are migrated, partial, excluded, or future work.  Some historical
gap notes below record earlier stages of the same consolidation and should be
read in that context.

## 2026-06-24 Mainline Freeze Addendum

The counted frozen artifact remains:

`/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25`

The current mainline-facing entry point is:

`python -m act.pipeline --verify hybridz-benchmark --category frozen`

The benchmark runner is now package-owned, not script-owned:

`act/pipeline/hybridz_benchmark_runner.py`

The safenlp projected one-ReLU exact-HZ portfolio branch is also package-owned:

`act/pipeline/hybridz_projected_relu_mip.py`

Runner self-test now checks, for every frozen benchmark branch, that generated
commands are `python -m ...` package invocations, that no command references
`scripts/` or legacy worker names, that branch policy metadata contains no
`iid` or `instance-index` hook, and that no counted branch selects Gurobi in its
environment.  Pure HybridZ imports no longer eager-load or auto-detect the
Gurobi backend; Gurobi remains available only when explicitly requested as a
diagnostic oracle, and is not part of the counted HybridZ proof path.
The same self-test also guards that plain `act.pipeline` imports do not pull in
conversion, profiling, verifier-validation, LLM-probe, or optional solver
diagnostic modules that are irrelevant to the counted HybridZ benchmark runner.

The frozen-result CSVs and ranking artifacts are preserved under the frozen
artifact directory, including:

- `FINAL_HYBRIDZ_RESULTS_20260624.csv`
- `FINAL_CROSS_TOOL_RANKING_20260624.csv`
- `FROZEN_ARTIFACT_AUDIT.csv`
- `FUTURE_WORK_HYBRIDZ_20260624.md`

2026-06-25 frontend smoke evidence:

Direct standard verifier entrypoint:

```bash
ACT_HYBRIDZ_MEM_FLOOR_GB=0 /usr/bin/timeout 120 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify vnnlib \
  --category safenlp_2024 \
  --instance-index 0 \
  --solvers hybridz \
  --hybridz-timeout 5 \
  --hybridz-results-dir /tmp/act_hybridz_vnnlib_direct_smoke_20260625 \
  --device cpu \
  --dtype float64
```

Result: command exited `0`; summary was `safenlp_2024,N=1,CERT=0,ADV=1,V+A=1,
P0=0`; detail had `solver=hybridz`, `engine=dense_hz_objbound`, and
`hz_witness_source=milp_objective_bound`.

Direct standard verifier CERT smoke:

```bash
ACT_HYBRIDZ_MEM_FLOOR_GB=0 /usr/bin/timeout 120 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify vnnlib \
  --category metaroom_2023 \
  --instance-index 0 \
  --solvers hybridz \
  --hybridz-timeout 5 \
  --hybridz-results-dir /tmp/act_hybridz_vnnlib_direct_metaroom_smoke_20260625 \
  --device cpu \
  --dtype float64
```

Result: command exited `0`; summary was `metaroom_2023,N=19,CERT=19,ADV=0,
V+A=19,P0=0`; detail had `solver=hybridz` and `engine=dense_hz_objbound`.

Benchmark runner entrypoint:

```bash
ACT_HYBRIDZ_MEM_FLOOR_GB=0 /usr/bin/timeout 180 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark \
  --category safenlp_2024 \
  --max-instances 1 \
  --hybridz-workers 1 \
  --hybridz-timeout-cap 5 \
  --hybridz-results-dir /tmp/act_hybridz_safenlp_frontend_smoke_20260625 \
  --device cpu \
  --dtype float64
```

Result: command exited `0`; summary was `safenlp_2024,N=1,CERT=0,ADV=1,V+A=1,
P0=0`; detail used `engine=sparse_hz_objbound`; output files included
`safenlp_2024_hybridz_detail.csv`, `safenlp_2024_hybridz_summary.csv`,
`safenlp_2024_icse_detail.csv`, `safenlp_2024_icse_index.csv`, and
`_MANIFEST.sha256`.

The frozen-match guard was also checked:

```bash
ACT_HYBRIDZ_MEM_FLOOR_GB=0 /usr/bin/timeout 60 \
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark \
  --category frozen \
  --max-instances 1 \
  --hybridz-workers 1 \
  --hybridz-timeout-cap 1 \
  --hybridz-results-dir /tmp/act_hybridz_require_match_reject_20260625 \
  --hybridz-require-frozen-match \
  --device cpu \
  --dtype float64
```

Result: command exited `1` with
`--hybridz-require-frozen-match requires a full frozen suite; do not pass
--max-instances`.  This verifies that the full frozen gate cannot be
accidentally satisfied by a partial smoke run.

## Non-Negotiable Rules

- Count only pure HybridZ engine results.
- No input split.
- No sampling/PGD or LP-witness promotion.
- No CROWN/backward rescue.
- No Gurobi-counted proof.
- ORT is audit only.
- Exact ReLU verdict path only; no ReLU triangle decision.
- Any migration or cleanup that reduces frozen V+A or creates `P0>0` must be
  reverted.

## Current State

Branch: `hz-cam-1`.

Upstream baseline after fetch: `upstream/main` at `3a2853615`.

Current diff against `upstream/main` after the 2026-06-25 refresh:

- `42` tracked files changed under `act/`;
- about `6591` insertions / `2117` deletions under `act/`;
- hybridz-specific additions include:
  - `act/back_end/hybridz_tf/algorithms/order_reduce.py`;
  - `act/back_end/hybridz_tf/algorithms/sgm.py`;
  - `act/back_end/solver/solver_hz_verdict.py`;
  - large changes in `hybridz_tf.py`, `tf_mlp.py`, `tf_cnn.py`,
    `solver_hz.py`.

This broad diff includes branch-level history outside the current Stage-II
work.  The narrower tracked/untracked Stage-II worktree footprint is recorded
in `HYZOR_STAGE2_DIFF_BOUNDARY_AUDIT_20260624.md`; the current uncommitted
tracked Stage-II footprint is `26` files, about `4802` insertions / `687`
deletions, excluding local `scripts/` and docs.  Dirty working tree also has
local docs/scripts. The remaining `scripts/` directory is source-only after
CIFAR matrix dumps were removed.

## Backend Capability Already Present

These are already in ACT backend and should be hardened, not reimplemented:

| Capability | Backend location | Status |
|---|---|---|
| Dense `HZono` 6-tuple with equality/inequality senses | `act/back_end/solver/solver_hz.py` | present |
| Generator identity tracking for residual/share merge | `solver_hz.py`, `hybridz_tf/algorithms/sgm.py` | present |
| Exact redundancy removal | `hybridz_tf/algorithms/order_reduce.py`, `tf_mlp.hz_reduce` | strict product path is exact-only; lossy Girard helper remains explicit audit/ablation code, not called by propagation |
| Exact eq_lagr ReLU | `hybridz_tf/tf_mlp.py:hz_apply_relu` | present |
| Compressed exact ReLU formulation | `hz_apply_relu(..., compressed=True)` | present in dense HZ path |
| Redundant exact ReLU valid cuts | `hz_apply_relu(..., valid_cuts=True)` | present; must remain selective |
| S-curve piecewise HZ encoding | `hz_apply_sigmoid`, `hz_apply_tanh` | present; dense, compressed default |
| AvgPool2d / ConvTranspose2d exact affine HZ | `hybridz_tf/tf_cnn.py` | present |
| MaxPool2d exact pairwise max | `hybridz_tf/tf_cnn.py:hz_maxpool2d` | present; exact-only |
| MatMul constant-side HZ handling | `hybridz_tf/tf_mlp.py:tf_matmul` | present |
| Sparse final MILP construction from dense HZ constraints | `solver_hz_verdict.py` | present |
| HiGHS objective-target / cutoff-row exact verdict | `solver_hz_verdict.py` | present |
| Optional SCIP backend / HiGHS->SCIP portfolio | `solver_hz_verdict.py` | present |
| Exact singleton/equality substitution presolve | `solver_hz_verdict.py` | present; dataset-dependent |

## Script-Only Capability That Must Move Or Be Retired

| Script | Role today | Stage-II decision |
|---|---|---|
| `scripts/cifar_sparse_exact_probe.py` | Main sparse CSR exact-HZ propagation and sparse MILP probe; supports Conv2d, ConvTranspose, AvgPool, MaxPool, MatMul, Softmax, S-curve cuts, base-HZ feasibility guard, SCIP/HiGHS/Gurobi diagnostics | Move core data structure and operators into backend; keep only a thin regression/debug CLI or delete |
| `scripts/hz_sparse_worker.py` | Subprocess wrapper around `cifar_sparse_exact_probe.py`; parses stdout into CERT/ADV/UNKNOWN | Replace with backend solver mode; no subprocess in product path |
| `scripts/hz_full_driver.py` | Per-benchmark portfolio, wall time, worker count, memory governor, branch scheduling | Move benchmark profiles to backend/config; keep as optional batch runner only |
| `scripts/hz_full_worker.py` | One-instance dense-HZ worker with RLIMIT, exact verdict, ORT audit | Replace by front-end `--solvers hybridz`; keep as legacy reproduction helper until parity |
| `scripts/hz_result_ledger.py`, `hz_export_icse_csv.py`, `hz_failure_taxonomy.py` | Reporting/export | Keep outside verifier; move stable reporting into `tools/` or ICSE artifact scripts later |
| `scripts/hz_operator_audit.py`, `validate_sparse_ops.py`, `hz_sparse_attention_operator_audit.py` | Toy/operator regression gates | Keep as tests; convert the stable subset to `tests/` |
| `scripts/hz_projected_relu_mip.py` | Safenlp one-hidden-ReLU exact-HZ projection experiment | Selected safenlp portfolio behavior migrated to `act/pipeline/hybridz_projected_relu_mip.py`; keep script as legacy provenance |
| `scripts/hz_projected_graph_mip.py`, `layer_graph_*`, `primal_cex_finder.py` | Alternative projected/primal experiments | Do not productize as HybridZ unless a backend exact-HZ formulation is formally selected |
| `scripts/*_census.py`, `*_audit.py`, `*_diag.py` | Exploration/census | Keep as research notes or delete after corresponding backend tests exist |

## Productization Gaps

1. `--solvers hybridz` is not a first-class solver mode.

   Current ACT frontends support `--tf-modes hybridz`, but the solver remains
   `torchlp/gurobi/dual`. `HZSolver` only exposes `compute_bounds`; it does not
   own the output-spec verdict. The dense exact-HZ verdict lives in
   `solver_hz_verdict.py` and is called from scripts, not the normal
   `verify_once` path.

2. Sparse exact-HZ is not in backend.

   Backend dense `HZono` still materializes `Gc/Gb/Ac/Ab` as torch dense
   tensors and has a `cell_budget` drop-to-interval path. The sparse CSR engine
   that avoids representation drop is still implemented in
   `scripts/cifar_sparse_exact_probe.py`.

3. Per-benchmark portfolio configuration is hard-coded in scripts.

   The frozen result depends on benchmark-wide settings such as
   `BENCH_WORKERS`, `MILP_FRACTION`, `COMPRESSED_RELU_DEFAULT`,
   `MILP_ENV_DEFAULTS`, sparse fallback choices, S-curve `K`, and official wall
   handling. These need a backend/config representation so one front-end command
   can reproduce the frozen table.

4. Strict-mode guards are distributed.

   The pure-HZ rule is enforced by a mix of script conventions, ledger filters,
   and solver guards. Stage-II needs a single backend policy object that decides
   what is allowed to count.

5. Exactness/performance tests are not yet in the normal test tree.

   Toy bit-exact checks exist in scripts, but they are not conventional backend
   regression tests. They should become small deterministic tests before major
   refactors.

## Minimal Migration Order

1. Backend configuration module.

   Move benchmark-wide HybridZ profile constants out of `hz_full_driver.py` into
   an ACT backend module. The driver should import that module, preserving the
   frozen behavior while eliminating script-only source of truth.

2. First-class hybridz solver dispatch.

   Add `hybridz` to solver choices and route `verify_once` through:

   - `set_transfer_function_mode("hybridz")`;
   - normal `analyze()`;
   - extract the final HZ object from `HybridzTF`;
   - call `hz_objbound_decide` / sparse equivalent;
   - return `VerifyResult` with CERT/ADV/UNKNOWN semantics.

   Initial acceptance can be B=1 only, because VNN-COMP instances are per-row;
   batched support should be explicit later.

3. Sparse HZ backend package.

   Introduce a backend sparse HZ type and move only the production operators:

   - input box;
   - affine/dense/conv/convtranspose/avgpool/reshape/gather/concat/add;
   - exact ReLU compressed/uncompressed;
   - S-curve compressed graph-domain cuts;
   - final margin row materialization;
   - base-HZ feasibility guard.

   Keep Gurobi diagnostic code out of counted backend paths.

4. Portfolio scheduler.

   Replace `hz_full_driver.py` branch logic with a backend portfolio object that
   runs HiGHS/SCIP/open-source branches under one wall clock, records per-branch
   diagnostics, and counts the whole wall.

5. Test conversion and cleanup.

   Convert stable operator audits into tests:

   - point consistency for affine/spatial ops;
   - dense-HZ vs sparse-HZ toy ReLU bit-close;
   - compressed exact ReLU equivalence;
   - S-curve soundness toy gates;
   - solver verdict toy CERT/ADV/UNKNOWN.

   Once tests pass and a frontend frozen rerun matches, delete or demote scripts.

## Low-Risk Cleanup Items

- Remove Python `__pycache__` directories from the working tree.
- Keep `docs/` and frozen ICSE artifacts; they are part of the handoff.
- Do not delete `scripts/cifar_sparse_exact_probe.py` until sparse backend
  parity is proven.
- Do not enable blanket ReLU cuts or equality substitution as defaults: cGAN
  audits showed both can be representation-negative.
- Keep MaxPool2D exact-only in backend.  The previous convex/DeepZ helper was
  removed from the production HybridZ transfer functions; unsupported
  non-exact MaxPool requests now drop the HZ representation instead of using a
  triangle relaxation.
- Keep sparse operator loading lazy in `HybridzTF`.  The module is cached after
  first use, so normal dense-HZ imports do not eagerly load the sparse backend
  while sparse propagation avoids repeated dynamic imports.

## First Code Step Chosen

Move per-benchmark HybridZ profile constants from `scripts/hz_full_driver.py`
into a backend config module while preserving the driver behavior. This is
small, does not change math, and directly advances the front-end/config
productization requirement.

## Progress In This Pass

- Added `act/back_end/hybridz_config.py` as the backend home for
  benchmark-wide HybridZ profiles.
- Updated `scripts/hz_full_driver.py` to import those profiles instead of
  defining them locally. The script remains a legacy runner, but it is no
  longer the only source of profile truth.
- Added `HybridzTF.get_hz(layer_id)` as the read-only backend boundary needed
  by a future `--solvers hybridz` verdict dispatch.  Later cleanup removed the
  unused `has_hz` convenience wrapper; callers use `get_hz(...) is not None`.
- Verified:
  - `python -m py_compile act/back_end/hybridz_config.py
    act/back_end/hybridz_tf/hybridz_tf.py scripts/hz_full_driver.py`;
  - `python scripts/hz_full_driver.py --help`;
  - representative profile values for safenlp, dist_shift, linearizenn, cGAN,
    metaroom, relusplitter tail, and guided-cora diagnostics.

Additional mainline progress:

- Added `solver="hybridz"` to `BackendConfig` and backend/pipeline CLI help.
- Added `HybridZConfig` under `BackendConfig.hybridz` so strict HybridZ runs
  can declare a benchmark-wide profile and verdict timeout through YAML/CLI
  instead of script-local constants.
- Added backend CLI flags:
  - `--hybridz-bench`;
  - `--hybridz-timeout`.
- Updated pipeline `--verify vnnlib --solvers hybridz` to pass
  `hybridz_bench=<category>` into the backend config. TorchVision uses the
  dataset name as the profile key.
- Added B=1 dense-HZ `verify_once(..., backend_cfg=...)` dispatch:
  - forces `HybridzTF`;
  - extracts the final HZ through `HybridzTF.get_hz`;
  - calls `solver_hz_verdict.hz_objbound_decide`;
  - returns CERT on SAFE;
  - returns UNKNOWN on HZ drop, batched `B>1`, undecided exact MILP, or UNSAFE
    witness until exact input replay/audit is productized.
- Verified:
  - `python -m py_compile act/back_end/config.py act/back_end/verifier.py
    act/back_end/cli.py act/pipeline/cli.py act/back_end/hybridz_config.py`;
  - `BackendConfig.from_yaml(solver="hybridz",
    hybridz_bench="dist_shift_2023")` resolves the profile wall to `300s`;
  - explicit `hybridz_timeout=7.5` overrides the profile wall;
  - `python -m act.back_end --help` exposes `--hybridz-bench` and
    `--hybridz-timeout`;
  - `python -m act.pipeline --help` exposes `--hybridz-timeout` and
    `--solvers hybridz` examples.

Sparse verdict-boundary progress:

- Added `act/back_end/solver/sparse_hz.py` with the backend `SparseHZono`
  carrier:
  - CSR `Gc/Gb/Ac/Ab/Auc/Aub`;
  - exact equality and upper-inequality rows;
  - dense-HZ to sparse-HZ conversion;
  - no benchmark loader, ORT replay, Gurobi diagnostic path, or per-instance
    rescue logic.
- Updated `solver_hz_verdict.py` so `hz_row_max`, `hz_certify_spec`, and
  `hz_objbound_decide` accept either dense `HZono` or `SparseHZono`.
- Added sparse/dense parity self-test in `solver_hz_verdict.py`:
  - LP row max parity;
  - CERT margin parity;
  - HiGHS SAFE parity when highspy is installed;
  - HiGHS binary UNSAFE witness parity when highspy is installed.
- Wired `BackendConfig.hybridz.engine="sparse_hz_objbound"` to use the backend
  CSR carrier before the final verdict.  The path now prefers a propagated
  sparse HZ from `HybridzTF.get_sparse_hz(...)`; when that is unavailable but
  dense HZ survived, it falls back to dense-HZ to sparse-HZ conversion for
  compatibility.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/verifier.py act/back_end/config.py
    act/back_end/solver/sparse_hz.py act/back_end/solver/solver_hz_verdict.py
    act/back_end/cli.py act/pipeline/cli.py`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `BackendConfig.from_yaml(..., hybridz_engine="sparse_hz_objbound")`;
  - `git diff --check`.

Sparse propagation primitive progress:

- Added `act/back_end/hybridz_tf/sparse_ops.py` with production-safe sparse HZ
  structural/affine operators extracted from the prototype:
  - `sparse_hz_from_bounds`;
  - `sparse_hz_linear`;
  - `sparse_dense_matrix_from_layer`;
  - `sparse_conv2d_matrix_from_layer`;
  - `sparse_convtranspose2d_matrix_from_layer`;
  - `sparse_avgpool2d_matrix_from_layer`;
  - `sparse_hz_apply_dense_layer`;
  - `sparse_hz_apply_conv2d_layer`;
  - `sparse_hz_apply_convtranspose2d_layer`;
  - `sparse_hz_apply_avgpool2d_layer`;
  - `sparse_hz_apply_matmul_const_layer`;
  - `sparse_hz_apply_maxpool2d_layer`;
  - `sparse_hz_apply_relu_exact`;
  - `sparse_hz_apply_sigmoid_piecewise`;
  - `sparse_hz_apply_tanh_piecewise`;
  - `sparse_hz_add_const`;
  - `sparse_hz_scale`;
  - `sparse_hz_gather_rows`;
  - `sparse_hz_concat`;
  - `sparse_hz_add_same_frame`;
  - `sparse_hz_sub_same_frame`;
  - frame padding and zero-row helpers.
- These operators assume a single global generator frame: shared factors keep
  the same column index across branches.  This is the sparse-carry invariant
  needed for residual/concat without dense materialization.
- `sparse_hz_apply_relu_exact` implements the same exact eq_lagr semantics as
  the dense path, including the compressed exact formulation and optional
  redundant valid cuts.  It does not introduce triangle relaxation.
- `sparse_hz_apply_sigmoid_piecewise` and `sparse_hz_apply_tanh_piecewise`
  implement the dense default compressed/pruned S-curve encoding: zero-width
  inflection-side segments are deleted and local segment boxes are represented
  by exact inequality rows rather than slack equality columns.
- Deliberately not included yet:
  - S-curve exact-valid domain/range/graph cuts;
  - sparse var-var MatMul product-envelope handling;
  - sparse Softmax simplex/product-envelope handling;
  - sparse redundancy removal after each nonlinear sparse op;
  - full `HybridzTF` sparse state cache coverage for every operator family.
- Added a module self-test comparing sparse ops to dense HZ support functions:
  - box support;
  - affine support;
  - gather support;
  - same-frame add support;
  - concat with correlated duplicated rows.
  - Conv2D support;
  - AvgPool2D support.
  - constant-side MatMul support, both variable-left and variable-right;
  - ConvTranspose2D support;
  - MaxPool2D one-hot upper/lower support against analytically computed window
    bounds;
  - exact ReLU support, both uncompressed and compressed;
  - compressed exact ReLU with valid cuts preserves the support function on the
    tested rows.
  - single-layer compressed/pruned sigmoid and tanh support parity against the
    dense default encoding.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/hybridz_tf/sparse_ops.py
    act/back_end/solver/sparse_hz.py act/back_end/solver/solver_hz_verdict.py`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `git diff --check`.

Sparse HybridzTF cache progress:

- Added an explicit `HybridzTF.enable_sparse_hz(...)` switch.  Sparse cache is
  opt-in, so ordinary dense `--tf-mode hybridz` runs do not pay the CSR
  construction cost.
- Added `HybridzTF.get_sparse_hz(layer_id)` and sparse drop-reason metadata.
- Added conservative sparse side propagation for:
  - input box seeding and floating roots that read the same input box;
  - Dense/Bias/Scale/BatchNorm;
  - Conv2D/ConvTranspose2D/AvgPool2D;
  - MaxPool2D via exact pairwise `b + ReLU(a-b)` folds;
  - exact ReLU with compressed/valid-cut flags;
  - Sigmoid/Tanh via compressed/pruned S-curve piecewise HZ;
  - Flatten/Reshape/Squeeze/Unsqueeze/Transpose pass-through;
  - Slice/Gather/Expand/nearest Upsample row gathers;
  - Add/Sub/Concat over the same global generator frame;
  - Constant layers as zero-radius sparse boxes.
- Updated `verify_once(..., backend_cfg=...)` so
  `hybridz.engine in {"sparse_hz", "sparse_hz_objbound"}` enables the sparse
  cache for B=1 runs and prefers the propagated CSR HZ at the final output
  layer.  Metadata records whether the verdict used `sparse_source=propagated`
  or `sparse_source=dense_conversion`.
- Added smoke coverage executed from the shell:
  - `HybridzTF.enable_sparse_hz(True)` on an Input->Dense->ReLU->Dense toy,
    comparing dense-HZ and sparse-HZ support values at the final layer;
  - `HybridzTF.enable_sparse_hz(True)` on an Input->MaxPool2D toy, comparing
    sparse one-hot support values to exact window bounds;
  - dense `hz_maxpool2d` point consistency plus an exact-only guard showing
    `exact=False` returns `None` instead of a triangle-relaxed HZ;
  - `HybridzTF.enable_sparse_hz(True)` on one-layer Input->Sigmoid and
    Input->Tanh toys, comparing dense-HZ and sparse-HZ support values at the
    nonlinear output;
  - `HybridzTF.enable_sparse_hz(True)` on the ACT `layer_testing_matmul`
    fixture, confirming constant-side MatMul stays in the propagated sparse
    cache and matches dense-HZ support values;
  - `verify_once` with `BackendConfig.from_yaml(solver="hybridz",
    hybridz_engine="sparse_hz_objbound")`, confirming the sparse verdict path
    used `sparse_source=propagated`.

HybridZ witness replay gate progress:

- Added optional input replay metadata to backend `SparseHZono`:
  `input_center`, `input_radius`, `input_indices`, and `input_shape`.  These
  fields are solver-ignored metadata; they do not change the represented HZ set.
- Seeded sparse input boxes with replay metadata and taught `HybridzTF` to carry
  that metadata through sparse side-cache propagation, including multi-input
  Add/Sub/Concat when at least one input branch has replay metadata.
- Added conservative backend UNSAFE promotion in `verify_once`:
  - exact HZ MILP returns an unsafe `xi` witness;
  - backend reconstructs a concrete input from dense `col_ids` or sparse input
    metadata;
  - a caller-provided `model_fn` must replay that input and violate the encoded
    output spec;
  - only then does HybridZ return `VerifyStatus.FALSIFIED`.
- If the model replay function is missing, metadata is missing, the witness
  cannot be mapped back to input, or replay does not violate the spec, the
  result remains UNKNOWN.  This preserves the pure-HZ rule and avoids counting
  LP or over-approximation witnesses as ADV.
- Updated the pipeline VNNLIB/TorchVision normal path so `--solvers hybridz`
  passes the raw PyTorch model as replay audit function.  Other solvers are not
  changed.
- Added `verifier` self-test coverage:
  - dense `dense_hz_objbound` exact-HZ UNSAFE without replay remains UNKNOWN;
  - dense exact-HZ UNSAFE with replay becomes FALSIFIED;
  - sparse `sparse_hz_objbound` follows the same gate.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/solver/sparse_hz.py
    act/back_end/hybridz_tf/sparse_ops.py act/back_end/hybridz_tf/hybridz_tf.py
    act/back_end/verifier.py act/pipeline/cli.py
    act/back_end/solver/solver_hz_verdict.py act/back_end/config.py
    act/back_end/cli.py`;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `git diff --check`.

Base-HZ feasibility guard progress:

- Added `hz_base_feasibility(...)` to `solver_hz_verdict.py`.  The guard checks
  that the propagated HZ state itself is nonempty before any SAFE/UNSAFE verdict
  is accepted by the backend objective-bound path.
- The backend guard is stricter than the original sparse script LP guard: binary
  HZ variables are mapped through `xi_b = 2z - 1` and checked as integer
  `z in {0,1}`.  This prevents a binary-empty HZ from producing vacuous CERT.
- `hz_objbound_decide(...)` now requires base feasibility by default and returns
  UNKNOWN if the base HZ is infeasible or cannot be established feasible.
- `hz_certify_spec(...)` also refuses to certify when base feasibility is not
  established, preserving the older API's soundness.
- `verify_once(..., solver="hybridz")` records `hz_base_feasible` and
  `hz_base_feas_msg` in metadata and returns UNKNOWN before the spec verdict
  if the base guard does not prove FEASIBLE.
- Added solver regression coverage:
  - feasible dense/sparse HZ passes the guard;
  - constant-row empty HZ is INFEASIBLE and cannot certify;
  - binary-only empty HZ is INFEASIBLE even though its LP relaxation is feasible;
  - `hz_objbound_decide` returns UNKNOWN rather than SAFE on empty HZ states.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/solver/solver_hz_verdict.py
    act/back_end/verifier.py`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.verifier`.

Base-HZ constructive witness precheck progress:

- Added `hz_base_witness(...)` to expose a concrete feasible base-HZ `xi`
  point produced by the base feasibility solve.
- `hz_objbound_decide(...)` now does a default exact-HZ precheck:
  - first prove the base HZ is FEASIBLE;
  - retrieve the feasible base `xi`;
  - if that exact HZ point already lies in the unsafe output spec, return it as
    an UNSAFE witness before launching the objective-bound MILP.
- The precheck is not sampling and not LP-witness promotion.  The `xi` point
  satisfies the exact HZ base constraints, including binary variables checked
  as integers through `xi_b = 2z - 1`.
- `verify_once(..., solver="hybridz")` records `hz_witness_source`, so later
  reporting can distinguish `base_hz_witness`, `bare_point`, and
  `milp_objective_bound`.
- If a base witness fails concrete replay, `verify_once` does not immediately
  stop.  It reruns `hz_objbound_decide(..., base_witness_precheck=False)` so a
  different exact MILP witness can still be found and replayed, matching the
  script-side conservative behavior.
- Added solver/verifier regression coverage:
  - base witness exists and is the deterministic zero point for an
    unconstrained box HZ;
  - a boundary-unsafe spec returns `UNSAFE` with that base witness;
  - HybridZ replay-gate metadata records `hz_witness_source=base_hz_witness`.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/solver/solver_hz_verdict.py
    act/back_end/verifier.py`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.verifier`.

ONNX/ORT HybridZ witness replay adapter progress:

- VNNLIB loader now records original ONNX metadata on converted PyTorch models:
  `_act_onnx_path`, `_act_onnx_model`, and `_act_onnx_input_shape`.
- `model_synthesis.py` copies these metadata fields onto the synthesized
  `VerifiableModel`, so the normal ACT frontend path preserves the original
  ONNX artifact through wrapping.
- `act.pipeline --verify vnnlib --solvers hybridz` now builds a HybridZ replay
  function that prefers ONNXRuntime for VNNLIB-origin models:
  - lazy-creates an ORT session only when a HybridZ exact witness needs replay;
  - feeds the reconstructed witness input directly to the original ONNX model;
  - returns the raw output tensor to `verify_once` for the existing output-spec
    violation check.
- If ORT is unavailable or a particular ONNX replay fails, the adapter falls
  back to the converted PyTorch model.  A failed replay still cannot produce an
  ADV; `verify_once` keeps the existing conservative UNKNOWN behavior unless
  the model output violates the original spec.
- This keeps ORT as audit/replay only.  The proof remains the pure HybridZ
  exact-HZ engine; ORT time is not part of the HybridZ verdict solve.
- Added a smoke check that:
  - synthesizes a wrapper model with fake ONNX metadata;
  - verifies metadata survives synthesis;
  - verifies the replay helper falls back to PyTorch if ORT cannot load the
    fake ONNX path.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/front_end/vnnlib_loader/data_model_loader.py
    act/front_end/model_synthesis.py act/pipeline/cli.py act/back_end/verifier.py`;
  - metadata/fallback smoke snippet;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `python -m act.back_end.verifier`.

Immediate next engineering step: migrate the remaining sparse exact-HZ
production capabilities from `scripts/cifar_sparse_exact_probe.py`:
S-curve graph cuts and the open-source portfolio scheduler.
Those are required before the normal frontend can reproduce frozen ADV counts
and all large sparse benchmarks without relying on scripts.

S-curve future-work hook progress:

- Migrated the sparse S-curve domain/range and conditional graph-cut helpers
  into `act/back_end/hybridz_tf/sparse_ops.py`.
- The cuts are optional and default to `False`, so the frozen strict-HybridZ
  result path is unchanged.
- `HybridzTF` now carries default-disabled `_scurve_domain_cuts` and
  `_scurve_graph_cuts` flags for future benchmark-wide experiments.
- Added sparse S-curve regression coverage:
  - default sigmoid/tanh sparse encoding remains dense-HZ parity checked;
  - enabling cuts keeps the base HZ feasible and does not expand tested
    support rows.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/hybridz_tf/sparse_ops.py
    act/back_end/hybridz_tf/hybridz_tf.py act/back_end/solver/solver_hz_verdict.py
    act/back_end/verifier.py`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.verifier`;
  - `git diff --check`.

Frontend HybridZ profile application progress:

- `verify_once(..., solver="hybridz")` now applies benchmark-wide profile
  fields from `act.back_end.hybridz_config` during the normal frontend path.
- Forward propagation knobs are scoped to the single analyze call and then
  restored:
  - compressed exact ReLU;
  - exact ReLU valid cuts;
  - sigmoid `K`;
  - dense-HZ cell budget.
- Open-source MILP environment knobs are scoped to the exact verdict calls and
  then restored:
  - `HZ_MILP_CUTOFF_ROW`;
  - `HZ_MILP_THREADS`;
  - per-benchmark `HZ_MILP_*` entries from `MILP_ENV_DEFAULTS`.
- Profile timeout now honors benchmark-wide MILP fraction/cap when the user has
  not supplied an explicit `--hybridz-timeout`.
- Added verifier regression coverage that checks:
  - profile metadata is reported on HybridZ results;
  - cutoff-row env vars do not leak after verification;
  - temporary HybridZ TF attributes are restored.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/verifier.py
    act/back_end/config.py act/back_end/hybridz_config.py`;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`.

Frontend HybridZ CSV reporting progress:

- Added `act/pipeline/hybridz_results.py` as the mainline per-row and summary
  CSV reporter for strict HybridZ frontend runs.
- Added `--hybridz-results-dir` to `act.pipeline`.
- `act.pipeline --verify vnnlib --solvers hybridz` and
  `act.pipeline --verify torchvision --solvers hybridz` now optionally write:
  - `<bench>_hybridz_detail.csv`;
  - `<bench>_hybridz_summary.csv`.
- The summary schema follows the frozen artifact shape:
  `Bench,N,CERT,ADV,V+A,TIMEOUT,UNKNOWN,ERROR,P0,unsolved`.
  `P0` remains `0` in this reporter; concrete soundness audit remains a
  separate validation step.
- The reporter self-test now covers CERT/ADV/TIMEOUT/UNKNOWN/ERROR status
  mapping, metadata lane override, `unsolved=TIMEOUT+UNKNOWN+ERROR`,
  `P0=0`, and detail/summary CSV round-trip parsing.
- Verified a frontend smoke command:
  - `python -m act.pipeline --verify vnnlib --category acasxu_2023
    --max-instances 1 --solvers hybridz --hybridz-timeout 3
    --hybridz-results-dir /tmp/act_hybridz_results_smoke`;
  - produced detail and summary CSVs through the normal frontend path.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/pipeline/hybridz_results.py act/pipeline/cli.py`;
  - `python -m act.pipeline.hybridz_results`;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`.

Mainline HybridZ benchmark runner progress:

- Fixed VNNLIB instance enumeration so `instances.csv` files without a header
  no longer lose row 0.  The ACASXu local set now enumerates all `186`
  instances and `iid0` is `ACASXU_run2a_1_1_batch_2000_prop_1`.
- Added zero-based `instance_indices` support to
  `VNNLibSpecCreator.create_specs_for_data_model_pairs(...)`.
- Added `--instance-index` to `act.pipeline --verify vnnlib`, so a normal
  frontend command can run one official instances.csv row exactly.
- Added `act/pipeline/hybridz_benchmark_runner.py`:
  - lists official VNNLIB instances through the frontend loader;
  - runs each iid through `python -m act.pipeline --verify vnnlib
    --instance-index <iid> --solvers hybridz`;
  - applies official per-row wall/cap plus the HybridZ bench profile's MILP
    fraction/cap;
  - supports bounded worker parallelism;
  - aggregates per-iid frontend CSVs into benchmark detail/summary CSVs.
- Added `--verify hybridz-benchmark` plus `--hybridz-workers` and
  `--hybridz-timeout-cap` to `act.pipeline`.
- This is the mainline replacement skeleton for `hz_full_driver.py`; it still
  lacks the legacy script's sparse-first/fallback and specialized portfolio
  branches, so frozen full-table reproduction is not yet complete.
- Verified a frontend runner smoke command:
  - `python -m act.pipeline --verify hybridz-benchmark --category acasxu_2023
    --max-instances 1 --hybridz-workers 1 --hybridz-timeout-cap 5
    --hybridz-results-dir /tmp/act_hybridz_bench_smoke`;
  - produced aggregate detail and summary CSVs through the normal ACT frontend
    path without calling `scripts/hz_full_worker.py`.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/front_end/vnnlib_loader/data_model_loader.py
    act/front_end/vnnlib_loader/create_specs.py act/pipeline/cli.py
    act/pipeline/hybridz_benchmark_runner.py act/pipeline/hybridz_results.py`;
  - ACASXu loader count/index smoke (`186`, row 0 = `1_1`);
  - `python -m act.pipeline.hybridz_benchmark_runner`;
  - `python -m act.pipeline.hybridz_results`;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`.

Mainline exact-formulation portfolio progress:

- Added a benchmark-profile-driven branch plan inside
  `act/pipeline/hybridz_benchmark_runner.py`.
- Default branch is `normal`.
- If `HybridZBenchProfile.parallel_cutoff_portfolio` is enabled, the runner
  launches both:
  - `normal`;
  - `cutrow` with `HZ_MILP_CUTOFF_ROW=1`.
- Both branches call the normal ACT frontend path
  `python -m act.pipeline --verify vnnlib --solvers hybridz`; no legacy worker
  script is invoked.
- The first branch that returns `CERT` or `ADV` becomes the iid winner; if none
  is conclusive, the normal branch diagnostic is retained.
- This initially migrated only the open-source exact-MILP formulation
  portfolio skeleton used by `safenlp_2024`; the full frozen safenlp row also
  used sparse-compressed and projected one-ReLU exact-HZ branches.  The
  2026-06-24 full frontend frozen attempt exposed that gap:
  `safenlp_2024` reproduced only `431 CERT + 624 ADV = 1055` through
  `normal/cutrow`, versus the frozen row `432 CERT + 647 ADV = 1079`.
- The runner now carries a transitional fixed safenlp portfolio bridge for the
  remaining exact-HZ branch families:
  `normal_pscost1`, `sparse_comprelu`, `sparse_comprelu_heur`,
  `projected_relu_mip`, `projected_relu_scip-bigm`, and
  `projected_relu_scip-indicator`.  The bridge is benchmark-wide and pure HZ.
  Sparse branches now use the normal ACT frontend with
  `--hybridz-engine sparse_hz_objbound`; projected one-ReLU exact-HZ branches
  now run through `python -m act.pipeline.hybridz_projected_relu_mip`.  No
  `scripts/hz_sparse_worker.py` or `scripts/hz_projected_relu_mip.py` process
  is invoked by the mainline runner.
- Focused gap probe over the 24 safenlp rows missed by the first frontend run
  recovered the projected/sparse ADV tail.  The recovered ADV rows are
  `169, 243, 335, 343, 381, 396, 439, 464, 468, 481, 492, 498, 610, 648,
  704, 769, 817, 885, 886, 950, 952, 953, 1006`.
- The remaining focused parity blocker, `iid844`, is frozen as `CERT` via the
  historical `normal_pscost1` branch.  A 2026-06-24 package-frontend recheck
  with the current ACT entrypoint,
  `HZ_HIGHS_OPTIONS=mip_pscost_minreliable=1 python -m act.pipeline --verify
  vnnlib --category safenlp_2024 --instance-index 844 --solvers hybridz
  --hybridz-timeout 19`, now returns `CERT` with detail
  `wall_s=16.5278377532959`, `hz_timeout_s=19.0`, and profile MILP env
  `{"HZ_MILP_ELIM_SINGLETONS": "1"}`.  A runner-level one-iid check through
  `_run_one(HybridZBenchmarkConfig(...), iid844)` also selects
  `normal_pscost1` as the winning branch and writes verifier detail
  `wall_s=16.815699100494385`; its outer portfolio wall is about `21.05s` and
  is audit/orchestration time, not the counted HybridZ verifier time.
- Net status after this patch: the focused safenlp frontend parity checks now
  cover the frozen row `432 CERT + 647 ADV = 1079`, with the expected single
  unresolved row remaining `iid454`.  A full
  `--hybridz-require-frozen-match` frozen-suite rerun is still pending before
  declaring end-to-end acceptance.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/pipeline/hybridz_benchmark_runner.py
    act/pipeline/cli.py act/pipeline/hybridz_projected_relu_mip.py
    act/pipeline/hybridz_projected_utils.py`;
  - `python -m act.pipeline.hybridz_benchmark_runner`;
  - representative projected-module ADV checks:
    `safenlp_2024 iid243` via HiGHS projected exact-HZ and `iid439` via
    SCIP big-M projected exact-HZ, both with witness replay passing;
  - repeated 24-gap safenlp module probe for projected/sparse ADV recovery;
  - current-package `safenlp_2024 iid844 normal_pscost1` frontend recheck:
    `CERT`, `wall_s=16.5278377532959`, `hz_timeout_s=19.0`;
  - current-package runner-level `safenlp_2024 iid844` portfolio recheck:
    winner `normal_pscost1`, `CERT`, verifier-detail `wall_s=16.815699100494385`;
  - branch-plan smoke:
    `acasxu_2023 -> [normal, sparse_scip_witness]`,
    `safenlp_2024 -> [normal, cutrow, normal_pscost1,
    sparse_comprelu, sparse_comprelu_heur, projected_relu_mip,
    projected_relu_scip-bigm, projected_relu_scip-indicator]`;
  - ACASXu one-iid `hybridz-benchmark` smoke;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `git diff --check`.

Mainline sparse-engine branch progress:

- Added `--hybridz-engine {dense_hz_objbound,sparse_hz_objbound}` to
  `act.pipeline` and pass it through `BackendConfig.hybridz.engine`.
- Added `sparse_hz_objbound` branches to
  `act/pipeline/hybridz_benchmark_runner.py` from `HybridZBenchProfile`:
  - `sparse_first=True` -> sequential `sparse -> normal`;
  - `sparse_fallback=True` -> sequential `normal -> sparse`.
- Sparse branches use the normal frontend command with
  `--hybridz-engine sparse_hz_objbound`; they do not invoke
  `scripts/hz_sparse_worker.py`.
- Sequential sparse portfolios share the same per-iid deadline to avoid
  multiplying wall time and resource use.  The exact normal/cutrow formulation
  portfolio remains parallel because it is the same dense propagated HZ with
  different exact MILP formulations.
- Current branch-plan smoke:
  - `acasxu_2023 -> [normal, sparse_scip_witness]`;
  - `safenlp_2024 -> [normal, cutrow, normal_pscost1,
    sparse_comprelu, sparse_comprelu_heur, projected_relu_mip,
    projected_relu_scip-bigm, projected_relu_scip-indicator]`;
  - `metaroom_2023 -> [sparse, normal]`;
  - `relusplitter -> [normal, sparse]`.
- Verified a direct sparse frontend smoke:
  - `python -m act.pipeline --verify vnnlib --category acasxu_2023
    --instance-index 0 --solvers hybridz --hybridz-engine sparse_hz_objbound
    --hybridz-timeout 3 --hybridz-results-dir
    /tmp/act_hybridz_sparse_engine_smoke`;
  - detail CSV reports `engine=sparse_hz_objbound` and
    `sparse_source=propagated`.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/pipeline/cli.py
    act/pipeline/hybridz_benchmark_runner.py`;
  - `python -m act.pipeline.hybridz_benchmark_runner`;
  - ACASXu one-iid `hybridz-benchmark` smoke;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `git diff --check`.

Mainline S-curve branch configuration progress:

- Extended `BackendConfig.hybridz` with explicit override fields:
  - `sigmoid_k`;
  - `tanh_k`;
  - `scurve_domain_cuts`;
  - `scurve_graph_cuts`;
  - `compressed_relu`;
  - `relu_valid_cuts`;
  - `cell_budget`.
- Added frontend flags:
  - `--hybridz-sigmoid-k`;
  - `--hybridz-tanh-k`;
  - `--hybridz-scurve-domain-cuts`;
  - `--hybridz-scurve-graph-cuts`;
  - `--hybridz-compressed-relu`;
  - `--hybridz-relu-valid-cuts`;
  - `--hybridz-cell-budget`.
- `verify_once` now applies profile defaults first, then explicit config/CLI
  overrides, and records the override metadata as `cfg_*` fields.
- Added dist_shift sparse S-curve SCIP branches to the mainline runner:
  - `sparse_scurve_k2_scip`;
  - `sparse_scurve_k4_scip`;
  - `sparse_scurve_k6_scip`;
  - `sparse_scurve_k4_cutrow_scip`;
  - `sparse_scurve_k8_scip`.
- These branches use the normal frontend path with
  `--hybridz-engine sparse_hz_objbound`, deterministic `sigmoid_k`, `tanh_k=1`,
  and default-off exact-valid S-curve domain/graph cuts enabled.  The cutrow
  branch also sets `HZ_MILP_CUTOFF_ROW=1`.
- Verified a direct frontend smoke:
  - `python -m act.pipeline --verify vnnlib --category acasxu_2023
    --instance-index 0 --solvers hybridz --hybridz-engine sparse_hz_objbound
    --hybridz-sigmoid-k 4 --hybridz-tanh-k 1
    --hybridz-scurve-domain-cuts --hybridz-scurve-graph-cuts
    --hybridz-timeout 3 --hybridz-results-dir
    /tmp/act_hybridz_scurve_cfg_smoke`;
  - detail CSV reports `cfg_sigmoid_k=4`, `cfg_tanh_k=1`,
    `cfg_scurve_domain_cuts=true`, and `cfg_scurve_graph_cuts=true`.
- Verified under `/data1/Kane/miniconda3/envs/act-py312/bin/python`:
  - `python -m py_compile act/back_end/config.py act/back_end/verifier.py
    act/pipeline/cli.py act/pipeline/hybridz_benchmark_runner.py`;
  - `python -m act.pipeline.hybridz_benchmark_runner`;
  - config parsing smoke for S-curve overrides;
  - ACASXu one-iid `hybridz-benchmark` smoke;
  - `python -m act.back_end.verifier`;
  - `python -m act.back_end.solver.solver_hz_verdict`;
  - `python -m act.back_end.hybridz_tf.sparse_ops`;
  - `git diff --check`.

Mainline ACASXu sparse-SCIP witness branch progress:

- Added direct `act.pipeline` and `act.back_end` CLI overrides for the exact
  ReLU/carry knobs that were already present in `BackendConfig.hybridz`:
  `compressed_relu`, `relu_valid_cuts`, and `cell_budget`.
- Extended `HybridZRunBranch` with:
  - `timeout_override_s`, so short solver-slice branches can be scheduled
    without changing the official per-row wall;
  - `accept_verdicts`, so diagnostic/fallback branches can be ADV-only when
    that is the pure-HZ counted behavior.
- Added the benchmark-wide `acasxu_2023` `sparse_scip_witness` branch:
  - normal branch still runs first;
  - fallback uses the normal frontend path with
    `--hybridz-engine sparse_hz_objbound --hybridz-compressed-relu`;
  - open-source SCIP is selected by `HZ_MILP_BACKEND=scip`;
  - branch solver timeout is
    `ACASXU_SCIP_WITNESS_MILP_TIMEOUT`;
  - only `ADV` is accepted from this fallback branch.
- This migrates the sound part of
  `scripts/hz_full_driver.py::_run_acasxu_scip_witness_fallback` into the
  mainline runner.  The more experimental ACASXu cuts/FBBT fallback remains
  script-only because the backend still lacks the corresponding sparse-probe
  options (`relu_cuts`, FBBT passes, relax-precheck budget, and base-binary
  MIP start) as first-class, audited exact-HZ configuration.

Mainline per-query objective parallelism progress:

- Migrated the script-side `HZ_QUERY_WORKERS` mechanism into
  `solver_hz_verdict.hz_objbound_decide` for the `is_unsafe_linear=False`
  case.  These rows are independent top1/row objective-bound queries over the
  same propagated HZ; this is not input split and does not change the HZ set.
- `UNSAFE_LINEAR` remains a single joint epigraph MILP, because those rows are
  conjunctive and cannot be split without changing the problem.
- `verify_once` now exports `HybridZBenchProfile.query_workers` through
  `HZ_QUERY_WORKERS` while the exact HybridZ solver runs, and records
  `profile_query_workers` in metadata.
- Added a solver self-test that compares serial and `HZ_QUERY_WORKERS=2`
  verdicts on a two-row exact HZ query.

Mainline frozen-suite runner progress:

- Added `FROZEN_BENCHMARK_SUITE` to `act/back_end/hybridz_config.py` with the
  frozen-report benchmark order:
  `safenlp_2024`, `metaroom_2023`, `sat_relu`, `malbeware`, `cersyve`,
  `acasxu_2023`, `linearizenn_2024`, `dist_shift_2023`,
  `tllverifybench_2023`, `cora_2024`, `relusplitter`, and `cgan_2023`.
- Moved frozen acceptance counts, frozen summary fields, tool names, and
  cross-tool comparison counts into `act/back_end/hybridz_config.py`, so
  `act/pipeline/hybridz_benchmark_runner.py` owns execution/export logic rather
  than carrying large report tables.
- Added `python -m act.back_end.hybridz_config` self-test for frozen-suite
  metadata invariants: benchmark order/counts alignment, `V+A=CERT+ADV`,
  `unsolved=N-(V+A)`, suite headline, `P0=0`, complete competitor tool coverage,
  and no Gurobi backend selection in counted benchmark profiles.  The current
  soundfix headline is `2213/977 CERT/786 ADV/1763 V+A`; the older
  `2213/983 CERT/785 ADV/1768 V+A` headline is retained only as historical
  provenance.
- `python -m act.back_end.hybridz_config
  /data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/FINAL_HYBRIDZ_RESULTS_20260624.csv`
  now validates that the backend frozen oracle matches the saved ICSE frozen
  CSV, including benchmark order.  This caught and fixed an order-only drift
  where `dist_shift_2023` and `linearizenn_2024` were swapped in the runner
  suite tuple while all numeric fields still matched.
- Added suite-level runner support in
  `act/pipeline/hybridz_benchmark_runner.py`:
  - `resolve_hybridz_benchmark_categories("frozen")`;
  - `HybridZBenchmarkSuiteConfig`;
  - `run_hybridz_benchmark_suite`;
  - `hybridz_suite_detail.csv`;
  - `hybridz_suite_summary.csv` with a `TOTAL` row.
- `act.pipeline --verify hybridz-benchmark --category frozen` now dispatches
  the suite through normal frontend `--solvers hybridz` invocations.  The suite
  itself is conservative and sequential across benchmarks; each benchmark keeps
  its own profile workers/portfolio internally.
- This is a reporting/productization bridge.  The safenlp sparse/projected
  bridge branches have been migrated into ACT package code, and the focused
  `iid844` `normal_pscost1` parity check now passes inside the 19s engine wall.
  Full frozen-table acceptance still requires a clean complete frozen-suite
  rerun through `--hybridz-require-frozen-match`.

Mainline ICSE CSV export progress:

- Added ICSE/VNN-COMP style CSV export directly to
  `act/pipeline/hybridz_benchmark_runner.py`.
- Every mainline benchmark run now writes, in addition to the internal
  HybridZ detail/summary CSVs:
  - `{bench}.csv` with `onnx,vnnlib,result,time_sec`;
  - `{bench}_icse_index.csv`;
  - `{bench}_icse_detail.csv`.
- Suite runs now merge these into root-level:
  - one `{bench}.csv` per benchmark;
  - `_INDEX.csv`;
  - `_DETAIL.csv`;
  - `README_REPRODUCIBILITY.md`.
- Result token mapping follows the frozen ICSE artifact convention:
  `CERT -> unsat`, `ADV -> sat`, `TIMEOUT -> timeout`,
  `UNKNOWN -> unknown`, and errors to `error`.
- The exported `time_sec` uses the verifier detail-row `wall_s` when present,
  so it tracks the ACT-HybridZ verification call rather than the outer
  subprocess/model-loading wall.  If detail timing is unavailable, it falls back
  to branch wall time.
- This migrates the live-run reporting part of `scripts/hz_export_icse_csv.py`
  into the mainline runner.
- Suite runs now also write cross-tool comparison files using the frozen
  competitor baselines and the current run's HybridZ counts:
  - `FINAL_HYBRIDZ_RESULTS.csv`;
  - `FINAL_CROSS_TOOL_RANKING.csv`;
  - `_CROSS_TOOL_SUMMARY.csv`.
- Suite runs now also write conservative live-run failure taxonomy files:
  - `failure_taxonomy_detail.csv`;
  - `failure_taxonomy_summary.csv`.
- Full frozen-suite runs now write current-vs-frozen reproduction checks:
  - `FROZEN_REPRO_COMPARISON.csv`;
  - `FROZEN_REPRO_COMPARISON.json`.
  These files compare the current frontend-run counts against
  `FINAL_HYBRIDZ_RESULTS_20260624.csv`; they do not feed the verifier.
- Added the opt-in CLI gate `--hybridz-require-frozen-match` for
  `--verify hybridz-benchmark --category frozen`.  The command exits as failed
  after writing artifacts if the current summary does not exactly match the
  frozen table.
- Historical ledger selection and deeper sparse/probe taxonomy fields remain
  script/artifact tooling for now.

Mainline manifest / JSON summary progress:

- Added deterministic SHA256 manifest generation to
  `act/pipeline/hybridz_benchmark_runner.py`.
- Every mainline benchmark output directory now writes:
  - `{bench}_run_summary.json`;
  - `_MANIFEST.sha256`.
- Suite output directories now write:
  - `hybridz_suite_summary.json`;
  - `FROZEN_REPRO_COMPARISON.csv` and `.json` for full frozen-suite runs;
  - `failure_taxonomy_detail.csv`;
  - `failure_taxonomy_summary.csv`;
  - root `_MANIFEST.sha256`.
- Manifest entries use paths relative to the output root so the produced
  artifact can be moved without invalidating path strings.  This differs from
  the older frozen artifact, whose manifest used absolute paths, but keeps the
  same checksum purpose.
- Runner self-tests now validate benchmark JSON/manifest output and suite
  manifest inclusion, cross-tool reporting, failure taxonomy, and the frozen
  match gate.

## Current Backend/Script Gap Matrix

This matrix is the current productization boundary after the profile-application
work above.  It should be updated whenever a script-only branch is migrated or
retired.

Diff boundary note: `HYZOR_STAGE2_DIFF_BOUNDARY_AUDIT_20260624.md` records the
current `upstream/main` baseline, the tracked/untracked Stage-II footprint, and
the non-HybridZ files that need explicit justification before commit.

| Capability | Backend status | Script-only residue | Next action |
|---|---|---|---|
| Benchmark-wide strict-HZ profiles | `act/back_end/hybridz_config.py` is the source of truth; `verify_once` applies forward knobs, MILP env knobs, query-workers, and profile-derived timeout/fraction/cap; explicit CLI/config overrides cover S-curve K/cuts, exact ReLU/carry knobs, and cell budget; `hybridz-benchmark` applies official per-row wall/cap, normal/cutrow exact formulation portfolio, safenlp sparse/projected exact-HZ portfolio bridge through package code, sequential sparse-first/fallback branch order, dist_shift S-curve sparse branches (`k=2/4/6/8` plus `k=4` cutrow), ACASXu sparse-SCIP ADV-only fallback, per-instance `RLIMIT_AS` caps via `ACT_HYBRIDZ_RLIMIT_AS_GB`, host free-RAM launch pausing via `HybridZBenchProfile.mem_floor_gb` / `ACT_HYBRIDZ_MEM_FLOOR_GB`, and frozen-suite ordering | `hz_full_driver.py` remains a legacy reproduction runner; ACASXu cuts/FBBT remains script-only | Run a clean complete frozen-suite acceptance pass |
| First-class frontend solver mode | `--solvers hybridz` routes through `verify_once`, exact HZ verdict, base-HZ guard, and ORT audit-only replay; `--hybridz-engine` selects dense or sparse exact-HZ verdict; S-curve/relu/cell-budget overrides are frontend-declarable; `--verify hybridz-benchmark` provides a mainline per-iid runner, exact normal/cutrow portfolio, safenlp sparse/projected bridge branches, sparse-first/fallback branches, dist_shift S-curve branches (`k=2/4/6/8` plus `k=4` cutrow), ACASXu sparse-SCIP ADV-only fallback, frozen-suite dispatch via `--category frozen`, aggregate CSVs, ICSE-style export files, and the opt-in `--hybridz-require-frozen-match` acceptance gate | Focused safenlp parity blocker `iid844` now passes through the package frontend; full frozen reproduction has not yet been rerun end to end | Keep frozen expected counts immutable; require a clean full rerun before declaring a new freeze |
| Sparse CSR exact-HZ representation | `SparseHZono` and core sparse ops live under `act/back_end/solver` and `act/back_end/hybridz_tf` | `cifar_sparse_exact_probe.py` still carries exploratory propagation code, extra diagnostics, FBBT/OBBT prototypes, save/load dumps, and CLI-specific branch logic | Keep backend production ops; retire or demote script code once frontend sparse-first parity is proven |
| Sparse exact-HZ propagation | Backend supports input, affine/dense, constant-side MatMul, conv, convtranspose, avgpool, exact maxpool, ReLU, sigmoid/tanh, structural gather/concat/add/sub paths | Script probe still has wider experiment-only handling and large CLI surface, including var-var MatMul and Softmax relaxations that are not counted exact core | Add focused backend tests for any op still needed by frozen rows before deleting script copies |
| S-curve tightening | Backend has default-off domain/range/graph cut hooks with toy regression | Script probes still contain broader S-curve census and pair-correlation experiments | Keep default off; use only benchmark-wide experiments with full P0/frozen rerun |
| Exact verdict solver | `solver_hz_verdict.py` owns sparse/dense exact MILP, HiGHS/SCIP portfolio env, cutoff-row formulation, base-HZ feasibility, per-row top1 objective parallelism, and witness extraction | Some branch selection remains in workers/drivers; Gurobi diagnostics remain script-only and uncounted | Move open-source branch scheduling into backend profile/runner; keep Gurobi out of counted paths |
| ADV replay gate | `verify_once` reconstructs dense/sparse HZ witnesses and uses ORT/PyTorch replay only as audit | Worker scripts still duplicate replay parsing/reporting logic | Consolidate result ledger fields around verifier metadata |
| Reporting/export | Frozen CSV/MD artifacts are saved under `/data1/Kane/ICSE/...`; frontend HybridZ runs can emit detail/summary CSVs via `--hybridz-results-dir`; `hybridz-benchmark` now emits ICSE-style per-benchmark CSVs plus `_INDEX.csv`, `_DETAIL.csv`, reproducibility README, JSON summaries, frozen-baseline cross-tool ranking CSVs, current-vs-frozen reproduction checks, conservative live-run failure taxonomy, and SHA256 manifests | `hz_export_icse_csv.py`, ledgers, taxonomy scripts still provide historical artifact selection and deeper sparse/probe taxonomy fields | Keep artifact tooling until the frontend runner emits the full historical taxonomy payload |
| Cleanup/deletion | `__pycache__` removed; excluded guided falsification is isolated under `scripts/_excluded_guided_falsification` | Many one-off census/prototype scripts remain untracked/local | Delete only after their production behavior is either migrated or explicitly classified as research-only |

## 2026-06-24 Frontend/Productization Recheck

Latest lightweight gate:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline --help \
  | rg -n "hybridz|frozen|benchmark|require-frozen|results-dir|engine|solvers"
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_benchmark_runner
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf
/data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
mods = [
    'act.back_end.hybridz_config',
    'act.back_end.hybridz_tf.sparse_ops',
    'act.back_end.solver.sparse_hz',
    'act.pipeline.hybridz_benchmark_runner',
    'act.pipeline.hybridz_projected_relu_mip',
    'act.pipeline.hybridz_projected_utils',
    'act.pipeline.hybridz_results',
]
for name in mods:
    __import__(name)
    print('IMPORT_OK', name)
PY
rg -n "import scripts|from scripts|hz_full_driver|hz_full_worker|hz_sparse_worker|cifar_sparse_exact_probe" act docs FULLRUN_HANDOFF.md
/usr/bin/timeout 180 /data1/Kane/miniconda3/envs/act-py312/bin/python \
  -m act.pipeline --verify hybridz-benchmark --category acasxu_2023 \
  --max-instances 1 --hybridz-workers 1 --hybridz-timeout-cap 5 \
  --hybridz-results-dir /tmp/act_hybridz_frontend_smoke_1782288246
```

Current evidence:

- `act.pipeline --help` exposes `--solvers hybridz`, `--verify
  hybridz-benchmark`, `--hybridz-engine`, HybridZ S-curve/ReLU/cell-budget
  knobs, `--hybridz-results-dir`, `--hybridz-workers`, timeout cap, and
  `--hybridz-require-frozen-match`.
- `python -m act.pipeline.hybridz_benchmark_runner` passes its self-test,
  including frozen-suite resolution and frozen-match gate checks.
- `python -m act.back_end.hybridz_tf` passes the package-level HybridZ TF
  self-tests: sparse MatMul constant propagation, exact CNN MaxPool-only path,
  and exact MLP multiply point/variable-drop path.
- The mainline candidate modules import cleanly as ACT package modules:
  `act.back_end.hybridz_config`, `act.back_end.hybridz_tf.sparse_ops`,
  `act.back_end.solver.sparse_hz`, `act.pipeline.hybridz_benchmark_runner`,
  `act.pipeline.hybridz_projected_relu_mip`,
  `act.pipeline.hybridz_projected_utils`, and `act.pipeline.hybridz_results`.
- The production `act/` tree has no `import scripts` or `from scripts`
  dependency.  The mainline runner uses normal frontend subprocesses and
  package modules, not local research-script entrypoints.
- Historical `scripts/` references remain in docs and `FULLRUN_HANDOFF.md` as
  provenance/migration notes, not as current product-path dependencies.
- The one-instance frontend smoke completed successfully and wrote
  `acasxu_2023_hybridz_detail.csv`, `acasxu_2023_hybridz_summary.csv`,
  ICSE-style CSVs, a run JSON, and `_MANIFEST.sha256`.  Its summary was
  `N=1, CERT=0, ADV=0, UNKNOWN=1, ERROR=0, P0=0`; this is a pipeline/output
  smoke, not a frozen-result claim.
- Follow-up smoke with `/tmp/hz_stage2_frontend_smoke`, `--hybridz-timeout-cap
  8`, `--hybridz-workers 1`, `--device cpu`, and `--dtype float64` again
  completed successfully through the normal ACT frontend and wrote detail,
  summary, ICSE detail/index, run-summary JSON, and SHA256 manifest.  The
  short cap intentionally produced UNKNOWN with
  `hybridz_base_hz_not_feasible` / HiGHS base-feasibility timeout, while
  preserving `ERROR=0` and `P0=0`.
- Current frontend smoke:
  `ACT_HYBRIDZ_MEM_FLOOR_GB=0 python -m act.pipeline --verify
  hybridz-benchmark --category acasxu_2023 --max-instances 1
  --hybridz-workers 1 --hybridz-timeout-cap 6 --hybridz-results-dir
  /tmp/act_hybridz_frontend_smoke_stage2 --device cpu --dtype float64`
  completed through the normal ACT frontend.  It emitted detail/summary CSVs,
  ICSE-style CSVs, run-summary JSON, and `_MANIFEST.sha256`; summary was
  `N=1, CERT=0, ADV=0, UNKNOWN=1, ERROR=0, P0=0`.  The run-summary recorded
  the benchmark-wide ACASXu portfolio `normal;sparse_scip_witness`.  This is a
  frontend wiring/profile smoke under a short cap, not a frozen-result claim.
- Current frozen-suite data availability check:
  `list_downloaded_pairs()` sees all 12 frozen benchmarks locally
  (`safenlp_2024=1080`, `metaroom_2023=100`, `sat_relu=100`,
  `malbeware=150`, `cersyve=12`, `acasxu_2023=186`,
  `dist_shift_2023=72`, `linearizenn_2024=60`,
  `tllverifybench_2023=32`, `cora_2024=180`, `relusplitter=220`,
  `cgan_2023=21`).  The external-root frozen-suite smoke below exercises one
  instance from each frozen benchmark through the normal frontend and reaches
  `ERROR=0`, so full frozen reproduction with `--hybridz-require-frozen-match`
  is now a compute-time acceptance task rather than a missing-data task.
- Frozen acceptance preflight:
  `FROZEN_HYBRIDZ_EXPECTED_COUNTS` and
  `frozen_hybridz_expected_summary(...)` in
  `act/back_end/hybridz_config.py` match
  `/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/FINAL_HYBRIDZ_RESULTS_20260624.csv`
  for every field in every frozen benchmark.  `list_hybridz_benchmark_instances`
  returns dense iid ranges for all 12 frozen benchmarks, and the
  `--hybridz-require-frozen-match` comparison gate accepts a synthetic full
  summary built from the config helper.  This verifies the acceptance oracle
  and data enumeration before spending compute on a full frozen rerun.
- `solver_hz.py` now defaults exact HZ bounds to the open-source scipy/HiGHS
  backend.  Gurobi remains available only as an explicit diagnostic oracle via
  `HZ_BOUNDS_BACKEND=gurobi` or `HZ_BOUNDS_GUROBI=1`; it is not a counted
  HybridZ proof dependency.
- Mainline `hybridz-benchmark` now applies `HybridZBenchProfile.mem_gb` to
  verifier subprocesses.  The runner sets `ACT_HYBRIDZ_RLIMIT_AS_GB`, and the
  child `act.pipeline` process applies `RLIMIT_AS` before verification starts.
  This migrates the scripts' per-instance memory isolation into the ACT
  frontend path without changing the HZ formulation or verdict semantics.
- Mainline `hybridz-benchmark` now also mirrors the scripts' host memory
  launch governor.  `HybridZBenchProfile.mem_floor_gb` defaults to the legacy
  20GB floor; `ACT_HYBRIDZ_MEM_FLOOR_GB=0` disables it, and
  `ACT_HYBRIDZ_MEM_POLL_S` controls the wait interval.  The runner keeps at
  most `workers` pending futures and checks host memory immediately before
  submitting each new instance, so queueing time is not counted as HybridZ
  verifier wall time.
- Governor smoke:
  `ACT_HYBRIDZ_MEM_FLOOR_GB=0 python -m act.pipeline --verify
  hybridz-benchmark --category acasxu_2023 --max-instances 1
  --hybridz-timeout-cap 6 --hybridz-workers 1 --device cpu --dtype float64`
  completed through the frontend runner and wrote detail/summary/ICSE CSVs,
  JSON summary, and manifest.  The run summary recorded
  `profile.mem_floor_gb=20.0`; the one-row result was UNKNOWN due to the short
  cap, with `ERROR=0` and `P0=0`.
- Runner cleanup: removed the duplicate `BENCH_WORKERS` lookup from
  `hybridz_benchmark_runner.py` because `HybridZBenchProfile.workers` already
  resolves that table, and replaced the last `typing.List` annotation with
  `list[...]`.  This is behavior-preserving cleanup covered by
  `python -m act.pipeline.hybridz_benchmark_runner`.
- Runner process isolation: sequential sparse/fallback branches now use the
  same process-group execution model as parallel formulation branches.  A
  branch timeout kills the full subprocess group instead of only the direct
  child process, reducing the risk of leftover solver/ORT children consuming
  CPU, GPU, or memory during a full frozen frontend rerun.  This does not
  alter verdict semantics or HybridZ constraints.  Covered by
  `python -m act.pipeline.hybridz_benchmark_runner`, including normal-exit and
  timeout self-test cases for the process-group helper.
- Frozen profile regression: `python -m act.pipeline.hybridz_benchmark_runner`
  now asserts the branch order for every benchmark in
  `FROZEN_BENCHMARK_SUITE`, plus the key frozen profile knobs that affect
  scheduling/formulation (`workers`, `milp_fraction`, timeout cap, sparse
  first/fallback, compressed ReLU, S-curve `k`, query workers, per-instance
  memory cap, and cutoff-row portfolio).  This prevents future cleanup from
  silently drifting away from the frozen frontend reproduction plan.
- Run-summary profile audit: per-benchmark `*_run_summary.json` now records
  the complete HybridZ profile surface used by the runner, including
  `milp_env`, `milp_threads`, S-curve `sigmoid_k`, `cell_budget`, compressed
  ReLU, valid cuts, cutoff-row, sparse portfolio flags, query workers, and
  memory caps.  This makes a frontend run's scheduling/formulation knobs
  auditable from the artifact itself.  Covered by
  `python -m act.pipeline.hybridz_benchmark_runner` plus an ACASXu profile JSON
  smoke that checks `HZ_MILP_BACKEND` is emitted.
- Suite-summary profile audit: root-level `hybridz_suite_summary.json` now
  includes a `profiles` map for every benchmark in the suite.  The benchmark
  run-summary and suite summary share one profile-serialization helper, so the
  artifact schema cannot drift between per-benchmark and full-suite outputs.
  Covered by `python -m act.pipeline.hybridz_benchmark_runner`.
- Missing-data taxonomy: diagnostic suite runs with `--max-instances` can now
  continue past benchmarks whose VNNLIB files are not downloaded, emitting
  synthetic ERROR rows and classifying them as `missing_downloaded_data`.
  Strict full-suite runs still raise on missing benchmark data.  Covered by
  `python -m act.pipeline.hybridz_benchmark_runner` and the frozen-suite smoke
  above.
- Strict suite data preflight: full suite runs and
  `--hybridz-require-frozen-match` now check downloaded VNNLIB coverage before
  launching benchmark subprocesses.  If data is missing, the runner reports all
  missing benchmarks in one error instead of failing at the first benchmark.
- VNN-COMP external-root discovery: `data_model_loader.py` now searches
  optional environment roots (`ACT_VNNLIB_ROOTS`,
  `ACT_VNNCOMP_BENCHMARK_ROOTS`), the repository-local `data/vnnlib`, and the
  adjacent VNN-COMP checkout
  `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks`.  It preserves
  official `instances.csv` row paths such as
  `onnx/medical/perturbations_0.onnx` instead of collapsing them with
  `Path(...).name`, and records `root_dir`/`category_dir` for downstream
  loading.  Duplicate `(benchmark, iid)` rows across roots are resolved by
  first-root precedence, preserving local ACT downloads while filling missing
  frozen benchmarks from the external VNN-COMP tree.
- External-root validation:
  `/data1/Kane/miniconda3/envs/act-py312/bin/python -m
  act.front_end.vnnlib_loader.data_model_loader`,
  `/data1/Kane/miniconda3/envs/act-py312/bin/python -m
  act.front_end.vnnlib_loader.create_specs`, and the strict frozen-suite data
  preflight all pass.  `list_downloaded_pairs()` now sees all frozen benchmark
  counts (`safenlp_2024=1080`, `metaroom_2023=100`, `sat_relu=100`,
  `malbeware=150`, `cersyve=12`, `acasxu_2023=186`, `dist_shift_2023=72`,
  `linearizenn_2024=60`, `tllverifybench_2023=32`, `cora_2024=180`,
  `relusplitter=220`, `cgan_2023=21`).
- External-root frontend smoke:
  `ACT_HYBRIDZ_MEM_FLOOR_GB=0
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline --verify
  hybridz-benchmark --category safenlp_2024 --max-instances 1
  --hybridz-workers 1 --hybridz-timeout-cap 5 --device cpu --dtype float64`
  completed through the normal ACT frontend and produced `ADV=1`, `ERROR=0`,
  `P0=0`.  The run loaded
  `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks/safenlp_2024/onnx/medical/perturbations_0.onnx`,
  proving that subdirectory-preserving VNN-COMP paths now work end to end.  A
  current repeat with `--hybridz-results-dir
  /tmp/act_hybridz_safenlp_frontend_smoke_current` finished in 7.9s and wrote
  `safenlp_2024.csv`, `safenlp_2024_icse_detail.csv`,
  `safenlp_2024_run_summary.json`, and `_MANIFEST.sha256`; the ICSE row was
  `onnx/medical/perturbations_0.onnx,vnnlib/medical/hyperrectangle_418.vnnlib,sat,1.10`,
  with winner branch `sparse_comprelu` and `P0=0`.
- Frozen-suite external-root smoke:
  `ACT_HYBRIDZ_MEM_FLOOR_GB=0 /usr/bin/timeout 240
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline --verify
  hybridz-benchmark --category frozen --max-instances 1 --hybridz-workers 2
  --hybridz-timeout-cap 3 --device cpu --dtype float64` completed 12
  instances across all 12 frozen benchmarks through the normal frontend.
  The initial diagnostic result was `TOTAL N=12, V+A=3, TIMEOUT=2,
  UNKNOWN=5, ERROR=2, P0=0`; `_DETAIL.csv` contained no
  `missing_downloaded_data`.  After the zero-indegree source seeding fix below,
  the same smoke completed with `TOTAL N=12, V+A=3, TIMEOUT=2, UNKNOWN=7,
  ERROR=0, P0=0`.  This is a data-plumbing/source-seeding smoke, not a
  frozen-count reproduction.
  A later current-package repeat without `/usr/bin/timeout`,
  `/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline --verify
  hybridz-benchmark --category frozen --max-instances 1 --hybridz-workers 2
  --hybridz-timeout-cap 6 --hybridz-results-dir
  /tmp/act_hybridz_frozen_smoke_stage2_continue`, also completed all 12
  benchmarks through the normal frontend and wrote suite CSV/JSON/manifest
  artifacts.  Its summary was `TOTAL N=12, CERT=1, ADV=3, V+A=4, TIMEOUT=2,
  UNKNOWN=6, ERROR=0, P0=0`; this remains a short-cap wiring smoke, not a
  frozen-count reproduction.
- Zero-indegree source seeding fix: the frozen-suite smoke exposed that
  `cersyve` and `linearizenn_2024` could still hit frontend `ERROR` even after
  the data root was fixed.  Root cause: `TorchToACT` can emit non-CONSTANT
  zero-indegree model layers for branches that read the network input directly.
  `analyze.py` seeded only CONSTANT extra sources, leaving these model sources
  with output-shaped `+/-inf` default bounds; HybridZ then tried to apply a
  first dense layer to a one-row HZ where the true input dimension was four.
  `analyze.py` now seeds non-CONSTANT zero-indegree model layers with the
  entry input fact, including cache-refinement reruns.  Direct and runner
  smoke checks now return normal `UNKNOWN` instead of `ERROR`:
  `cersyve iid0 -> UNKNOWN, ERROR=0, P0=0`; `linearizenn_2024 iid0 ->
  UNKNOWN, ERROR=0, P0=0`.  This changes only source-layer dataflow seeding;
  it does not add any post-HybridZ rescue or relaxation.  Covered by
  `python -m act.back_end.verifier`, specifically
  `_test_hybridz_zero_indegree_source_uses_entry_fact`, and by the 12-benchmark
  frontend smoke above.
- Reporting cleanup: replaced the remaining `typing.List` annotation in
  `hybridz_results.py` with `list[...]`.  This is behavior-preserving and
  covered by `python -m act.pipeline.hybridz_results`.
- Sparse witness metadata cleanup: sparse structural/affine/nonlinear
  operators now preserve `SparseHZono` input replay metadata
  (`input_center`, `input_radius`, `input_indices`, `input_shape`) whenever
  the represented variable frame is inherited from the input.  Multi-input
  concat/add preserve it only when all parts carry identical metadata.  This
  does not change the exact HZ tuple; it improves exact-HZ ADV auditability by
  keeping sparse MILP witnesses replayable through the normal ORT/model
  witness gate.  Covered by `python -m act.back_end.hybridz_tf.sparse_ops`,
  `python -m act.back_end.solver.solver_hz_verdict`, and
  `python -m act.back_end.verifier`.
- Replay-source regression: `_test_hybridz_witness_replay_gate` now asserts
  that dense HybridZ witnesses replay through `dense_col_ids` and sparse
  HybridZ witnesses replay through `sparse_input_metadata`.  This prevents the
  sparse path from silently losing its input metadata and falling back to a
  non-replayable witness representation.
- Dense-conversion replay fallback: `SparseHZono.from_dense_hz` now preserves
  dense `col_ids`/`bcol_ids` as solver-ignored replay metadata.  The verifier's
  dense witness reconstruction accepts both torch tensors and numpy arrays for
  these ids, so a sparse-engine dense-conversion fallback can still replay an
  exact witness via `dense_col_ids` when sparse input metadata is unavailable.
- Replay id invariant: `SparseHZono` now validates that `col_ids` and
  `bcol_ids` lengths match the continuous and binary generator counts.  The
  solver verdict self-test covers both dense-to-sparse preservation and
  mismatch rejection, preventing future cleanup from silently corrupting exact
  witness replay metadata.
- Sparse input replay invariant: `SparseHZono` now also validates the sparse
  input witness metadata used by the ADV replay gate.  Partial metadata,
  center/radius size mismatches, out-of-range or duplicate input indices, more
  input factors than continuous generator columns, and incompatible
  `input_shape` are rejected when the sparse HZ object is created.  This does
  not alter the represented HZ set or solver tuple; it only prevents bad audit
  metadata from reaching exact witness replay.  Covered by
  `python -m act.back_end.solver.solver_hz_verdict`,
  `python -m act.back_end.hybridz_tf.sparse_ops`, and
  `python -m act.back_end.verifier`.
- Sparse operator point-exactness gate: the mainline
  `python -m act.back_end.hybridz_tf.sparse_ops` self-test now also checks
  scale/add, per-batch linear, constant-side MatMul, Gather, Concat, exact
  ReLU, Sigmoid/Tanh S-curve, Conv2D, AvgPool2D, MaxPool2D, and
  ConvTranspose2D point propagation directly against NumPy/PyTorch operators
  or structural center oracles.  This migrates the low-risk part of
  `scripts/validate_sparse_ops.py` into backend regression coverage: for
  degenerate input boxes, the sparse-HZ center must equal the real affine
  or exact max operator output, catching index-map and broadcast bugs without
  using sampled points as verification evidence.
- Latest local backend validation under `act-py312`:
  `python -m act.back_end.hybridz_tf.sparse_ops` passed
  `_test_sparse_affine_structural_ops`; `python -m act.back_end.verifier`
  passed all 13 verifier self-tests, including the HybridZ zero-indegree source
  and witness replay gate checks.
- Future work record: promising post-freeze directions are preserved in
  `docs/FUTURE_WORK_HYBRIDZ_20260624.md` and in the frozen ICSE artifact copy
  `/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/FUTURE_WORK_HYBRIDZ_20260624.md`.
  They are explicitly excluded from the current frozen count until a new clean
  full rerun proves `P0=0` and frozen-result parity/improvement.
- Handoff record: `FULLRUN_HANDOFF.md` has been refreshed to the current
  2026-06-25 strict pure-HybridZ soundfix artifact
  (`1763/2213 = 977 CERT + 786 ADV`, `P0=0`), with the 2026-06-24 and
  2026-06-19 artifacts marked as historical provenance rather than
  authoritative current results.
- 2026-06-25 safenlp reproducibility edge: a package-level frontend rerun of
  `safenlp_2024` initially reproduced `432 CERT / 646 ADV / 2 UNKNOWN`, one
  ADV below the frozen `432 / 647 / 1`.  The single mismatch was iid704
  (`onnx/medical/perturbations_0.onnx`,
  `vnnlib/medical/hyperrectangle_2710.vnnlib`), which is a HiGHS branch-and-
  bound scheduling edge.  A fixed benchmark-wide open-source solver portfolio
  branch, `normal_seed2` (`HZ_HIGHS_OPTIONS=random_seed=2`), recovers the same
  pure HybridZ MILP witness within the 19s engine budget.  This is not an
  iid-specific configuration: the branch is enabled for all `safenlp_2024`
  instances and is covered by the runner self-test's no-iid-policy guard.
- Safenlp seed2 validation: full package frontend rerun
  `/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline --verify
  hybridz-benchmark --category safenlp_2024 --max-instances 1080
  --hybridz-results-dir
  /data1/Kane/ICSE/act_hybridz_safenlp_seed2_full_20260625` produced
  `432 CERT / 647 ADV / 1079 V+A / 1 UNKNOWN / P0=0`, with zero per-instance
  result mismatches against
  `/data1/Kane/ICSE/act_hybridz_clean_20260624_cora25/safenlp_2024.csv`.
  The key iid704 winner is branch `normal_seed2`, detail wall `18.13s`,
  `hz_witness_source=milp_objective_bound`, and replay
  `model_fn_replay_unsafe:dense_col_ids`.

This proves the frontend entrypoint and script-independence parts of the
productization work at the lightweight-smoke level.  It does not by itself
prove the full Stage-II DoD: the final acceptance gate is still a clean

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark --category frozen \
  --hybridz-require-frozen-match
```

with `FROZEN_REPRO_COMPARISON.json` reporting `ok: true`.
