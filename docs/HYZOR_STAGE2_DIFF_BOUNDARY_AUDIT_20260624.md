# HyZor Stage-II Diff Boundary Audit

Date: 2026-06-24

Purpose: keep the Stage-II consolidation work honest about diff size.  This
file separates branch-level divergence from the current uncommitted HybridZ
productization work, so cleanup decisions are not made from a misleading
`git diff upstream/main` alone.

## Git Baseline

- Current branch: `hz-cam-1`
- `upstream/main`: `3a2853615898a8c6f109d40309d441a6df5c72a8`
- `HEAD`: `271e92d188a1fe57795979ad11f0bbf941489010`
- merge-base: `e831a0f580acd3fb6e7986f17bc4bb4247f38c61`

Command run:

```bash
git remote -v
git rev-parse upstream/main HEAD
git merge-base upstream/main HEAD
git log --oneline --decorate --left-right --cherry-pick HEAD...upstream/main
```

## Key Finding

`git diff upstream/main` includes pre-existing branch history outside the
current Stage-II work.  Notable branch-level differences include:

- `.github/workflows/*`
- `act/back_end/bab/*`
- `act/back_end/dual_tf/*`
- `act/back_end/solver/solver_dual.py`
- `act/front_end/torchvision_loader/*`
- `act/pipeline/fuzzing/*`

These must not be treated as automatically removable Stage-II changes.  They
need a separate ownership decision or a branch rebase/cleanup plan.

Latest local baseline recheck, using the existing local `upstream/main` ref
without fetching, shows:

- branch-only committed delta since merge-base:
  `git diff --stat upstream/main...HEAD` reports
  `14 files changed, 1932 insertions(+), 229 deletions(-)`;
- direct local comparison against `upstream/main`:
  `git diff --stat upstream/main HEAD` reports
  `29 files changed, 2087 insertions(+), 1743 deletions(-)`;
- reason for the gap: `upstream/main` contains newer upstream commits not yet
  merged into `hz-cam-1`, while `hz-cam-1` has five local branch commits since
  the merge-base.  Stage-II cleanup should therefore use the current
  uncommitted footprint and the three-dot branch delta, not direct
  `upstream/main HEAD` alone.

## Current Uncommitted Stage-II Footprint

Commands run:

```bash
git diff --stat HEAD -- . ':!scripts' ':!docs'
git diff --numstat -- act ':!scripts' ':!docs' ':!FULLRUN_HANDOFF.md'
```

Current tracked worktree delta, excluding local `scripts/` and docs:

```text
26 files changed, 4802 insertions(+), 687 deletions(-)
```

Largest current tracked deltas:

| File | + | - | Stage-II reason |
|---|---:|---:|---|
| `act/back_end/solver/solver_hz_verdict.py` | 1295 | 66 | exact HZ verdict MILP, sparse/dense HiGHS/SCIP, witness/base guard |
| `act/back_end/hybridz_tf/tf_mlp.py` | 819 | 274 | exact ReLU/S-curve/compressed HZ propagation; removed backend convex ReLU helper |
| `act/back_end/verifier.py` | 740 | 13 | first-class `hybridz` verifier mode and witness replay gate |
| `act/back_end/hybridz_tf/hybridz_tf.py` | 346 | 14 | sparse carry/cache, lazy-cached sparse ops import, HybridZ profile wiring, and sparse MatMul propagation smoke; unused `has_*` wrappers removed |
| `act/pipeline/cli.py` | 346 | 8 | frontend `--solvers hybridz` and benchmark runner flags |
| `act/front_end/vnnlib_loader/data_model_loader.py` | 306 | 83 | external VNNLIB root discovery and official row-path preservation |
| `act/back_end/interval_tf/tf_cnn.py` | 173 | 25 | frontend graph compatibility needed by HybridZ path |
| `act/pipeline/verification/utils.py` | 107 | 17 | ONNX operator conversion and graph wiring needed by HybridZ replay/front-end |
| `act/back_end/solver/solver_hz.py` | 104 | 39 | open-source HZ bounds default, lazy Gurobi diagnostic import, row-select tight bounds |
| `act/back_end/cli.py` | 104 | 6 | direct backend HybridZ CLI knobs |
| `act/front_end/vnnlib_loader/create_specs.py` | 90 | 3 | downloaded instance-index filtering for official VNNLIB rows |
| `act/back_end/hybridz_tf/tf_cnn.py` | 58 | 11 | exact affine/spatial HZ ops and exact-only MaxPool cleanup |
| `act/pipeline/verification/__init__.py` | 55 | 39 | lazy verification package facade; no eager validator/LLM/Gurobi import |
| `act/pipeline/__init__.py` | 53 | 66 | lazy top-level pipeline facade; no eager conversion/profiler import |
| `act/back_end/config.py` | 52 | 4 | HybridZ profile/config integration |
| `act/pipeline/verification/act2torch.py` | 37 | 0 | ACT-to-Torch replay compatibility for ONNX Resize/UPSAMPLE |

Untracked mainline candidate files:

| File | Lines | Role |
|---|---:|---|
| `act/back_end/hybridz_config.py` | 498 | benchmark-wide HybridZ profiles, strict counted knobs, frozen acceptance counts, cross-tool reporting metadata, and config invariant self-test |
| `act/back_end/hybridz_tf/__main__.py` | 28 | package-level HybridZ TF smoke-test entrypoint |
| `act/back_end/hybridz_tf/sparse_ops.py` | 2294 | sparse exact-HZ operators and mainline sparse affine/spatial point-exactness tests |
| `act/back_end/solver/sparse_hz.py` | 274 | sparse HZ 6-tuple container and replay metadata invariants |
| `act/pipeline/hybridz_benchmark_runner.py` | 2016 | frontend benchmark suite runner, artifact export, and strict import/path policy self-tests |
| `act/pipeline/hybridz_projected_relu_mip.py` | 788 | safenlp one-hidden-ReLU projected exact-HZ package branch |
| `act/pipeline/hybridz_projected_utils.py` | 186 | package-local VNNLIB/interval/query helpers for projected exact-HZ branch |
| `act/pipeline/hybridz_results.py` | 173 | strict HybridZ frontend detail/summary recorder |

## Script Residue Boundary

`scripts/` remains untracked/local by user request.  Current script families:

- reproduction/ledger/export: `hz_full_driver.py`, `hz_full_worker.py`,
  `hz_sparse_worker.py`, `hz_result_ledger.py`, `hz_export_icse_csv.py`,
  `hz_failure_taxonomy.py`, `_local_finish_frozen_official.py`;
- sparse/operator experiments: `cifar_sparse_exact_probe.py`,
  `cifar_lazy_exact_census.py`, `validate_sparse_ops.py`,
  `hz_sparse_mip_structure_audit.py`, `hz_sparse_attention_operator_audit.py`;
- projected exact MIP: `hz_projected_relu_mip.py` is now legacy provenance for
  the migrated safenlp package branch; `hz_projected_graph_mip.py`,
  `layer_graph_milp.py`, and `layer_graph_scip.py` remain exploratory;
- benchmark-specific census/probes: `distshift_compsig_census.py`,
  `tll_tight_schedule_census.py`, `acas_projected_census.py`,
  `layer_unstable_diag.py`, `hz_query_trace.py`, `hz_trace_instance.py`;
- excluded/guided diagnostics: `scripts/_excluded_guided_falsification/*`.

Only benchmark-wide, pure-HZ behavior should migrate from these scripts into
`act/back_end/*` or `act/pipeline/*`.  Guided diagnostics, Gurobi-only probes,
and per-instance rescue should remain excluded.

## Temporary-Code Scan

Command run:

```bash
rg -n "TODO|FIXME|HACK|temporary|one-off|debug|Gurobi|guided|input split|drop" \
  act/back_end/hybridz_tf act/back_end/solver act/back_end/hybridz_config.py \
  act/back_end/verifier.py act/pipeline/hybridz_benchmark_runner.py \
  act/pipeline/hybridz_results.py
```

Findings:

- No obvious `TODO/FIXME/HACK` markers in the new mainline sparse-HZ files.
- `solver_hz_verdict.py` documents that the counted verdict path is
  scipy/HiGHS/SCIP and forbids Gurobi.
- `solver_hz.py` lazy-loads ACT's older `GurobiSolver` only when explicitly
  requested as a diagnostic oracle.  Exact HZ bounds default to scipy/HiGHS.
  Gurobi is reachable only with explicit `HZ_BOUNDS_BACKEND=gurobi` or
  `HZ_BOUNDS_GUROBI=1`, and remains outside counted strict verdict paths.
- `hybridz_config.py` no longer exports unused guided/prefilter helper
  functions or empty base-witness toggles.  Uncounted diagnostics stay outside
  the product profile instead of appearing as dormant frontend knobs.
- `hybridz_tf/hybridz_tf.py` has representation-drop guards.  These are not
  proof shortcuts; strict HybridZ treats drops as UNKNOWN.
- `hybridz_tf/tf_cnn.py` is now exact-only for MaxPool2D.  The old backend
  convex/DeepZ ReLU helper was removed from `tf_mlp.py`; legacy
  `hz_maxpool2d(..., exact=False)` call sites now receive `None`, which makes
  the HZ representation drop rather than producing a triangle-relaxed proof.

## Open-Source Bounds Default Cleanup

`act/back_end/solver/solver_hz.py` now routes `hz_compute_bounds(...,
exact=True)` to the open-source scipy/HiGHS LP backend by default.  The legacy
Gurobi bounds helper is lazy-loaded for explicit diagnostics only:

```bash
HZ_BOUNDS_BACKEND=gurobi ...
# or
HZ_BOUNDS_GUROBI=1 ...
```

This change does not alter the frozen counted verdict solver, which already
uses `solver_hz_verdict.py` with scipy/HiGHS/SCIP and forbids Gurobi-counted
proof.  It removes the remaining ambiguity that a default HZ bounds call could
silently prefer a commercial backend.

Follow-up hardening: `act/back_end/__init__.py`,
`act/back_end/solver/__init__.py`, `act/pipeline/verification/torch2act.py`,
and `act/pipeline/verification/validate_verifier.py` now also lazy-load the
Gurobi backend.  A runner self-test imports `act.back_end.solver` and
`act.pipeline` in a subprocess and fails if the import prints Gurobi license
detection or imports `gurobipy`.

`act/pipeline/verification/__init__.py` is now a lazy facade instead of
wildcard-importing `torch2act`, `act2torch`, verifier validation, utilities, and
LLM probe modules at package import time.  The common convenience symbols
(`TorchToACT`, `ModelFactory`, `VerificationValidator`, profiling helpers, and
explicit submodules) remain available via `__getattr__`, while a plain
`import act.pipeline` no longer loads validator, LLM, or Gurobi diagnostic
modules.  The HybridZ runner self-test guards this import boundary.

`act/pipeline/__init__.py` is also a lazy facade for the same reason: top-level
convenience symbols such as `TorchToACT`, `ModelFactory`, and profiling helpers
still work, but importing the package for command-module dispatch does not load
conversion utilities, YAML model factories, profiler helpers, verifier
validation, LLM probes, or optional solver diagnostics.

2026-06-25 frontend smoke:

- Direct standard verifier entrypoint `python -m act.pipeline --verify vnnlib
  --solvers hybridz --category safenlp_2024 --instance-index 0 ...` completed
  with `ADV=1`, `P0=0`, `solver=hybridz`, `engine=dense_hz_objbound`.
- Direct standard verifier entrypoint `python -m act.pipeline --verify vnnlib
  --solvers hybridz --category metaroom_2023 --instance-index 0 ...` completed
  with `CERT=19`, `P0=0`, `solver=hybridz`, `engine=dense_hz_objbound`.
- `python -m act.pipeline --verify hybridz-benchmark --category safenlp_2024
  --max-instances 1 ...` completed with `ADV=1`, `P0=0`, and wrote
  `/tmp/act_hybridz_safenlp_frontend_smoke_20260625/*`.
- `--hybridz-require-frozen-match` with `--max-instances 1` correctly exited
  `1`, proving the frozen acceptance gate rejects partial runs.

This is an entrypoint/safety-valve smoke, not a replacement for the complete
frozen-suite acceptance pass.

## Immediate Cleanup Targets

1. Re-audit non-HybridZ touched files before commit:

   - `act/back_end/interval_tf/tf_cnn.py`
   - `act/front_end/model_synthesis.py`
   - `act/pipeline/verification/act2torch.py`
   - VNNLIB loader changes

   Each must have a direct HybridZ reproduction or soundness reason.  Otherwise
   it is a candidate for rollback or isolation behind the HybridZ path.

   First pass on `act/back_end/interval_tf/tf_cnn.py`: keep for now, but require
   focused smoke coverage before commit.  The current changes are shape/operator
   compatibility fixes used by the frontend/ACT graph path that HybridZ relies
   on:

   - prefer valid `input_shape` metadata for rectangular conv feature maps;
   - allow flatten `output_shape` with or without the batch dimension;
   - avoid materializing huge AvgPool2d equivalent dense matrices;
   - normalize Upsample mode/size/scale tuples to PyTorch semantics, matching
     the corresponding `act2torch` replay change.

   This is not a HybridZ mathematical operator change, so it should remain
   documented as frontend compatibility support rather than core HZ logic.
   Focused smoke now exists in `act/back_end/interval_tf/tf_cnn.py` and covers
   the four compatibility points above:

   ```bash
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.back_end.interval_tf.tf_cnn import _test_interval_tf_cnn_compat
   _test_interval_tf_cnn_compat()
   print("PASS _test_interval_tf_cnn_compat")
   PY
   ```

   First pass on VNNLIB loader changes: keep for now.  They are required by
   the mainline per-iid HybridZ runner:

   - `data_model_loader.py` now preserves zero-based `instances.csv` row
     indices for both headerless and headered files;
   - `create_specs.py` can filter by explicit `instance_indices`, so each
     subprocess can run exactly one official row with its own wall clock;
   - focused smokes cover both behaviors without loading real ONNX/VNNLIB:

   ```bash
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.front_end.vnnlib_loader.data_model_loader import _test_list_downloaded_pairs_preserves_iid_indices
   _test_list_downloaded_pairs_preserves_iid_indices()
   print("PASS _test_list_downloaded_pairs_preserves_iid_indices")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.front_end.vnnlib_loader.create_specs import _test_instance_indices_filtering
   _test_instance_indices_filtering()
   print("PASS _test_instance_indices_filtering")
   PY
   ```

   First pass on `act/front_end/model_synthesis.py`: keep.  It preserves
   `_act_onnx_path`, `_act_onnx_model`, and `_act_onnx_input_shape` when a
   VNNLIB ONNX-converted model is wrapped into a batched `VerifiableModel`.
   This metadata is needed by HybridZ ADV witness replay/audit and does not
   alter HZ propagation or verdict logic.  A focused import-level smoke covers
   the propagation without running the heavy CUDA-oriented module `__main__`:

   ```bash
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.front_end.model_synthesis import _test_preserves_act_onnx_metadata
   _test_preserves_act_onnx_metadata()
   print("PASS _test_preserves_act_onnx_metadata")
   PY
   ```

   First pass on `act/pipeline/verification/act2torch.py` and
   `act/pipeline/verification/utils.py`: keep.  These changes are frontend
   graph/replay support, not HybridZ proof shortcuts:

   - ONNX Resize positional inputs are resolved as `(input, roi, scales,
     sizes)`, so an empty/floating `roi` tensor is no longer mistaken for the
     scale tensor;
   - full-rank ONNX `sizes/scales` are converted to spatial-only PyTorch
     `F.interpolate` parameters and ONNX `linear/cubic` modes are mapped to
     PyTorch's `bilinear/trilinear/bicubic` names;
   - explicit predecessor overrides preserve correct ACT DAG wiring for
     `Pow(x, 2) -> MUL(x, x)`, `Split`, and var-var binary ops;
   - `_assert_dag` now treats duplicate predecessor entries as one graph edge
     for cycle detection, while the transfer functions can still see duplicate
     positional inputs where they are semantically required.

   Focused smokes cover the two fragile boundaries without loading real
   VNNLIB/ONNX files:

   ```bash
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.pipeline.verification.act2torch import _test_act_graph_upsample_replay_onnx_compat
   _test_act_graph_upsample_replay_onnx_compat()
   print("PASS _test_act_graph_upsample_replay_onnx_compat")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.pipeline.verification.utils import _test_resize_mode_and_dag_helpers
   _test_resize_mode_and_dag_helpers()
   print("PASS _test_resize_mode_and_dag_helpers")
   PY
   ```

2. Keep mainline runner/reporting, but keep frozen acceptance and cross-tool
   reporting tables in `act/back_end/hybridz_config.py` rather than growing the
   runner. This has been applied for the current frozen tables.

3. Do not delete `scripts/` yet.  Several frozen-result reproduction branches
   are still not covered by backend/mainline tests, especially ACASXu cuts/FBBT,
   TLL/cGAN sparse-probe options, and projected exact-ReLU experiments.

4. Before any commit, rerun:

   ```bash
   /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_benchmark_runner
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.back_end.interval_tf.tf_cnn import _test_interval_tf_cnn_compat
   _test_interval_tf_cnn_compat()
   print("PASS _test_interval_tf_cnn_compat")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.front_end.vnnlib_loader.data_model_loader import _test_list_downloaded_pairs_preserves_iid_indices
   _test_list_downloaded_pairs_preserves_iid_indices()
   print("PASS _test_list_downloaded_pairs_preserves_iid_indices")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.front_end.vnnlib_loader.create_specs import _test_instance_indices_filtering
   _test_instance_indices_filtering()
   print("PASS _test_instance_indices_filtering")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.front_end.model_synthesis import _test_preserves_act_onnx_metadata
   _test_preserves_act_onnx_metadata()
   print("PASS _test_preserves_act_onnx_metadata")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.pipeline.verification.act2torch import _test_act_graph_upsample_replay_onnx_compat
   _test_act_graph_upsample_replay_onnx_compat()
   print("PASS _test_act_graph_upsample_replay_onnx_compat")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python - <<'PY'
   from act.pipeline.verification.utils import _test_resize_mode_and_dag_helpers
   _test_resize_mode_and_dag_helpers()
   print("PASS _test_resize_mode_and_dag_helpers")
   PY
   /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.solver.solver_hz_verdict
   /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.verifier
   /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf
  /data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf.sparse_ops
  git diff --check
  ```

## 2026-06-24 Config Cleanup Pass

Small cleanup applied after the frontend smoke:

- removed unreferenced `adv_prefilter_timeout()` and `sparse_first_enabled()`
  helpers from `act/back_end/hybridz_config.py`;
- removed unreferenced exported constants for guided/prefilter/base-witness
  diagnostics and the not-yet-productized ACASXu cuts/FBBT flag;
- kept benchmark-wide strict counted knobs that are actually consumed by
  `verify_once` or `hybridz_benchmark_runner.py`.

Validation:

```bash
rg -n "\b(adv_prefilter_timeout|sparse_first_enabled|BASE_HZ_WITNESS_FIRST|BASE_HZ_WITNESS_AFTER_UNKNOWN|ACASXU_CUTS_FBBT_FALLBACK|PHASEFIX_ADV_PORTFOLIO|ADV_PREFILTER_TIMEOUT|acasxu_cuts_fbbt_fallback)\b" act FULLRUN_HANDOFF.md
/data1/Kane/miniconda3/envs/act-py312/bin/python -m py_compile \
  act/back_end/hybridz_config.py act/pipeline/hybridz_benchmark_runner.py \
  act/back_end/verifier.py
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_benchmark_runner
```

The production/handoff reference scan is empty and the runner self-test passes.
This is an implementation cleanup only; it does not change the frozen result
table.

## 2026-06-24 Strict MUL Cleanup

Dense `hybridz_tf/tf_mlp.py` previously carried a dormant
`_hz_mul_mccormick(...)` helper behind an unconfigured
`_allow_nonlinear_relaxation` attribute.  That path was not part of the counted
pure-HZ frontend, and it would have represented var-var multiplication with a
relaxation rather than an exact HZ operator.

Cleanup applied:

- removed `_hz_mul_mccormick(...)` and its `_empty_ids(...)` helper;
- removed the hidden `_allow_nonlinear_relaxation` branch from `tf_mul`;
- kept exact point-times-HZ multiplication as elementwise scaling;
- made var-var `MUL` drop the dense HZ state so strict HybridZ returns
  `UNKNOWN` instead of silently propagating a non-exact relaxation;
- tightened `_hz_is_point(...)` to accept HZ states whose generator columns are
  all numerically zero, which is how `hz_from_bounds(lb == ub)` represents
  constants.

Regression added:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf
```

now runs `_test_hz_mul_exact_point_and_var_drop` in addition to the existing
sparse MatMul and exact MaxPool checks.

Validation:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m py_compile \
  act/back_end/hybridz_tf/tf_mlp.py act/back_end/hybridz_tf/__main__.py
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.hybridz_tf.sparse_ops
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.solver.solver_hz_verdict
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.back_end.verifier
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_benchmark_runner
git diff --check
```

All passed.  This cleanup only removes a dormant non-exact option; it should
not reduce any strict pure-HZ frozen result because counted runs could not
enable that hidden attribute through the frontend/profile path.

## 2026-06-24 Exact-Only Reduce Cleanup

Strict product propagation now treats `hz_reduce(...)` as exact redundancy
removal only:

- removed the silent binary-relax and Girard calls from `tf_mlp.hz_reduce`;
- kept `max_order` only as a legacy call-site compatibility parameter;
- left `order_reduce.hz_girard_reduce(...)` available as an explicit
  script/audit ablation helper, not as a product-path call;
- updated `hybridz_tf/algorithms/__init__.py` to state that lossy helpers are
  not used by strict HybridZ propagation.

This makes the fallback behavior explicit: if exact HZ propagation grows beyond
the frontend/solver capacity, strict HybridZ must drop the representation and
return `UNKNOWN` rather than silently continuing with a lossy HZ.

## 2026-06-25 Safenlp Seed2 Portfolio Boundary

The package runner now includes one extra fixed `safenlp_2024` branch:
`normal_seed2`, which sets `HZ_HIGHS_OPTIONS=random_seed=2` for the standard
dense exact-HZ objective-bound solver.  This is a solver scheduling portfolio
knob only:

- it is enabled for every `safenlp_2024` instance, not for a selected iid;
- it does not change the HZ tuple, ReLU encoding, or witness acceptance rule;
- accepted ADV rows still require a HybridZ MILP witness and replay gate;
- the runner self-test checks branch metadata does not contain iid-specific
  policy text.

Reason: a fresh package frontend rerun without this branch reproduced
`safenlp_2024` as `432 CERT / 646 ADV / 2 UNKNOWN`, one below the frozen
artifact.  The only mismatch was iid704, a HiGHS branch-and-bound edge case.
`normal_seed2` recovers iid704 as pure HybridZ ADV within the 19s engine budget.

Validation:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline \
  --verify hybridz-benchmark --category safenlp_2024 --max-instances 1080 \
  --hybridz-results-dir /data1/Kane/ICSE/act_hybridz_safenlp_seed2_full_20260625
```

Result: `432 CERT / 647 ADV / 1079 V+A / 1 UNKNOWN / P0=0`, with zero
per-instance result mismatches against the frozen `safenlp_2024.csv`.  The
critical iid704 winner is `normal_seed2`, `wall_s=18.13`, and
`hz_witness_source=milp_objective_bound`.

## Acceptance Implication

The Stage-II goal is not complete until a full frontend run:

```bash
python -m act.pipeline --verify hybridz-benchmark --category frozen \
  --hybridz-require-frozen-match
```

produces `FROZEN_REPRO_COMPARISON.json` with `ok: true`, and the remaining
script-only capabilities are either migrated or explicitly classified as
research-only/excluded.
