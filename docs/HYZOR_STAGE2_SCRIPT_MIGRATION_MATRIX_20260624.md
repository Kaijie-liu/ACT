# HyZor Stage-II Script Migration Matrix

Date: 2026-06-24

Scope: current-state inventory of `scripts/` versus the productized ACT
HybridZ path.  The goal is to decide what is already in `act/back_end` /
`act/pipeline`, what still needs migration, and what must remain excluded from
the counted pure-HybridZ method.

Current soundfix artifact:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625`

Soundfix headline: `1763/2213 = 977 CERT + 786 ADV`, `P0=0`.

2026-06-25 erratum: the old 2026-06-24 headline
`1768/2213 = 983 CERT + 785 ADV` is superseded for reporting. The package
runner now applies the correct split-disjunct aggregation rule; metaroom is
`94 CERT / 1 ADV / 5 TIMEOUT = 95 V+A`.

Current productization status and remaining Definition-of-Done gaps are tracked
in `docs/HYZOR_STAGE2_PRODUCTIZATION_STATUS_20260625.md`.

2026-06-27 current-state addendum: the active frozen artifact is now
`/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25`
with `1780/2213 = 980 CERT + 800 ADV`, `P0=0`, `ERROR=0`.
`python -m act.pipeline --verify vnnlib ... --solvers hybridz` is now a
first-class frontend path, and `python -m act.pipeline --verify
hybridz-benchmark --category frozen --hybridz-require-frozen-match` is the
packaged frozen-suite gate.  Use the 2026-06-27 diff-boundary audit for the
latest completion evidence.

## Rules Used For This Matrix

- Productize only benchmark-wide pure-HybridZ behavior.
- Do not migrate guided rescue that lets ORT decide after the HZ engine returns
  `UNKNOWN`.
- Keep Gurobi-only code as diagnostic/oracle code outside the counted path.
- Keep local `scripts/` untracked for now; delete only after the mainline path
  has equivalent tests or an explicit excluded classification.

## Current Mainline Anchors

| Capability | Mainline location | Current status |
|---|---|---|
| First-class `solver="hybridz"` | `act/back_end/config.py`, `act/back_end/verifier.py`, `act/pipeline/cli.py` | present |
| Benchmark-wide profile constants | `act/back_end/hybridz_config.py` | present |
| Normal frontend per-iid runner | `act/pipeline/hybridz_benchmark_runner.py` | present |
| Strict detail/summary CSV recorder | `act/pipeline/hybridz_results.py` | present |
| Dense exact-HZ propagation | `act/back_end/hybridz_tf/*`, `act/back_end/solver/solver_hz.py` | present |
| Sparse CSR HZ carrier | `act/back_end/solver/sparse_hz.py` | present |
| Sparse exact-HZ operators | `act/back_end/hybridz_tf/sparse_ops.py` | present for current frozen core ops, full/pruned S-curve paths, structural row maps, and constructive center witness helpers |
| Safenlp projected one-ReLU exact-HZ branch | `act/pipeline/hybridz_projected_relu_mip.py`, `act/pipeline/hybridz_projected_utils.py` | present as a package module used only by the safenlp benchmark-wide portfolio |
| Sparse propagation cache | `act/back_end/hybridz_tf/hybridz_tf.py` | present |
| Exact HZ verdict MILP | `act/back_end/solver/solver_hz_verdict.py` | present |
| HiGHS/SCIP/open-source verdict backend | `solver_hz_verdict.py` + `HZ_MILP_BACKEND` env | present |
| Cutoff-row/objective-target formulations | `solver_hz_verdict.py`, `hybridz_benchmark_runner.py` | present |
| Constructive base-HZ witness guard | `sparse_ops.py`, `solver_hz_verdict.py`, `hybridz_sparse_exact_probe.py` | present; backend builds/checks the center witness, probe/worker only consume it |
| Exact witness replay gate | `act/back_end/verifier.py` | present |
| ICSE-style CSV/ranking/frozen match | `hybridz_config.py` frozen/report metadata + `hybridz_benchmark_runner.py` export/check logic | present |

## Script Families

| Script or family | Role in old workflow | Mainline equivalent | Decision |
|---|---|---|---|
| `hz_full_driver.py` | Full benchmark orchestration, per-bench profiles, memory governor, branch scheduling | `hybridz_config.py` + `hybridz_benchmark_runner.py` | partially migrated; per-instance RLIMIT cap and host free-RAM launch pausing are now mainline, but specialized legacy branch experiments remain |
| `hz_full_worker.py` | One-instance dense HybridZ worker, RLIMIT, ORT audit, optional guided flags | `verify_once(... solver=hybridz ...)` + `hybridz_results.py` + `ACT_HYBRIDZ_RLIMIT_AS_GB` | mostly migrated; guided flags remain excluded and must not enter mainline defaults |
| `hz_sparse_worker.py` | Wrapper around `cifar_sparse_exact_probe.py` for sparse fallback branches | `sparse_hz.py`, `sparse_ops.py`, `HybridzTF` sparse cache, `sparse_hz_objbound` engine, `hybridz_benchmark_runner.py` sparse branches | product path migrated; keep script as historical debug wrapper until sparse-probe-only diagnostics are either tested or retired |
| `cifar_sparse_exact_probe.py` | Large sparse CSR prototype, sparse propagation, sparse MILP, witness checks, diagnostics | split across `sparse_hz.py`, `sparse_ops.py`, `solver_hz_verdict.py`, `verifier.py`, and the packaged `act.pipeline.hybridz_sparse_exact_probe` wrapper | core migrated; keep local script only as regression/debug source until unsupported probe features are classified |
| `hz_result_ledger.py` | Frozen provenance ledger and accepted/excluded result selection | `FINAL_*` frozen artifact files + runner frozen-match check | keep as provenance; not verifier code |
| `hz_export_icse_csv.py` | Export ICSE-style per-benchmark CSVs | `hybridz_config.py` frozen/report metadata + `hybridz_benchmark_runner.py` export/check logic | migrated; script can be retired after one clean mainline full export rerun |
| `hz_failure_taxonomy.py` | Post-run unresolved taxonomy | `hybridz_benchmark_runner.py` failure taxonomy export | migrated enough for frozen artifact; keep old script as provenance only |
| `p0_audit.py` | P0/result consistency audit over logs | frozen manifest + runner summaries | keep as artifact audit helper; not backend |
| `hz_operator_audit.py` | Broad toy/operator tightness and soundness audit, with Gurobi optional oracle | `sparse_ops.py` self-test, `solver_hz_verdict.py` self-test, `verifier.py` self-test | partially migrated; keep broader Gurobi-oracle cases local |
| `validate_sparse_ops.py` | Real ONNX point-exact validation for sparse affine/spatial ops | `sparse_ops.py` deterministic toy self-test for scale/add, per-batch linear, constant-side MatMul, Gather/Concat, exact ReLU, Sigmoid/Tanh S-curve, and Conv2D/AvgPool2D/MaxPool2D/ConvTranspose2D point-exactness | deterministic backend gate migrated; real ONNX gate still useful before deleting sparse scripts |
| `hz_sparse_attention_operator_audit.py` | cGAN attention-style sparse operator toy gate | none complete | keep as research/test candidate |
| `hz_sparse_mip_structure_audit.py` | Sparse MILP matrix structure census | none complete | keep as diagnostic; not proof path |
| `hz_scurve_pair_correlation_audit.py` | S-curve pair-correlation exploration | `sparse_ops.py` S-curve domain/range/graph cuts only | keep as future-work exploration |
| `distshift_compsig_census.py` | dist_shift S-curve K/cut census | `hybridz_config.py` + `hybridz_benchmark_runner.py` dist_shift branches | migrated for current frozen profile; keep as history |
| `tll_tight_schedule_census.py` | TLL sparse/tight schedule scout | benchmark profile and sparse fallback are partially mainline | keep; not all schedule logic is mainline |
| `acas_projected_census.py` | Uniform ACAS projected-graph scout | no counted mainline equivalent | keep as future-work experiment |
| `cifar_lazy_exact_census.py` | Matrix-free/lazy exact-HZ structure census for CIFAR | no production lazy HZ yet | keep as future work |
| `cifar_sparse_exact_probe.py` CIFAR paths | CIFAR sparse feasibility/scaling diagnosis | only generic sparse ops are mainline | keep as future work, not frozen requirement |
| `hz_projected_relu_mip.py` | Exact projected one-hidden-ReLU MIP | `act/pipeline/hybridz_projected_relu_mip.py` plus `hybridz_projected_utils.py` | selected safenlp portfolio behavior migrated; keep script as legacy provenance only |
| `hz_projected_graph_mip.py` | Exact projected affine/ReLU graph MIP | no selected mainline formulation | keep as future work only |
| `layer_graph_milp.py`, `layer_graph_scip.py` | Primal layer-graph MILP experiments | no selected mainline formulation | keep as future work only |
| `primal_cex_finder.py` | Primal AND-polytope counterexample finder | no counted HybridZ equivalent | keep out of current mainline; engine type differs from HZ factor solver |
| `obbt_toy_validate.py` | Toy OBBT validation over exact-HZ propagation | no mainline OBBT | keep as future work only |
| `layer_unstable_diag.py`, `hz_query_trace.py`, `hz_trace_instance.py` | Per-instance diagnostics | no mainline requirement | keep local; do not productize |
| `hz_parallel_task_runner.py`, `hz_mem_governor.sh` | Local scheduling/resource helpers | runner has bounded workers, profile/env memory governor, per-instance `RLIMIT_AS`, and process-group timeout cleanup for both sequential sparse/fallback and parallel formulation branches | keep local only for ad hoc external scheduling |
| `hz_merge_exact_portfolio.py` | Merge ad hoc exact portfolio outputs | runner now records branches directly | retire after mainline full rerun parity |
| `_local_finish_frozen_official.py` | Local artifact completion helper | frozen artifact now exists | keep as provenance only |
| `nnv_strict_run_one.m`, `run_nnv_strict_vnncomp2025.sh` | External NNV comparison | outside HybridZ method | keep outside product path |
| `_excluded_guided_falsification/*` | LP-witness/guided falsification after HZ `UNKNOWN` | intentionally none | must remain excluded |

## Feature-Level Migration Status

| Feature | From scripts | Current backend/pipeline home | Status |
|---|---|---|---|
| Benchmark-wide workers/walls/MILP fractions | `hz_full_driver.py` | `hybridz_config.py`, `hybridz_benchmark_runner.py` | migrated |
| Frozen suite branch/profile drift guard | `hz_full_driver.py` implicit branch tables | `hybridz_benchmark_runner.py` self-test over `FROZEN_BENCHMARK_SUITE` | migrated for current mainline branch plan |
| Diagnostic missing-data suite behavior | ad hoc local rerun handling | `hybridz_benchmark_runner.py` synthetic ERROR rows for `--max-instances` only, with `missing_downloaded_data` taxonomy | migrated as diagnostics; full frozen match still fails on missing data |
| Strict suite data preflight | manual dataset availability checks | `hybridz_benchmark_runner.py` downloaded-instance preflight for full suites / `--hybridz-require-frozen-match` | migrated; reports all missing benchmarks before launching subprocesses |
| External VNN-COMP benchmark root discovery | manual symlink/copy/local path setup | `vnnlib_loader/data_model_loader.py` root candidates + subdirectory-preserving `instances.csv` resolver | migrated; full frozen suite data is now discoverable from `/data1/Kane/data/vnncomp2025_benchmarks/benchmarks` without copying into `ACT/data/vnnlib` |
| Official `instances.csv` iid isolation | `hz_full_driver.py` | `VNNLibSpecCreator(... instance_indices=...)`, `hybridz_benchmark_runner.py` | migrated |
| Multi-root ONNX input branches | script workers implicitly ran model-specific graph paths | `analyze.py` seeds non-CONSTANT zero-indegree model layers with the entry input fact | migrated; fixes cersyve/linearizenn frontend ERROR without changing HybridZ verdict semantics |
| ICSE CSV/ranking export | `hz_export_icse_csv.py` | `hybridz_config.py` frozen/report metadata + `hybridz_benchmark_runner.py` export/check logic | migrated |
| Frozen match gate | scripts/manual artifact checks | `--hybridz-require-frozen-match` | migrated |
| Dense exact ReLU eq_lagr | dense backend/scripts | `tf_mlp.hz_apply_relu` | migrated |
| Compressed exact ReLU | scripts/probe + dense path | `tf_mlp.py`, `sparse_ops.py` | migrated |
| ReLU valid cuts | scripts/probe | `tf_mlp.py`, `sparse_ops.py` | migrated but must stay selective |
| Sparse CSR HZ data structure | `cifar_sparse_exact_probe.py` | `solver/sparse_hz.py` | migrated |
| Sparse Conv2D/Dense affine | `cifar_sparse_exact_probe.py` | `sparse_ops.py` | migrated |
| Sparse constant-side MatMul exact affine | dense path/probe audit | `sparse_ops.py`, `HybridzTF` sparse MATMUL branch | migrated for point-left/point-right operands |
| Sparse var-var MatMul product relaxation | `cifar_sparse_exact_probe.py` | `sparse_ops.py` | migrated as a sound sparse product-interval lift with backend structural self-test; not an exact-affine core operator |
| Sparse Softmax/simplex relaxation | `cifar_sparse_exact_probe.py` | `sparse_ops.py` | migrated as a sound simplex/range sparse operator with backend structural self-test; keep its exactness boundary explicit |
| Sparse ConvTranspose/AvgPool/MaxPool | `validate_sparse_ops.py`, probe | `sparse_ops.py` | migrated with toy self-test |
| Sparse Add/Sub/Concat/Gather/Shape ops | probe | `sparse_ops.py`, `HybridzTF` sparse propagation | migrated for core paths |
| Sparse Sigmoid/Tanh S-curve | probe/dist_shift census | `sparse_ops.py`, CLI/profile flags | migrated for current profile, including compressed/pruned and full/non-pruned construction plus uniform/curvature grids |
| Base-HZ feasibility guard | probe/worker | `solver_hz_verdict.py`, `verifier.py` | migrated |
| Constructive center witness extension | sparse probe | `sparse_ops.py` backend helpers; package probe only keeps witness state | migrated with backend ReLU/full-S-curve/pruned-S-curve checks |
| Exact witness input reconstruction | worker/probe | `verifier.py` dense/sparse replay | migrated |
| ORT replay audit | worker/probe | `verifier.py` | migrated; audit only |
| HiGHS exact MILP | worker/probe | `solver_hz_verdict.py` | migrated, including sparse cutoff engine |
| SCIP branch | projected/sparse scripts | `solver_hz_verdict.py` via `HZ_MILP_BACKEND=scip` | migrated for verdict layer |
| HiGHS/SCIP portfolio | full driver/sparse scripts | `solver_hz_verdict.py` + runner branch profiles | partially migrated |
| Query/disjunct parallelism | full driver/env | `solver_hz_verdict.py` `HZ_QUERY_WORKERS` | migrated |
| Equality substitution / singleton presolve | sparse probe | `solver_hz_verdict.py` singleton projection, equality-substitution option, sparse FBBT, relaxation EMPTY precheck, LP min-margin prefilter, and solver start conversion | migrated for selected exact solver path; diagnostic variants remain local |
| FBBT-lite / OBBT-like phase tightening | sparse/projection scripts | none selected | not migrated |
| Connected-component sparse presolve | sparse probe | none selected | not migrated |
| LP witness pool / guided ADV | excluded scripts | none by design | excluded |
| Projected/primal layer-graph MIP | projected/primal scripts | none selected | future work, not current HZ product path |

## Current Cleanup Consequences

1. Do not delete `scripts/cifar_sparse_exact_probe.py` yet.  It still contains
   historical sparse probe behavior not fully represented in mainline tests.

2. Do not delete `scripts/hz_sparse_worker.py` yet.  The mainline runner no
   longer calls it, but it documents worker-level fallback behavior and sparse
   probe diagnostics that still need either tests or explicit retirement.

3. `hz_export_icse_csv.py`, `hz_failure_taxonomy.py`, and
   `hz_merge_exact_portfolio.py` are likely retireable after one clean
   mainline `--verify hybridz-benchmark --category frozen
   --hybridz-require-frozen-match` rerun.

4. Guided falsification and LP-witness code is not a migration target.  It
   should stay under `_excluded_guided_falsification/` and remain absent from
   counted configs.

5. Remaining projected-graph/primal MIP scripts should be recorded as future
   work, not folded into the current HybridZ artifact, unless the method
   definition is expanded and re-audited as exact-HZ-preserving.  The selected
   safenlp one-hidden-ReLU projected exact-HZ branch is the exception and is
   already migrated as a package module.

6. The packaged sparse probe has been reduced to a package-local orchestration
   wrapper (`act/pipeline/hybridz_sparse_exact_probe.py`, 1159 lines as of
   2026-06-27).  Its remaining large responsibilities are graph-state
   propagation over ACT's analyzed layer graph, worker CLI option handling, and
   per-query LP/MILP scheduling/logging.  Backend-independent operator and
   solver kernels should continue to move out when they can be tested without
   moving this whole orchestration function.

## First Function-Level Probe Audit Result

The first function-family audit of `cifar_sparse_exact_probe.py` found one
clean exact-affine migration target and two non-exact-affine sparse operators
that need an explicit soundness boundary:

- constant-side `MATMUL` is now represented as an exact sparse affine operator
  when either operand is a point HZ state;
- var-var `MATMUL` product relaxation is now in `sparse_ops.py` with a backend
  structural self-test, but it remains a sound overapproximation rather than an
  exact affine HZ operator;
- `SOFTMAX` simplex/range relaxation is now in `sparse_ops.py` with a backend
  structural self-test, but it must remain outside any exact-only claim unless
  the benchmark profile explicitly allows that sound relaxation.

The next cleanup pass should continue auditing `cifar_sparse_exact_probe.py`
against `sparse_ops.py` and `solver_hz_verdict.py` by function family:

- mark functions whose logic is already covered by mainline tests;
- mark functions still used only by legacy scripts;
- mark functions that are diagnostics only;
- add missing mainline tests for any migrated logic before deleting script
  copies.

## Current Function-Level Probe Audit Result

The 2026-06-27 cleanup passes moved the remaining solver/operator kernels that
were cleanly separable from the package probe:

- sparse HiGHS and SCIP cutoff MILP engines now live in
  `solver_hz_verdict.py`;
- sparse LP min-margin, row-range infeasibility, sparse FBBT, EMPTY-only
  relaxation precheck, solver start conversion, and exact singleton/equality
  presolve support are solver-layer helpers;
- compressed/pruned and full/non-pruned S-curve construction, including
  uniform and curvature segment grids plus domain/range/graph cut metadata, now
  lives in `sparse_ops.py`;
- nearest-neighbor UPSAMPLE, SLICE, and GATHER row maps now live in
  `sparse_ops.py`;
- exact ReLU and S-curve constructive center witness extension plus final
  residual checking now live in `sparse_ops.py`.

Regression evidence for the latest state:

```bash
python -m py_compile act/back_end/hybridz_tf/sparse_ops.py act/pipeline/hybridz_sparse_exact_probe.py
python -m act.back_end.hybridz_tf.sparse_ops
python -m act.pipeline.hybridz_sparse_exact_probe --self-test
python -m act.pipeline.hybridz_sparse_worker --bench sat_relu --iid 0 \
  --lp-queries 1 --milp-timeout 5 --worker-timeout 45
python -m act.pipeline.hybridz_sparse_worker --bench dist_shift_2023 --iid 0 \
  --lp-queries 1 --milp-timeout 5 --worker-timeout 60 \
  --compressed-sigmoid --sigmoid-prune-degenerate --sigmoid-k 2 --tanh-k 1 \
  --scurve-domain-cuts --scurve-graph-cuts
```

The `sat_relu` smoke returned `ADV`, `P0=false`, and `ort_verified=true`; the
`dist_shift_2023` S-curve smoke returned `UNKNOWN`, `P0=false`, final HZ
`n=10 ng=5.3k nb=2.6k nc=1.8k`, and constructive-center residual
`max_eq=1.36e-12 max_ub=0`.  Both packaged frontend smokes and the frozen
manifest check also remained clean.

What remains in the package probe is no longer a standalone backend primitive:
it is the product sparse-worker orchestration over analyzed ACT layer graphs
plus query scheduling/reporting.  Moving it wholesale into `hybridz_tf` would
increase coupling to pipeline-only data (`net`, `before`, `after`, benchmark
queries, stdout JSON contract), so the next lower-risk cleanup target is either
worker CLI/result reporting or deleting/archiving local scripts after a final
full frozen frontend parity run.

## Current Product-Path Dependency Check

Latest check:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline --help \
  | rg -n "hybridz|frozen|benchmark|require-frozen|results-dir|engine|solvers"
/data1/Kane/miniconda3/envs/act-py312/bin/python -m act.pipeline.hybridz_benchmark_runner
rg -n "import scripts|from scripts|hz_full_driver|hz_full_worker|hz_sparse_worker|cifar_sparse_exact_probe" act docs FULLRUN_HANDOFF.md
/usr/bin/timeout 180 /data1/Kane/miniconda3/envs/act-py312/bin/python \
  -m act.pipeline --verify hybridz-benchmark --category acasxu_2023 \
  --max-instances 1 --hybridz-workers 1 --hybridz-timeout-cap 5 \
  --hybridz-results-dir /tmp/act_hybridz_frontend_smoke_1782288246
```

Observed state:

- The ACT frontend now exposes the HybridZ solver mode, benchmark runner,
  result directory, worker/timeout controls, exact-HZ formulation knobs, and
  frozen-match gate.
- The mainline runner self-test passes.
- `python -m act.back_end.hybridz_tf` passes the package-level HybridZ TF
  self-test entrypoint, and the mainline candidate modules import cleanly:
  `hybridz_config`, `sparse_ops`, `sparse_hz`,
  `hybridz_benchmark_runner`, `hybridz_projected_relu_mip`,
  `hybridz_projected_utils`, and `hybridz_results`.
- No production `act/` file imports `scripts/`.  The remaining references are
  comments/docstrings or migration/provenance documentation.
- The one-instance ACASXu frontend smoke completed and emitted detail/summary,
  ICSE-style CSVs, JSON, and manifest files through the mainline runner.

Consequence: `scripts/` should remain local/untracked by user request, but it
is no longer a product-path import dependency for the normal HybridZ frontend
entrypoint.  Deletion is still deferred until one full frozen frontend rerun
matches the frozen table and any still-useful script-only diagnostics are
either migrated into tests or explicitly retired.

## Private Backend Helper Imports From Legacy Scripts

Latest cleanup scan found one backend-private helper retained only because
local legacy scripts still import it:

| Script | Private helper | Current use | Cleanup action |
|---|---|---|---|
| `scripts/hz_full_worker.py` | `solver_hz_verdict._hz_relax_np_sparse` | LP-relaxation witness/guided diagnostic code, explicitly excluded from counted pure-HZ results | Keep until full frontend frozen parity is proven; then either move the helper into the local script or retire the diagnostic branch |
| `scripts/hz_full_worker.py` | `solver_hz._split_eq_le` | Phase-fix/guided diagnostic helper logic | Same treatment: do not expose as product API |
| `scripts/hz_query_trace.py` | `solver_hz_verdict._hz_np_sparse` | Query tracing/debug of solver matrices | Keep as debug-only script dependency until replaced by a proper trace API or deleted |

The production `act/` path does not import `scripts/`, and these private-helper
imports are not part of `--solvers hybridz` or `--verify hybridz-benchmark`.
