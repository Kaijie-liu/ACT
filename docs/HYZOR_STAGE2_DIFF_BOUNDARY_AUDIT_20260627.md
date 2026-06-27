# HyZor Stage-II Diff Boundary Audit

Date: 2026-06-27

Baseline: `upstream/main` after `git fetch upstream`.

Current branch: `hz-cam-1` after sparse-probe consolidation passes, frozen-gate
productization, sparse-worker option deduplication, package-local HybridZ
CLI option/spec helper consolidation, benchmark reporting helper
consolidation, solver-helper consolidation, and sparse MILP cutoff-engine
consolidation, S-curve backend consolidation, and sparse structural row-map
helper consolidation, full S-curve backend consolidation, and constructive
center witness backend consolidation, and tight-ReLU apply-helper backend
consolidation, plus local untracked `scripts/`.
The local `scripts/` directory is not tracked and is not part of this diff.

## Summary

Current upstream diff after `git fetch upstream` on 2026-06-27:

`80 files changed, 24652 insertions(+), 2217 deletions(-)`

Directory-level split:

| Area | Files | Added | Deleted | Status |
|---|---:|---:|---:|---|
| `act/pipeline` | 19 | 7424 | 159 | largest remaining consolidation target |
| `act/back_end/hybridz_tf` | 8 | 6592 | 292 | core HZ operator/product path |
| `act/back_end/solver` | 5 | 4102 | 423 | verdict/sparse HZ solver path |
| `docs` | 16 | 4123 | 0 | audit/provenance/future-work docs |
| `act/back_end/other` | 17 | 1696 | 986 | frontend/backend integration hooks |
| `act/front_end` | 7 | 508 | 334 | benchmark/data loading integration |
| `FULLRUN_HANDOFF.md` | 1 | 200 | 0 | run handoff/provenance |
| other | 7 | 7 | 23 | package metadata / small compatibility files |

Largest files by changed lines:

| File | Added | Deleted | Interpretation |
|---|---:|---:|---|
| `act/back_end/hybridz_tf/sparse_ops.py` | 4104 | 0 | sparse exact-HZ propagation core; SOFTMAX, var-var MATMUL, exact ReLU graph construction, exact ReLU tight-LP bound tightening/apply helper, S-curve return metadata, uniform/curvature pruned S-curve support, full S-curve construction, constructive center witness helpers, and structural row-map helpers moved here from the probe |
| `act/back_end/solver/solver_hz_verdict.py` | 3488 | 0 | exact verdict MILP, open-source solver portfolio, reusable sparse MILP presolve / relaxation helpers, LP min-margin prefilter, and sparse HiGHS/SCIP cutoff engines |
| `act/pipeline/hybridz_benchmark_runner.py` | 1783 | 0 | product runner and branch portfolio; reporting helpers are now in `hybridz_results.py` |
| `act/pipeline/hybridz_sparse_exact_probe.py` | 1135 | 0 | remaining package-level prototype; sparse cutoff engines, tight-ReLU application, S-curve construction, constructive witness helpers, structural row maps, and most reusable solver helpers have moved into backend layers |
| `act/back_end/hybridz_tf/tf_mlp.py` | 1253 | 246 | dense exact ReLU/compressed ReLU and nonlinear operators |
| `act/pipeline/hybridz_results.py` | 1227 | 0 | benchmark/suite CSV and JSON exports, frozen comparison, cross-tool ranking, failure taxonomy, and P0 reporting helpers |
| `act/pipeline/hybridz_projected_relu_mip.py` | 744 | 0 | safenlp projected exact-ReLU branch; still pure-HZ but specialized |
| `act/back_end/verifier.py` | 740 | 13 | frontend solver integration and metadata/soundness guards |
| `act/pipeline/hybridz_sparse_census.py` | 660 | 0 | diagnostic/census path; not needed for frozen one-command verification |
| `act/pipeline/hybridz_sparse_worker.py` | 452 | 0 | sparse worker wrapper; common probe options deduplicated |
| `act/back_end/hybridz_config.py` | 509 | 0 | frozen oracle and benchmark profiles |

## Product Boundary Assessment

The current product path no longer imports `scripts/`.  A scan over `act/`
finds only the runner self-test assertion that generated commands must not
contain `scripts/` paths.

Frontend smoke evidence after the latest cleanup:

```bash
python -m act.pipeline --verify hybridz-benchmark --category sat_relu \
  --max-instances 1 --hybridz-workers 1 --hybridz-timeout-cap 120 \
  --hybridz-results-dir /tmp/hyzor_stage2_frontend_smoke
```

This packaged ACT entry produced
`/tmp/hyzor_stage2_frontend_smoke/sat_relu_hybridz_summary.csv` with
`N=1, CERT=0, ADV=1, V+A=1, P0=0`.  This proves the current frontend path is
callable; it is not the full frozen-suite reproduction gate.

Standard VNNLIB frontend smoke evidence after the latest cleanup:

```bash
python -m act.pipeline --verify vnnlib --category sat_relu --max-instances 1 \
  --solvers hybridz --device cpu --dtype float64 --hybridz-timeout 30 \
  --hybridz-results-dir /tmp/hyzor_stage2_vnnlib_hybridz_smoke
```

This normal `--solvers hybridz` entry produced
`/tmp/hyzor_stage2_vnnlib_hybridz_smoke/sat_relu_hybridz_summary.csv` with
`N=1, CERT=0, ADV=1, V+A=1, P0=0`, and the detail CSV records
`solver=hybridz`, `engine=dense_hz_objbound`, `hz_verdict=UNSAFE`,
`hz_witness_source=milp_objective_bound`, and
`witness_replay=model_fn_replay_unsafe:dense_col_ids`.  This is separate
evidence that the first-class frontend solver mode is callable without using
the benchmark-suite runner.

Full frozen gate evidence after the latest cleanup:

```bash
OUT=/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25
python -m act.pipeline --verify hybridz-benchmark --category frozen \
  --hybridz-require-frozen-match --hybridz-results-dir "$OUT" \
  --device cpu --dtype float64
python -m act.back_end.hybridz_config "$OUT/FINAL_HYBRIDZ_RESULTS.csv"
cd "$OUT" && sha256sum -c _MANIFEST.sha256
```

The final gate returned `PASSED`, `FROZEN_REPRO_COMPARISON.json` reports
`"ok": true`, `FINAL_HYBRIDZ_RESULTS.csv` passes the backend frozen oracle, and
the manifest check is clean.  The final gate used strict clean-log reuse:
each benchmark directory was accepted only when every frozen summary field
matched the backend oracle.  This is intentionally different from sample-level
rescue and does not promote any post-HybridZ result.

The remaining large diff is therefore not a `scripts/` dependency problem; it is
a consolidation problem.  The frontend can reproduce the frozen table from
packaged modules, but too much sparse exact-HZ probe logic still lives in
`act.pipeline` rather than being reduced into:

- `act/back_end/hybridz_tf/sparse_ops.py`
- `act/back_end/solver/sparse_hz.py`
- `act/back_end/solver/solver_hz_verdict.py`
- small pipeline orchestration wrappers

## Required Follow-Up Before Calling Stage-II Fully Done

0. First consolidation passes completed after this audit:
   `act/pipeline/hybridz_sparse_exact_probe.py` now reuses backend
   `SparseHZono` as its carrier, backend sparse frame/gather/merge helpers,
   backend input-spec construction, backend
   Conv/Dense/AvgPool/MaxPool-candidate/scale/bias/linear helpers, backend
   same-frame add/sub helpers, backend UPSAMPLE row maps, backend
   sigmoid/tanh primitives, backend base-HZ feasibility checking, and the
   backend S-curve cut matrix builders, and the save/load HZ debug bypass was
   removed from the packaged probe, max-layer partial propagation was removed
   from the packaged probe, and manual `--fix-binaries` / base-witness debug
   entry points and the old fixed-phase bound override plumbing were removed,
   and sparse-probe MILP was collapsed to the cutoff formulation used by the
   product worker.  LP-relaxation solutions are no longer returned as witness
   candidates or MIP starts; sparse MIP starts are limited to the constructive
   base point.  The local duplicate probe code dropped from 5751 to 4707
   lines
   while preserving the packaged probe self-test, sparse-ops structural
   self-test, and a backend UPSAMPLE 3D/4D row-map regression.

   A direct sparse-worker smoke also now succeeds after fixing the probe
   layer-index GC hook:

   ```bash
   python -m act.pipeline.hybridz_sparse_worker --bench sat_relu --iid 0 \
     --lp-queries 1 --milp-timeout 5 --worker-timeout 40
   ```

   This returned `ADV` with `ort_verified=true`, `P0=false`, and
   `milp_status=TARGET:Optimal`.

   Follow-up cleanup on 2026-06-27 also deduplicated repeated HiGHS heuristic
   option tuples in `hybridz_benchmark_runner.py`, common sparse probe keyword
   arguments in `hybridz_sparse_worker.py`, repeated CLI `name=value`
   solver-option parsing in the sparse and projected pipeline entry points,
   and repeated VNNLIB spec/witness helper functions in the sparse and
   projected pipeline entry points.  The sparse worker wrapper dropped from
   517 to 452 lines, the sparse probe dropped from 4707 to 4622 lines, the
   projected one-ReLU branch dropped from 788 to 744 lines, and
   `hybridz_projected_utils.py` dropped from 186 to 118 lines.  Regression
   evidence:
   `python -m act.pipeline.hybridz_benchmark_runner`,
   `python -m act.pipeline.hybridz_sparse_exact_probe --self-test`, and
   `python -m act.pipeline.hybridz_sparse_worker --bench sat_relu --iid 0
   --lp-queries 1 --milp-timeout 5 --worker-timeout 45`, which returned
   `ADV`, `p0=false`, `ort_verified=true`.  CLI option parsing was also
   checked through `py_compile`, `--help`, and invalid-option smoke tests for
   both sparse and projected entry points.  The projected helper path was
   checked with `python -m act.pipeline.hybridz_projected_relu_mip --bench
   safenlp_2024 --iid 0 --lp-queries 1 --milp-timeout 5 --mip-start none`.
   The sparse SOFTMAX simplex relaxation and var-var MATMUL product-interval
   lift have also been moved from the packaged probe into
   `act/back_end/hybridz_tf/sparse_ops.py`, with backend structural self-tests
   covering the simplex equality, center witness construction, and product
   interval center witness.  The sparse exact ReLU graph construction now also
   lives in the backend and exposes optional `return_info` metadata so the
   packaged probe can preserve its existing witness-extension bookkeeping while
   keeping tight-LP policy in the pipeline wrapper.
   Regression evidence: `python -m act.back_end.hybridz_tf.sparse_ops`,
   `python -m act.pipeline.hybridz_sparse_exact_probe --self-test`,
   `python -m act.pipeline.hybridz_benchmark_runner`, and the same `sat_relu`
   sparse-worker smoke above.

   The suite CSV reader, deterministic SHA256 manifest writer, benchmark and
   suite ICSE/CSV/JSON export helpers, frozen cross-tool ranking/export
   helpers, frozen reproduction comparison gate, and failure taxonomy/P0
   reporting helpers have also been moved from `hybridz_benchmark_runner.py`
   into `hybridz_results.py`.  This keeps the benchmark runner focused on
   scheduling and portfolio execution.  The runner dropped from 2562 to 1783
   lines while preserving `python -m
   act.pipeline.hybridz_results`, `python -m act.pipeline.hybridz_benchmark_runner`,
   the standard `--verify vnnlib --solvers hybridz` smoke, and the frozen
   oracle/manifest checks.

   The exact ReLU tight-LP bound helper used for sparse phase fixing was also
   moved from `hybridz_sparse_exact_probe.py` into
   `act/back_end/hybridz_tf/sparse_ops.py` as backend sparse-HZ operator
   support.  The probe dropped from 4017 to 3850 lines.  Regression evidence:
   `python -m act.back_end.hybridz_tf.sparse_ops`, `python -m
   act.pipeline.hybridz_sparse_exact_probe --self-test`, `python -m
   act.pipeline.hybridz_sparse_worker --bench sat_relu --iid 0 --lp-queries 1
   --milp-timeout 5 --worker-timeout 45`, the standard `--verify vnnlib
   --solvers hybridz` smoke, and the frozen oracle check.

   The sparse MILP row-range infeasibility and feasibility-based bound
   tightening presolve helpers were also moved from
   `hybridz_sparse_exact_probe.py` into `act/back_end/solver/solver_hz_verdict.py`.
   The probe dropped from 3850 to 3636 lines while preserving
	   `python -m act.back_end.solver.solver_hz_verdict`, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, the same `sat_relu`
	   sparse-worker smoke, the standard `--verify vnnlib --solvers hybridz`
	   smoke, and the frozen oracle check.

	   The solver start-vector conversion and HiGHS continuous-relaxation EMPTY
	   precheck were also moved from `hybridz_sparse_exact_probe.py` into
	   `act/back_end/solver/solver_hz_verdict.py`.  The probe dropped from 3636
	   to 3523 lines.  Regression evidence: `py_compile` for the touched files,
	   `python -m act.back_end.solver.solver_hz_verdict`, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, the `sat_relu`
	   sparse-worker smoke, the packaged `--verify hybridz-benchmark` smoke, the
	   standard `--verify vnnlib --solvers hybridz` smoke, and
	   `cd "$OUT" && sha256sum -c _MANIFEST.sha256` for the frozen artifact.

	   The sparse LP min-margin prefilter was also moved from
	   `hybridz_sparse_exact_probe.py` into `solver_hz_verdict.py`, and the input
	   box center/radius/index extraction helper moved into `hybridz_spec_utils.py`.
	   The probe dropped from 3523 to 3437 lines.  Regression evidence:
	   `py_compile` for the touched files, `python -m
	   act.back_end.solver.solver_hz_verdict`, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, the `sat_relu`
	   sparse-worker smoke, the packaged `--verify hybridz-benchmark` smoke, the
	   standard `--verify vnnlib --solvers hybridz` smoke, and the frozen manifest
	   check.

	   The compressed/pruned uniform S-curve path now calls backend
	   `sparse_hz_apply_scurve_piecewise(..., return_info=True)` instead of
	   rebuilding that counted nonlinear operator entirely in the probe.  The
	   local curvature-grid fallback remains in the probe until backend parity is
	   implemented for that diagnostic mode.  Regression evidence: backend
	   `sparse_ops` self-test checks the new metadata, probe self-test compares
	   backend vs local-pruned exact supports on a toy S-curve HZ, `sat_relu`
	   sparse-worker smoke returned `ADV/P0=false`, a `dist_shift_2023` compressed
	   S-curve smoke returned `UNKNOWN/P0=false` without construction errors, both
	   frontend smokes produced `N=1, ADV=1, P0=0`, and the frozen manifest check
	   remained clean.

	   The sparse HiGHS and SCIP MILP cutoff engines were then moved from
	   `hybridz_sparse_exact_probe.py` into
	   `act/back_end/solver/solver_hz_verdict.py`.  The probe now imports these
	   solver-layer engines and keeps only branch policy / frontend orchestration.
	   The probe dropped from 3437 to 2072 lines.  Regression evidence:
	   `py_compile` for the touched solver/probe files, `python -m
	   act.back_end.solver.solver_hz_verdict`, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, `python -m
	   act.pipeline.hybridz_sparse_worker --bench sat_relu --iid 0 --lp-queries 1
	   --milp-timeout 5 --worker-timeout 45`, the compressed `dist_shift_2023`
	   S-curve sparse-worker smoke, the packaged `--verify hybridz-benchmark`
	   smoke, the standard `--verify vnnlib --solvers hybridz` smoke, and
	   `cd "$OUT" && sha256sum -c _MANIFEST.sha256`.

	   The remaining compressed/pruned S-curve fallback for curvature grids was
	   also moved into `act/back_end/hybridz_tf/sparse_ops.py`.  The backend
	   `sparse_hz_apply_scurve_piecewise` now accepts `grid="uniform"` or
	   `grid="curvature"` and returns the same witness metadata needed by the
	   sparse probe.  The probe now calls backend S-curve construction for both
	   counted uniform branches and diagnostic curvature branches, and the local
	   `_sigmoid_piecewise_pruned` implementation was deleted.  The probe dropped
	   from 2072 to 1720 lines.  Regression evidence: `py_compile` for touched
	   files, `python -m act.back_end.hybridz_tf.sparse_ops`, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, the `sat_relu`
	   sparse-worker smoke returning `ADV/P0=false`, uniform and curvature
	   `dist_shift_2023` compressed S-curve sparse-worker smokes returning
	   `UNKNOWN/P0=false` without construction errors, both frontend smokes
	   producing `N=1, ADV=1, P0=0`, and the frozen manifest check.

	   The sparse structural row-map helpers for nearest-neighbor UPSAMPLE,
	   SLICE, and GATHER were also moved into
	   `act/back_end/hybridz_tf/sparse_ops.py`.  The sparse probe now calls
	   backend `sparse_upsample_nearest_row_indices`,
	   `sparse_slice_row_indices`, and `sparse_gather_row_indices`, and no longer
	   imports dense `tf_mlp` private helper functions.  The probe dropped from
	   1720 to 1710 lines.  Regression evidence: `py_compile` for touched files,
	   `python -m act.back_end.hybridz_tf.sparse_ops` with explicit row-map toy
	   checks, `python -m act.pipeline.hybridz_sparse_exact_probe --self-test`,
	   the `sat_relu` sparse-worker smoke returning `ADV/P0=false`, both
	   frontend smokes producing `N=1, ADV=1, P0=0`, and the frozen manifest
	   check.

	   The full/non-pruned sparse S-curve construction was also moved from the
	   packaged probe into `act/back_end/hybridz_tf/sparse_ops.py` as
	   `sparse_hz_apply_scurve_piecewise_full`.  The probe now only selects
	   between backend pruned and backend full S-curve APIs and preserves the
	   existing witness metadata side channel.  The probe dropped from 1710 to
	   1397 lines.  Regression evidence: `py_compile` for touched files,
	   `python -m act.back_end.hybridz_tf.sparse_ops`, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, the `sat_relu`
	   sparse-worker smoke returning `ADV/P0=false/ort_verified=true`, the
	   compressed `dist_shift_2023` S-curve sparse-worker smoke returning
	   `UNKNOWN/P0=false` with final HZ `n=10 ng=5.3k nb=2.6k nc=1.8k`, both
	   frontend smokes producing `N=1, ADV=1, P0=0`, and
	   `sha256sum -c _MANIFEST.sha256` on the frozen artifact.

	   Constructive center witness extension for exact ReLU and S-curve layers,
	   plus the final equality/inequality residual check, was also moved from the
	   packaged probe into `act/back_end/hybridz_tf/sparse_ops.py`.  The probe now
	   keeps only witness state and calls backend helpers; the acceptance
	   tolerance and error messages remain equivalent.  The probe dropped from
	   1397 to 1159 lines.  Regression evidence: `py_compile` for touched files,
	   `python -m act.back_end.hybridz_tf.sparse_ops` with explicit ReLU, full
	   S-curve, and pruned S-curve witness checks, `python -m
	   act.pipeline.hybridz_sparse_exact_probe --self-test`, the `sat_relu`
	   sparse-worker smoke returning `ADV/P0=false/ort_verified=true` with
	   `constructive_center max_eq=0 max_ub=-inf`, the compressed
	   `dist_shift_2023` S-curve sparse-worker smoke returning `UNKNOWN/P0=false`
	   with `constructive_center max_eq=1.36e-12 max_ub=0`, and both frontend
	   smokes producing `N=1, ADV=1, P0=0`.

	   The sparse exact-ReLU application helper that combines LP bound
	   tightening, the `off-only` active-phase preservation policy, and the
	   backend exact ReLU call was also moved into
	   `act/back_end/hybridz_tf/sparse_ops.py` as
	   `sparse_hz_apply_relu_exact_tightened`.  The probe now keeps only a thin
	   compatibility wrapper for existing logging metadata.  The probe dropped
	   from 1159 to 1135 lines.  Regression evidence: `py_compile` for touched
	   files, `python -m act.back_end.hybridz_tf.sparse_ops`,
	   `python -m act.pipeline.hybridz_sparse_exact_probe --self-test`, a direct
	   tight-ReLU probe smoke on `sat_relu iid0` returning `ADV/P0=false` with
	   `tightLP solved/improved/fix_on/fix_off=68/0/0/0`, the standard
	   `sat_relu` sparse-worker smoke returning `ADV/P0=false/ort_verified=true`,
	   the compressed `dist_shift_2023` S-curve sparse-worker smoke returning
	   `UNKNOWN/P0=false`, both frontend smokes producing `N=1, ADV=1, P0=0`,
	   and the frozen manifest check.

1. Audit `act/pipeline/hybridz_sparse_exact_probe.py` function families.
   Move reusable exact-HZ propagation and MILP export helpers into backend
   modules, and delete diagnostic branches that are not counted in the pure-HZ
   frozen table.

2. Continue shrinking only if scheduling logic exposes more pure helpers.
   The main reporting/cross-tool/frozen/taxonomy output logic has moved to
   `hybridz_results.py`; remaining runner code is primarily CLI scheduling,
   branch portfolio execution, strict frozen reuse, and resource gating.

3. Re-check specialized product branches.
   `hybridz_projected_relu_mip.py`, sparse census, and sparse worker wrappers
   are valid current product/future-work modules, but each needs an explicit
   keep/move/delete decision after the frozen result is accepted.

4. Keep `scripts/` local until this consolidation is complete.
   Current state satisfies "no product dependency on scripts", but not yet
   "all useful script logic is minimized into backend layers".

## Current Frozen Baseline

Final accepted pure-HybridZ table:

`1780 / 2213 = 980 CERT + 800 ADV`, `P0=0`, `ERROR=0`.

Final files:

- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FINAL_HYBRIDZ_RESULTS.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FINAL_CROSS_TOOL_RANKING.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FROZEN_REPRO_COMPARISON.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FROZEN_REPRO_COMPARISON.json`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/_MANIFEST.sha256`

Operational caveat: `metaroom_2023` is wall-sensitive.  A 3-worker rerun
produced one extra timeout by losing `iid29`; a direct recheck of `iid29`
returned `19/19 CERT`.  Keep `metaroom_2023` at benchmark-wide worker count 1
for clean recomputation, or use the strict frozen reuse gate.
