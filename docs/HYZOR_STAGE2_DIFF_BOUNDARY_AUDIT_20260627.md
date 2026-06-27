# HyZor Stage-II Diff Boundary Audit

Date: 2026-06-27

Baseline: `upstream/main` after `git fetch upstream`.

Current branch: `hz-cam-1` after sparse-probe consolidation passes plus local
untracked `scripts/`.
The local `scripts/` directory is not tracked and is not part of this diff.

## Summary

Current upstream diff:

`60 files changed, 24044 insertions(+), 639 deletions(-)`

Directory-level split:

| Area | Files | Added | Deleted | Status |
|---|---:|---:|---:|---|
| `act/pipeline` | 15 | 11055 | 141 | largest remaining consolidation target |
| `act/back_end/hybridz_tf` | 8 | 4782 | 292 | core HZ operator/product path |
| `docs` | 16 | 3785 | 0 | audit/provenance/future-work docs |
| `act/back_end/solver` | 4 | 2128 | 51 | verdict/sparse HZ solver path |
| `act/back_end/other` | 11 | 1635 | 54 | frontend/backend integration hooks |
| `act/front_end` | 5 | 458 | 101 | benchmark/data loading integration |
| `FULLRUN_HANDOFF.md` | 1 | 200 | 0 | run handoff/provenance |

Largest files by changed lines:

| File | Added | Deleted | Interpretation |
|---|---:|---:|---|
| `act/pipeline/hybridz_sparse_exact_probe.py` | 5056 | 0 | biggest remaining package-level prototype; first post-freeze consolidation target is partially reduced |
| `act/pipeline/hybridz_benchmark_runner.py` | 2499 | 0 | product runner, branch portfolio, frozen comparison, ICSE export |
| `act/back_end/hybridz_tf/sparse_ops.py` | 2294 | 0 | sparse exact-HZ propagation core |
| `act/back_end/solver/solver_hz_verdict.py` | 1563 | 0 | exact verdict MILP and open-source solver portfolio |
| `act/back_end/hybridz_tf/tf_mlp.py` | 1253 | 246 | dense exact ReLU/compressed ReLU and nonlinear operators |
| `act/pipeline/hybridz_projected_relu_mip.py` | 788 | 0 | safenlp projected exact-ReLU branch; still pure-HZ but specialized |
| `act/back_end/verifier.py` | 740 | 13 | frontend solver integration and metadata/soundness guards |
| `act/pipeline/hybridz_sparse_census.py` | 660 | 0 | diagnostic/census path; not needed for frozen one-command verification |
| `act/pipeline/hybridz_sparse_worker.py` | 517 | 0 | sparse worker wrapper; should shrink after probe logic moves deeper |
| `act/back_end/hybridz_config.py` | 503 | 0 | frozen oracle and benchmark profiles |

## Product Boundary Assessment

The current product path no longer imports `scripts/`.  A scan over `act/`
finds only the runner self-test assertion that generated commands must not
contain `scripts/` paths.

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
   removed from the packaged probe.  The local duplicate probe code dropped
   from 5751 to 5056 lines
   while preserving the packaged probe self-test, sparse-ops structural
   self-test, and a backend UPSAMPLE 3D/4D row-map regression.

1. Audit `act/pipeline/hybridz_sparse_exact_probe.py` function families.
   Move reusable exact-HZ propagation and MILP export helpers into backend
   modules, and delete diagnostic branches that are not counted in the pure-HZ
   frozen table.

2. Shrink `act/pipeline/hybridz_benchmark_runner.py`.
   Keep the CLI runner and frozen comparison, but move static reporting helpers
   and cross-tool table construction into a smaller results/reporting module if
   that reduces duplication without changing outputs.

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

- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/FINAL_HYBRIDZ_RESULTS_20260627_FINAL.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/FINAL_CROSS_TOOL_RANKING_20260627_FINAL.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/FROZEN_REPRO_COMPARISON_20260627_FINAL.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/_FINAL_20260627_MANIFEST.sha256`
