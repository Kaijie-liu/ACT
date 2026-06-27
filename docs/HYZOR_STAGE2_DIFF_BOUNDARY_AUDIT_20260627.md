# HyZor Stage-II Diff Boundary Audit

Date: 2026-06-27

Baseline: `upstream/main` after `git fetch upstream`.

Current branch: `hz-cam-1` after sparse-probe consolidation passes, frozen-gate
productization, sparse-worker option deduplication, and package-local HybridZ
CLI option/spec helper consolidation, plus local untracked `scripts/`.
The local `scripts/` directory is not tracked and is not part of this diff.

## Summary

Current upstream diff after `git fetch upstream` on 2026-06-27:

`80 files changed, 23915 insertions(+), 2188 deletions(-)`

Directory-level split:

| Area | Files | Added | Deleted | Status |
|---|---:|---:|---:|---|
| `act/pipeline` | 19 | 10476 | 159 | largest remaining consolidation target |
| `act/back_end/hybridz_tf` | 8 | 4988 | 292 | core HZ operator/product path |
| `docs` | 16 | 3868 | 0 | audit/provenance/future-work docs |
| `act/back_end/solver` | 5 | 2177 | 423 | verdict/sparse HZ solver path |
| `act/back_end/other` | 17 | 1691 | 957 | frontend/backend integration hooks |
| `act/front_end` | 7 | 508 | 334 | benchmark/data loading integration |
| `FULLRUN_HANDOFF.md` | 1 | 200 | 0 | run handoff/provenance |

Largest files by changed lines:

| File | Added | Deleted | Interpretation |
|---|---:|---:|---|
| `act/pipeline/hybridz_sparse_exact_probe.py` | 4448 | 0 | biggest remaining package-level prototype; first post-freeze consolidation target is partially reduced |
| `act/pipeline/hybridz_benchmark_runner.py` | 2562 | 0 | product runner, branch portfolio, frozen comparison, ICSE export |
| `act/back_end/hybridz_tf/sparse_ops.py` | 2500 | 0 | sparse exact-HZ propagation core; SOFTMAX simplex operator moved here from the probe |
| `act/back_end/solver/solver_hz_verdict.py` | 1563 | 0 | exact verdict MILP and open-source solver portfolio |
| `act/back_end/hybridz_tf/tf_mlp.py` | 1253 | 246 | dense exact ReLU/compressed ReLU and nonlinear operators |
| `act/pipeline/hybridz_projected_relu_mip.py` | 744 | 0 | safenlp projected exact-ReLU branch; still pure-HZ but specialized |
| `act/back_end/verifier.py` | 740 | 13 | frontend solver integration and metadata/soundness guards |
| `act/pipeline/hybridz_sparse_census.py` | 660 | 0 | diagnostic/census path; not needed for frozen one-command verification |
| `act/pipeline/hybridz_sparse_worker.py` | 452 | 0 | sparse worker wrapper; common probe options deduplicated |
| `act/back_end/hybridz_config.py` | 503 | 0 | frozen oracle and benchmark profiles |

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
   The sparse SOFTMAX simplex relaxation has also been moved from the packaged
   probe into `act/back_end/hybridz_tf/sparse_ops.py`, with a backend structural
   self-test covering the simplex equality and center witness construction.
   Regression evidence: `python -m act.back_end.hybridz_tf.sparse_ops`,
   `python -m act.pipeline.hybridz_sparse_exact_probe --self-test`,
   `python -m act.pipeline.hybridz_benchmark_runner`, and the same `sat_relu`
   sparse-worker smoke above.

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

- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FINAL_HYBRIDZ_RESULTS.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FINAL_CROSS_TOOL_RANKING.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FROZEN_REPRO_COMPARISON.csv`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/FROZEN_REPRO_COMPARISON.json`
- `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25/_MANIFEST.sha256`

Operational caveat: `metaroom_2023` is wall-sensitive.  A 3-worker rerun
produced one extra timeout by losing `iid29`; a direct recheck of `iid29`
returned `19/19 CERT`.  Keep `metaroom_2023` at benchmark-wide worker count 1
for clean recomputation, or use the strict frozen reuse gate.
