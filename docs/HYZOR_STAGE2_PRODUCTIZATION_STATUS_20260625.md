# HyZor Stage-II Productization Status

Date: 2026-06-25

Current authoritative result artifact:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25`

Root-level final copies are also stored under
`/data1/Kane/ICSE/act_hybridz_soundfix_20260625` with the
`*_20260627_FINAL` suffix.

Current headline:

`1780/2213 = 980 CERT + 800 ADV`, `P0=0`, `ERROR=0`.

The older 2026-06-24 `1768/2213` headline and the 2026-06-25
`1763/2213` soundfix table are superseded for reporting.  The metaroom row is
`94 CERT / 1 ADV / 5 TIMEOUT = 95 V+A`; it remains rank `#1`.

## Current Evidence

| Requirement axis | Current evidence | Status |
|---|---|---|
| Frozen results saved in ICSE-style CSVs | strict frontend-gate source: `FINAL_HYBRIDZ_RESULTS.csv`, `FINAL_CROSS_TOOL_RANKING.csv`, `FROZEN_REPRO_COMPARISON.csv/json`, `_MANIFEST.sha256` under `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25`; root-level copied final tables use the `*_20260627_FINAL` suffix; previous 2026-06-25 soundfix tables are retained as historical provenance | done |
| Soundness aggregation bug fixed | `act/pipeline/hybridz_benchmark_runner.py` collapses split VNNLIB summaries as: any ADV -> ADV; CERT only if all emitted queries CERT; otherwise unresolved | done |
| P0 propagation in productized reporting | `act/pipeline/hybridz_results.py` and `act/pipeline/hybridz_benchmark_runner.py` now propagate explicit `p0/P0` flags into summary, ICSE detail, and taxonomy outputs instead of writing hard-coded zeros | done |
| Product path no longer imports `scripts/` | migration matrix scan records no production `act/` imports from `scripts/`; remaining references are docs/comments/local debug helpers | done for import dependency |
| Frontend entrypoints exist | `--solvers hybridz`, `--verify hybridz-benchmark`, `--hybridz-engine`, `--hybridz-require-frozen-match`, result-dir, worker/time controls are in `act/pipeline/cli.py` | present |
| Frontend benchmark smoke | `python -m act.pipeline --verify hybridz-benchmark --category acasxu_2023 --max-instances 1 --hybridz-workers 1 --hybridz-timeout-cap 5` wrote detail/summary/ICSE CSVs and manifest under `/tmp/act_hybridz_stage2_commit_boundary_smoke` with `P0=0`, `ERROR=0` | done as smoke |
| Focused regression checks | see test log below | done |
| Full frozen frontend reproduction | clean full `--verify hybridz-benchmark --category frozen --hybridz-require-frozen-match` gate completed under `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25`; `FROZEN_REPRO_COMPARISON.json` reports `ok: true`, 12/12 benchmark rows `match`, `P0=0`, and `ERROR=0` | done |
| Linearizenn packaged frontend rerun | after adding the benchmark-wide `linear_portfolio_m360` branch, full packaged rerun under `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/linearizenn_productized_m360_20260627` produced `39 CERT + 1 ADV = 40/60`, `P0=0`, `ERROR=0`; `iid13/41/46/51/52` are recovered by `linear_portfolio_m360`, while `iid34` is recovered by `normal` | frozen-green for bench |
| Git-tracked productization state | product/doc boundary committed in `4ed5712e4` (`Productize HybridZ stage II path`), post-status commit `78ab54b66`, packaged worker modules committed in `8738f50f6` (`Productize HybridZ packaged workers`), and subsequent hardening commits through `978d39597` expose public HybridZ helper APIs and remove stale probe shims; `scripts/` remains local/untracked and absent from commits | done for current product path |
| Relative-to-upstream minimal diff audit | tracked diff has been measured after the product boundary commit and refreshed after helper-publication cleanup; remaining large diff is explained as HybridZ product path, sparse backend, runner/reporting, data plumbing, and audit docs | done for freeze handoff |

## Product Files Committed

These files are part of the current product path or its documentation and must
not remain accidental local-only state at final handoff.  They were committed in
`4ed5712e4`:

- `FULLRUN_HANDOFF.md`
- `act/back_end/hybridz_config.py`
- `act/back_end/hybridz_tf/__main__.py`
- `act/back_end/hybridz_tf/sparse_ops.py`
- `act/back_end/solver/sparse_hz.py`
- `act/pipeline/hybridz_benchmark_runner.py`
- `act/pipeline/hybridz_projected_relu_mip.py`
- `act/pipeline/hybridz_projected_utils.py`
- `act/pipeline/hybridz_results.py`
- `docs/`

`scripts/` remains local/untracked by user request.  The migration target is not
to commit the local scripts, but to either move selected pure-HZ behavior into
`act/` or mark the script as excluded/provenance/future work.

## Packaged Worker Modules Included

The current runner imports these package modules, so they are product-path files
and must not remain accidental local-only state at final handoff.  They were
committed in `8738f50f6`:

- `act/pipeline/hybridz_full_worker.py`
- `act/pipeline/hybridz_sparse_worker.py`
- `act/pipeline/hybridz_sparse_exact_probe.py`
- `act/pipeline/hybridz_sparse_census.py`

They replace legacy `scripts/` workers in the frontend command path.  They are
not the local `scripts/` directory.

## Scripts Migration Conclusion

The current migration matrix classifies script families as follows:

| Family | Status |
|---|---|
| full-run scheduling/export/result ledger | migrated into `hybridz_config.py`, `hybridz_benchmark_runner.py`, and `hybridz_results.py`; legacy scripts kept only as local provenance after the full frontend frozen rerun |
| dense/sparse exact-HZ verdict and replay guard | migrated into `solver_hz_verdict.py`, `sparse_hz.py`, `sparse_ops.py`, and `verifier.py` |
| selected safenlp projected one-hidden-ReLU branch | migrated into `act/pipeline/hybridz_projected_relu_mip.py` as a benchmark-wide portfolio branch |
| guided LP-witness rescue / ORT promotion | intentionally excluded |
| CIFAR lazy/matrix-free census, OBBT, primal graph MIP, S-curve pair experiments | future work only, not counted in current pure-HZ artifact |

## Focused Test Log

Commands run with `/data1/Kane/miniconda3/envs/act-py312/bin/python`:

```bash
python -m act.pipeline.hybridz_results
python -m act.pipeline.hybridz_benchmark_runner
python -m act.back_end.hybridz_tf
python -m act.back_end.hybridz_config
python -m act.back_end.solver.solver_hz_verdict
git diff --check
git diff --cached --check
git diff --cached --name-only | rg '^scripts/' || true
sha256sum -c /data1/Kane/ICSE/act_hybridz_soundfix_20260625/_MANIFEST.sha256
rg -n "(^|\s)(import|from) scripts\b|hz_full_driver|hz_full_worker|hz_sparse_worker|cifar_sparse_exact_probe" act
/usr/bin/timeout 180 python -m act.pipeline --verify hybridz-benchmark \
  --category acasxu_2023 --max-instances 1 --hybridz-workers 1 \
  --hybridz-timeout-cap 5 \
  --hybridz-results-dir /tmp/act_hybridz_stage2_commit_boundary_smoke
sha256sum -c /tmp/act_hybridz_stage2_commit_boundary_smoke/_MANIFEST.sha256
/usr/bin/timeout 180 python -m act.pipeline --verify hybridz-benchmark \
  --category acasxu_2023 --max-instances 1 --hybridz-workers 1 \
  --hybridz-timeout-cap 5 \
  --hybridz-results-dir /tmp/act_hybridz_stage2_staged_smoke
sha256sum -c /tmp/act_hybridz_stage2_staged_smoke/_MANIFEST.sha256
/usr/bin/timeout 180 python -m act.pipeline --verify hybridz-benchmark \
  --category acasxu_2023 --max-instances 1 --hybridz-workers 1 \
  --hybridz-timeout-cap 5 \
  --hybridz-results-dir /tmp/act_hybridz_stage2_postcommit_smoke
sha256sum -c /tmp/act_hybridz_stage2_postcommit_smoke/_MANIFEST.sha256
```

Observed result:

- `PASS _test_hybridz_results`
- `PASS _test_hybridz_benchmark_runner`
- `PASS _test_sparse_matmul_const_propagation`
- `PASS _test_hz_cnn_exact_maxpool_only`
- `PASS _test_hz_mul_exact_point_and_var_drop`
- `PASS _test_hybridz_config`
- `PASS _test_sparse_hz_verdict_parity`
- `git diff --check` clean
- `git diff --cached --check` clean
- staged diff has no `scripts/` path
- soundfix manifest clean
- `act/` scan found no production `scripts` import; the only hit is a
  `hybridz_benchmark_runner.py` self-test assertion that legacy worker names
  are absent from generated commands
- frontend smoke completed and emitted `acasxu_2023_hybridz_summary.csv`,
  `acasxu_2023.csv`, ICSE detail/index CSVs, JSON summary, and `_MANIFEST.sha256`
- frontend smoke manifest clean
- staged-state frontend smoke under `/tmp/act_hybridz_stage2_staged_smoke`
  completed with `N=1, UNKNOWN=1, P0=0, ERROR=0` and clean manifest
- post-commit frontend smoke under `/tmp/act_hybridz_stage2_postcommit_smoke`
  completed with `N=1, UNKNOWN=1, P0=0, ERROR=0` and clean manifest
- latest post-worker-module gates on 2026-06-26:
  `hybridz_benchmark_runner`, `hybridz_sparse_exact_probe --self-test`,
  `solver_hz_verdict`, `hybridz_results`, relevant `py_compile`, and
  `git diff --check` all passed
- latest worker hardening on 2026-06-26 removed package full-worker guided
  LP/phase-fix CLI modes and removed the sparse probe's disabled
  input/binary-split branch; direct LP-relaxation witness replay now errors
  unless exact MILP witness-only replay is requested
- frontend package-worker smoke on `acasxu_2023 --max-instances 1` completed
  under `/tmp/act_hybridz_stage2_worker_product_smoke_1782400731` with
  `P0=0`, `ERROR=0`, and a clean manifest
- final full frontend frozen gate completed under
  `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/frontend_frozen_gate_20260627_pscost25`;
  `FROZEN_REPRO_COMPARISON.json` reports `ok: true`, 12/12 rows `match`,
  `P0=0`, and `ERROR=0`
- subsequent full packaged `linearizenn_2024` rerun with the productized
  `linear_portfolio_m360` branch produced
  `39 CERT / 1 ADV / 40 V+A / 0 P0 / 0 ERROR / 20 unsolved`; recovered
  frozen-gap rows are `iid13`, `iid41`, `iid46`, `iid51`, and `iid52`
- `cora_2024` in the full frontend reproduction produced
  `20 CERT / 20 ADV / 40 V+A / 0 P0 / 0 ERROR / 140 unsolved`; the extra
  ADV rows come from sparse exact-HZ MILP target/witness records with
  `witness_checked=True`, `real_unsafe=True`, and are counted as pure HybridZ
- `relusplitter` in the full frontend reproduction produced
  `43 CERT / 2 ADV / 45 V+A / 0 P0 / 0 ERROR / 175 unsolved`; the positive
  delta is from the normal full-HZ branch with `hz_dropped=false`
- `dist_shift_2023` in the full frontend reproduction produced
  `70 CERT / 0 ADV / 70 V+A / 0 P0 / 0 ERROR / 2 unsolved`; the
  `TIMEOUT/UNKNOWN` split was `0/2` instead of the frozen table's `2/0`, an
  audit-only delta like the earlier productized rerun
- `tllverifybench_2023` in the full frontend reproduction produced
  `5 CERT / 12 ADV / 17 V+A / 0 P0 / 0 ERROR / 15 unsolved`; the
  `TIMEOUT/UNKNOWN` split was `0/15` instead of the frozen table's `14/1`,
  audit-only for the current reproduction gate

## Remaining Cleanup / Future-Work Items

1. Keep `scripts/` local/untracked unless a later change deliberately migrates
   one script's pure-HZ behavior into `act/` with tests.
2. Do not delete local legacy sparse scripts until their remaining research
   value is either migrated into backend tests or explicitly archived as
   provenance.
3. Keep future-work directions separate from the counted artifact: tighter
   sound S-curve, sparse Schur/block presolve, OBBT-lite, lazy exact-HZ,
   open-source solver portfolio improvements, memory-governed
   benchmark-level worker profiles for serial tail benches such as `acasxu`,
   `cora`, and `relusplitter`.
