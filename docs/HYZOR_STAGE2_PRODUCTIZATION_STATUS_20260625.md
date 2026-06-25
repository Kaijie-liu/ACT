# HyZor Stage-II Productization Status

Date: 2026-06-25

Current authoritative result artifact:

`/data1/Kane/ICSE/act_hybridz_soundfix_20260625`

Current headline:

`1763/2213 = 977 CERT + 786 ADV`, `P0=0`, `ERROR=0`.

The older 2026-06-24 `1768/2213` headline is superseded for reporting.  The
metaroom row is now `94 CERT / 1 ADV / 5 TIMEOUT = 95 V+A`; it remains rank
`#1`.

## Current Evidence

| Requirement axis | Current evidence | Status |
|---|---|---|
| Frozen results saved in ICSE-style CSVs | `FINAL_HYBRIDZ_RESULTS_20260625_SOUNDFIX.csv`, `FINAL_CROSS_TOOL_RANKING_20260625_SOUNDFIX.csv`, `_CROSS_TOOL_SUMMARY_20260625_SOUNDFIX.csv`, `_MANIFEST.sha256` under the soundfix artifact | done |
| Soundness aggregation bug fixed | `act/pipeline/hybridz_benchmark_runner.py` collapses split VNNLIB summaries as: any ADV -> ADV; CERT only if all emitted queries CERT; otherwise unresolved | done |
| P0 propagation in productized reporting | `act/pipeline/hybridz_results.py` and `act/pipeline/hybridz_benchmark_runner.py` now propagate explicit `p0/P0` flags into summary, ICSE detail, and taxonomy outputs instead of writing hard-coded zeros | done |
| Product path no longer imports `scripts/` | migration matrix scan records no production `act/` imports from `scripts/`; remaining references are docs/comments/local debug helpers | done for import dependency |
| Frontend entrypoints exist | `--solvers hybridz`, `--verify hybridz-benchmark`, `--hybridz-engine`, `--hybridz-require-frozen-match`, result-dir, worker/time controls are in `act/pipeline/cli.py` | present |
| Frontend benchmark smoke | `python -m act.pipeline --verify hybridz-benchmark --category acasxu_2023 --max-instances 1 --hybridz-workers 1 --hybridz-timeout-cap 5` wrote detail/summary/ICSE CSVs and manifest under `/tmp/act_hybridz_stage2_commit_boundary_smoke` with `P0=0`, `ERROR=0` | done as smoke |
| Focused regression checks | see test log below | done |
| Full frozen frontend reproduction | no clean full `--verify hybridz-benchmark --category frozen --hybridz-require-frozen-match` rerun after the soundfix in this pass | not yet proven |
| Linearizenn packaged frontend rerun | profile900 full rerun under `/data1/Kane/ICSE/act_hybridz_soundfix_20260625/productized_reruns/linearizenn_60_profile900_20260625` produced `34 CERT + 1 ADV = 35/60`, `P0=0`; frozen is `39 CERT + 1 ADV = 40/60`; remaining frozen CERT gap is `iid13/41/46/51/52`, all current `UNKNOWN` | partial |
| Git-tracked productization state | product/doc boundary committed in `4ed5712e4` (`Productize HybridZ stage II path`) and post-status commit `78ab54b66`; after later worker-module productization edits, four package modules are currently untracked and must be included in the next productization commit if kept in the runner path; `scripts/` remains untracked and absent from commits | partial |
| Relative-to-upstream minimal diff audit | tracked diff has been measured after the product boundary commit; final audit still needs a full frozen frontend rerun | partial |

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

## Current Post-Commit Product Files Pending Staging

The current runner imports these package modules, so they are product-path files
and must not remain accidental local-only state at final handoff:

- `act/pipeline/hybridz_full_worker.py`
- `act/pipeline/hybridz_sparse_worker.py`
- `act/pipeline/hybridz_sparse_exact_probe.py`
- `act/pipeline/hybridz_sparse_census.py`

They replace legacy `scripts/` workers in the frontend command path.  They are
not the local `scripts/` directory and should be staged in a later
productization commit after the focused gates pass.

## Scripts Migration Conclusion

The current migration matrix classifies script families as follows:

| Family | Status |
|---|---|
| full-run scheduling/export/result ledger | migrated into `hybridz_config.py`, `hybridz_benchmark_runner.py`, and `hybridz_results.py`; legacy scripts kept as provenance until full frontend frozen rerun |
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

## Remaining DoD Gaps

1. Stage or retire the four current product-path worker modules under
   `act/pipeline/`; keep the local `scripts/` directory excluded.
2. Run one clean full frontend frozen reproduction through:

   ```bash
   python -m act.pipeline --verify hybridz-benchmark --category frozen \
     --hybridz-require-frozen-match --hybridz-results-dir <out>
   ```

   This is the evidence required for the "one command reproduces frozen table"
   acceptance criterion.
3. Resolve or explicitly document the current `linearizenn_2024` productized
   reproduction gap: packaged profile900 is `35/60`, while the frozen accepted
   line is `40/60`.
4. After full rerun, re-run the diff-boundary audit against `upstream/main`,
   including newly tracked product files, and remove or document any unrelated
   ACT-wide changes.
5. Continue the function-level audit of `scripts/cifar_sparse_exact_probe.py`
   against `sparse_ops.py` and `solver_hz_verdict.py` before deleting legacy
   sparse scripts.
6. Keep future-work directions separate from the counted artifact: tighter
   sound S-curve, sparse Schur/block presolve, OBBT-lite, lazy exact-HZ, and
   open-source solver portfolio improvements.
