# HyZor Stage-II Commit Boundary

Date: 2026-06-25

Post-commit note: the boundary defined here was used for the initial Stage-II
productization commits.  The packaged worker modules listed later in this file
were subsequently committed in `8738f50f6` (`Productize HybridZ packaged
workers`).  The local `scripts/` directory remains excluded.

Purpose: define the productization commit boundary before staging or committing
the current HybridZ work.  This avoids accidentally committing local experiment
scripts while also preventing product-path files from remaining untracked.

## Include In Productization Commit

These paths are part of the ACT product path, frontend runner, solver/verdict
layer, operator layer, result export, or audit documentation.

### Already Tracked Modified Files

- `act/back_end/__init__.py`
- `act/back_end/analyze.py`
- `act/back_end/cli.py`
- `act/back_end/config.py`
- `act/back_end/config.yaml`
- `act/back_end/hybridz_tf/algorithms/__init__.py`
- `act/back_end/hybridz_tf/algorithms/order_reduce.py`
- `act/back_end/hybridz_tf/hybridz_tf.py`
- `act/back_end/hybridz_tf/tf_cnn.py`
- `act/back_end/hybridz_tf/tf_mlp.py`
- `act/back_end/interval_tf/tf_cnn.py`
- `act/back_end/solver/__init__.py`
- `act/back_end/solver/solver_hz.py`
- `act/back_end/solver/solver_hz_verdict.py`
- `act/back_end/transfer_functions.py`
- `act/back_end/verifier.py`
- `act/front_end/model_synthesis.py`
- `act/front_end/vnnlib_loader/create_specs.py`
- `act/front_end/vnnlib_loader/data_model_loader.py`
- `act/pipeline/__init__.py`
- `act/pipeline/cli.py`
- `act/pipeline/verification/__init__.py`
- `act/pipeline/verification/act2torch.py`
- `act/pipeline/verification/torch2act.py`
- `act/pipeline/verification/utils.py`
- `act/pipeline/verification/validate_verifier.py`

### Product Files Currently Untracked

- `FULLRUN_HANDOFF.md`
- `act/back_end/hybridz_config.py`
- `act/back_end/hybridz_tf/__main__.py`
- `act/back_end/hybridz_tf/sparse_ops.py`
- `act/back_end/solver/sparse_hz.py`
- `act/pipeline/hybridz_benchmark_runner.py`
- `act/pipeline/hybridz_projected_relu_mip.py`
- `act/pipeline/hybridz_projected_utils.py`
- `act/pipeline/hybridz_results.py`
- `docs/ABSTRACT_DOMAIN_OPERATOR_AUDIT_TEMPLATE.md`
- `docs/CGAN_FRONTIER_ANALYSIS_20260624.md`
- `docs/DISTSHIFT_SCURVE_TAIL_AUDIT_20260624.md`
- `docs/FUTURE_WORK_HYBRIDZ_20260624.md`
- `docs/HYZOR_STAGE2_CONSOLIDATION_AUDIT_20260624.md`
- `docs/HYZOR_STAGE2_DIFF_BOUNDARY_AUDIT_20260624.md`
- `docs/HYZOR_STAGE2_PRODUCTIZATION_STATUS_20260625.md`
- `docs/HYZOR_STAGE2_SCRIPT_MIGRATION_MATRIX_20260624.md`
- `docs/HYZOR_STAGE2_COMMIT_BOUNDARY_20260625.md`
- `docs/HZ_FRONTIER_ASSESSMENT_20260624.md`
- `docs/HZ_STOP_OR_CONTINUE_DECISION_20260624.md`
- `docs/MILP_ROW_SCALING_AUDIT_20260624.md`
- `docs/SCURVE_PAIR_CORRELATION_AUDIT_20260624.md`
- `docs/SPARSE_MIP_STRUCTURE_AUDIT_20260624.md`

## Exclude From Productization Commit

The following remain local experiment/provenance/future-work scripts.  They are
not required by the normal ACT frontend HybridZ product path and should not be
added in the Stage-II productization commit unless a later pass migrates a
specific script into `act/` with tests.

- `scripts/`

This includes:

- `_excluded_guided_falsification/` and all LP-witness/guided rescue scripts;
- CIFAR lazy/matrix-free and sparse probe scripts;
- local full-run drivers/workers/mergers/exporters;
- projected graph/primal MIP research scripts;
- NNV comparison scripts;
- standalone audit/census scripts.

## Current Verification Before Staging

Latest focused checks:

```bash
python -m act.pipeline.hybridz_results
python -m act.pipeline.hybridz_benchmark_runner
python -m act.back_end.hybridz_tf
python -m act.back_end.hybridz_config
python -m act.back_end.solver.solver_hz_verdict
git diff --check
git diff --cached --check
git diff --cached --name-only | rg '^scripts/' || true
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

All passed in the current worktree.
The frontend smoke emitted detail/summary/ICSE CSVs, JSON summary, and manifest
files with `P0=0` and `ERROR=0`.
The `act/` legacy-script scan only hits a self-test assertion that generated
commands must not contain `hz_full_worker` or `hz_sparse_worker`.
The productization boundary was committed in `4ed5712e4` with 49 product/doc
files and no `scripts/` paths.
The post-commit smoke completed with `N=1, UNKNOWN=1, P0=0, ERROR=0` and a
clean manifest.

## Required After Further Edits

1. Re-run the focused self-tests above after any further edit.
2. Confirm `git status --short` shows no accidental `scripts/` additions.
3. The current runner now imports four package worker/probe modules that were
   created after the boundary commit and committed in `8738f50f6`:
   `act/pipeline/hybridz_full_worker.py`,
   `act/pipeline/hybridz_sparse_worker.py`,
   `act/pipeline/hybridz_sparse_exact_probe.py`, and
   `act/pipeline/hybridz_sparse_census.py`.  They are product-path modules, not
   local `scripts/` files.
4. If time permits, run the full frozen frontend reproduction with
   `--hybridz-require-frozen-match`; otherwise leave the goal active and record
   that the full reproduction remains unproven.
