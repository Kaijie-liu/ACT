# R4 complete old-input two-arm smoke: freeze, not execution

Authorized scope: freeze the complete old-input smoke AFTER the archived R4
representation PASS. This stage does not run it, select new inputs, or grant
automatic permission for a formal cohort. R1/R2/R3 failures remain immutable.

## Exact same-object task

Two existing physical tensors from paired R2, copied by manifest reference,
NOT rematerialized from datasets: CIFAR10 index0 (label3, clean prediction5)
and MNIST index0 (global label17, clean prediction17). Fixed call order:

1. CIFAR10_0 ACT;
2. CIFAR10_0 author backend;
3. MNIST_0 author backend;
4. MNIST_0 ACT.

This is FOUR full-request invocations, not necessarily four solver executions:
the known CIFAR clean error is expected to return through the symmetric
original-model center replay. Preserve it; do not substitute a clean-correct
input. MNIST exercises the verification paths. No outcome-selected replacement.

Same full checkpoint, all20 global classes/19 margins, exact materialized
CPU/float64 normalized box at epsilon2/255 clipped to[-10,10], margin1e-7,
two CPU threads and seed100. This is NOT a pixel-space radius claim. All
tie-legal top1 choices and zero-filled unselected classes remain obligations;
the selected raw-score division must be defined. Original-full-model replay
is the only UNSAFE acceptance path, not a surrogate/expert violation.

ACT uses unchanged class-separated guarded entry, functional non-inplace ReLU
spelling adapter (no BN folding), and R4 `csr_spatial_v2` with2GiB admission.
The original numerical Conv/ReLU/support/solver/acceptance logic is unchanged.
Author arm remains the pinned native PyTorch route-invariance sufficient
adapter, including nonzero selected-score and other-domain zero-block output
obligations; NOT an unchanged author-paper table or direct dynamic export.
Author positives are NUMERICAL_SUFFICIENT_FILTER, ACT positives conditional
HZ_POLICY_ACCEPTED; neither is source-complete floating-program certification.

Driver/tests/freezer use existing act-py312. Both model/backend worker
environments are inherited byte-for-byte in metadata from the successful R4
intake/R2 comparison, with no install/upgrade. This preserves their actual
dependency/kernel identities rather than changing them during the comparison.

## Execution and resource identity

New config `configs/recent_moe/metamoe_csr_paired_smoke_r4.json`, new root
`/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke`.
No resume, retry, overwrite, new-selection or formal-cohort CLI. Freezer
requires reviewed clean code and reconstructs the saved R4 diagnostic review
before writing; binds diagnostic config/archive, old requests, all inherited
sources plus new runner/auditor/tests/protocol. No inherited source rebinding.
Implementation committed first, manifest committed/pushed second, launch ONLY
after a separate execution instruction. No writers during any eventual run.

300 seconds per COMPLETE arm including subprocess environment setup/imports,
source/input validation, lowering, own router/guard analysis, all solves,
candidate serialization and own-process-group cleanup. R4's90s diagnostic is
a different experiment, not the smoke's cap. Same8GiB sampled process-group
RSS limit and50ms polling, not an instantaneous OS memory guarantee. Only the
new supervisor's own process group may be killed. Resource refusal is not a
negative proof. The worker receives remaining time; outer deadline dominates
late/partial candidates, including cleanup past deadline.

Parent log hashes/receipt write are disclosed OUTSIDE the per-request clock;
batch wall includes them, intermediate artifact hashes, terminals and summary.
The batch cost file's own write and later independent audit/replay are separate.
The4×300s allocation is NOT a claim that the entire workflow finishes in20min.
New per-call artifact hashes bind prepared/rewrite/obligation files and native
backend config/spec/logs/results where present. No hidden free route census.

ERROR/SOURCE_CHANGED fail-stop with explicit NOT_STARTED rows. Timeout and
resource refusal retained; other old inputs still execute once. Ordinary
complete numerical UNKNOWN may pass intake, but raw backend `timeout` mapped
to UNKNOWN by the historical parser must NOT open the smoke gate. Preserve its
raw status; no historical parser rewrite. Missing representations/resource
refusal/outer timeout/error or unknown unregistered reason closes the gate.

## Independent acceptance and controls

Saved auditor checks ordered denominator, identities, candidate/log hashes,
all completed receipts' finite costs, exit0/error=None, deadline and RSS,
backend raw terminal token, ACT positive obligation completeness, and separate
evidence grades. A preliminary terminal audit alone cannot pass the smoke:
original full-model replay must be run separately, bind the SAME config and
summary, cover exactly all accepted UNSAFE records once, and disclose its cost.
Final combined review always reports `opens_formal_cohort=false`. Passing an
execution gate is NOT an independent mathematical reproof of any positive.

Controls cover request/resource drift, missing source/prerequisite binding,
wrong candidate grade/coverage, late/nonfinite/error/RSS completed receipts,
partial/late candidate precedence, raw backend timeout, artifact tampering,
ordered four-call dispatch, fail-stop denominator, and missing/duplicate/wrong
replay binding. Existing outer supervisor controls are rerun unchanged.
Only synthetic records/processes are used at freeze time; no real inference
or solver query is added to the old inputs by these tests.

Pre-freeze validation:52 tests pass:15 new smoke/freeze,10 existing outer,
7 R4 diagnostic/auditor,11 sparse planner differentials and9 class-separated
semantic controls.
Read-only review identified two additional false-pass cases before freeze:
author unknown/unsafe raw tokens could masquerade as positives, and replay
rows could carry invalid labels or nonfinite margins. Both are now rejected,
with controls; no historical parser, numerical threshold or record changed.
An ACT numerical UNKNOWN may include internally budget-limited queries; an
intake PASS does NOT mean every native solver call reached optimality.

## Commands AFTER separate execution authorization

From the clean ACT checkout, with model jobs/resources inspected first:

```bash
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/metamoe_csr_paired_r4.py --config configs/recent_moe/metamoe_csr_paired_smoke_r4.json
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/audit_metamoe_csr_paired_r4.py --config configs/recent_moe/metamoe_csr_paired_smoke_r4.json --output /data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke/terminal_review.json
/data1/Kane/MOE/envs/moe-author-cpu-py312-20260921/bin/python scripts/replay_metamoe_paired.py --config configs/recent_moe/metamoe_csr_paired_smoke_r4.json --output /data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke/original_replay.json
/data1/Kane/miniconda3/envs/act-py312/bin/python scripts/audit_metamoe_csr_paired_r4.py --config configs/recent_moe/metamoe_csr_paired_smoke_r4.json --replay /data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke/original_replay.json --output /data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke/final_review.json
```

Stop on command failure, retain the incomplete run. Final archival rechecks
identities and original replay before any claim. No automatic Git writes or
new cohort follows these commands. Source/guard/HZ trust gaps stay unchanged.
