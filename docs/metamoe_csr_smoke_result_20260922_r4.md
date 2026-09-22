# MetaMoE complete old-input R4 smoke: execution gate PASS, ACT output unresolved

Executed ONCE after explicit user instruction, at clean branch
`feat/moe-route-verification`, HEAD `efa965880a6459611c7b5ee053b593814df38930`.
Implementation `c4813db94e84a7205c3cfa28ce3c2979efe1d935`;
config SHA256 `e8de03d21172dca52a70db958b5725c2b1e6919242adf39e9752dd080a656143`.
See the [frozen protocol](metamoe_csr_paired_protocol_20260922_r4.md) and
[compact saved-result archive](metamoe_csr_smoke_archive_20260922_r4.json).
Raw records remain under
`/data1/Kane/MOE/baseline_runs/metamoe_csr_20260922_r4_smoke`.

## Results, with all four requests retained

| Old input / arm | Outer terminal | Method terminal | Whole-request charged seconds | Evidence grade |
|---|---|---|---:|---|
| CIFAR10 index0 / ACT | COMPLETED, exit0 | UNSAFE_REPLAYED | 1.925880 | Original full-model replay |
| CIFAR10 index0 / author | COMPLETED, exit0 | UNSAFE_REPLAYED | 2.194112 | Original full-model replay |
| MNIST index0 / author | COMPLETED, exit0 | BACKEND_POSITIVE | 7.115386 | Numerical sufficient filter |
| MNIST index0 / ACT | COMPLETED, exit0 | UNKNOWN | 209.487829 | No accepted output certificate |

All four outer processes completed inside their original300s/8GiB limits.
No ERROR, outer TIMEOUT, missing request, retry or resource refusal. The two
CIFAR rows are the SAME already-known center misclassification (label3,
prediction5), not two independent inputs or newly discovered attacks. Both
original-model replays return minimum margin `-0.1627192347142865`; each saved
witness equals the old physical center and is inside its frozen box.
Consequently, those two calls exit at the symmetric center check, rather
than exercising either solver. The MNIST calls exercise the actual paths.

Author MNIST raw token is `safe-incomplete`, not timeout; the frozen parser
accepts it as `AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER`. The backend reports
2.6142s internally, but the comparison charges its COMPLETE7.115386s request.
This is the pinned route-invariance sufficient adaptation of the author's
native backend, NOT a direct unchanged author-paper table or source-complete
dynamic-model proof. It must not be relabeled formal SAFE.

## What remains unresolved on ACT MNIST?

The saved result records:

- candidate experts `[1]`, excluded `[0]`, unresolved `[]`, `candidate_minimal=true`;
- expert1 output verification status `unknown`, over19 global class properties;
- selected-score range `[3.345338179462453, 4.30685851511294]`, accepted nonzero;
- both support statuses `fast_fallback`, NOT a claim of optimal MILP support;
- final reason `nonzero_or_global_output_obligation_unproved`.

The detailed fields therefore locate the unresolved endpoint at expert output
verification, NOT the selected-score nonzero check or incomplete candidate
coverage. They do not separate solver-budget effects from relaxation strength:
this result contains no complete per-property primal/dual or native-call
trajectory. Do not infer the model is unsafe, that the LP itself is unable to
prove the property, or that adding time would solve it. No extra solve was
performed for this interpretation.

All138 recorded sparse admissions were accepted, max accounted607,780,524
bytes below the unchanged2GiB representation limit. Eight convolution plans
account for0.154202s INCLUDED in the request, not a separate matched timing
study. ACT sampled process-group peak RSS was2,360,549,376 bytes
(2.198433GiB); author MNIST peak was1,968,959,488 bytes, both below8GiB.
RSS sampling is not a proved instantaneous memory bound. The wider full
verification path's RSS is not the same quantity as CSR representation bytes.

## Audit and complete cost accounting

Preliminary saved terminal/identity/cost audit PASS, then separate original
full-model witness replay PASS for exactly both accepted UNSAFE rows, then
combined audit PASS with `execution_control_pass=true`, `smoke_gate_pass=true`.
The preliminary audit deliberately had `smoke_gate_pass=false` before the
independent replay was supplied. This was an unmet replay gate, not a failed
execution that was rerun or relabeled.

| Cost | Seconds | Accounting scope |
|---|---:|---|
| Sum of four charged requests |220.723207|startup/imports/identity/analysis/solves/candidate/cleanup|
| Batch wall through summary |221.023628|also top-level validation, parent records and hashes|
| Receipt postflight subset |0.000271|already included in batch, do not add twice|
| Preliminary independent audit |0.272328|separate from batch|
| Original-model witness replay |1.208246|separate from batch|
| Combined final audit |0.280050|separate from batch|

Audit/replay clocks are their instrumented function costs; interpreter launch,
interactive monitoring gaps, the final batch-cost file's own write and archive
preparation are not claimed to be included in a single overall stopwatch.
Archive read/hash cost is separately recorded. Do not report the sum of four
request clocks as the entire workflow, or compute a general speedup from this
one old MNIST request with different method conclusions and evidence grades.

Independent read-only reconstruction reproduced both reviews apart from their
own elapsed times;485 frozen source/input bindings and repository/environment
identities passed. The compact archive binds43 raw metadata/log/artifact files
by SHA256 and size, preserves complete receipts, and retains the ACT obligation
snapshot without committing raw tensors/checkpoint/witness arrays. A separate
read-only check found the two stored witnesses equal to the same old center;
it did not perform a second full-model inference or new solver query.

## Meaning of PASS and next boundary

This closes the full OLD-input execution/guarded-intake gate: the former ACT
representation error is no longer the stopping point. It does NOT show ACT
proved the MNIST output property, beat the author path, reproduced an author
table, or produced any new source-complete certificate. All source-to-HZ,
guard-lowering and numerical-policy trusted components remain disclosed.
Numerical grades are not equated; structural audit is not independent positive
bound reproof. Ordinary registered UNKNOWN was allowed by the frozen intake
protocol and remains UNKNOWN in the archive.

`opens_formal_cohort=false`; no new20-input cohort was selected or executed,
no thresholds/budgets/source files were changed, and all previous failed
attempts remain. A later formal comparison needs its OWN user-authorized
freeze/selection and identical passed implementation/environment identity.
The next stage is a separate decision, not an automatic continuation of this
four-call smoke. Do not extend the old MNIST budget or rerun it to chase SAFE.
