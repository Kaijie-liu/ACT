# Frozen 30-input scheduling confirmation R1

## R2 execution repair (supersedes R1 launch, not solver policy)

R1 stopped after its first **old** smoke input (index 3000): strict identity
checking found different represented input hashes. Selection used default-float32
ToTensor then cast to double; the existing CLI sets the global default to float64
before ToTensor. The maximum pixel difference on that old input is ~2.97e-8.
No new confirmation endpoint ran. R1 raw directory, failed terminal, original
selection and first clean-only audit remain immutable; that audit did not check
deployment preprocessing parity and is superseded, not evidence of correct parity.

The selector now calls the same device/dtype initializer before data loading.
New `schedule_confirmation_selection_r2.json` and its separate audit preserve
all thirty chosen indices and the SAME exclusion union. Model state, all smoke
input tensor hashes and the complete request identity now match the retained
actual CLI package. No solver config, halfspace, numeric gate or budget changed.
See `results/schedule_confirmation_smoke_dtype_repair_20260912_r1.json`.

Current executable config is `configs/schedule_confirmation_r2.json`; the
runner rejects the retired preprocessing selection. Repeat ALL nine smoke
requests at `data/moe/results/schedule_confirmation_smoke_20260912_r2`, then
use the new `schedule_confirmation_full_20260912_r2` root. Planned tmux
`moe-schedule-confirm-r2`, log `schedule_confirmation_pipeline_20260912_r2.log`.
No old positive package is borrowed to pass the gate. Following R1 registration
details preserve the original plan; R2 changes only preparation identity and
execution roots. The primary/secondary arms and analysis are unchanged.
R2 validation: 63 focused tests pass, including explicit ToTensor dtype-order
regression, retired-config rejection, unchanged thirty indices and complete
identity equality with the retained old-input worker package.

Authorized 2026-09-12: 30 new inputs, three unchanged bal010 checkpoints,
three arms, 270 requests. Smoke uses one **old** development input on three
models/arms (9 requests), not any new confirmation input. No adaptive stopping,
replacement, additional seeds, changed radius or numerical-policy relaxation.

## Selection and identity

`configs/schedule_confirmation_selection_r1.json` freezes the first 30 ordered
jointly clean-correct CIFAR-10 test inputs at/after index 4000, excluding the
union of registered prior endpoint indices. Selected indices range 4006--4086.
Selection uses CPU/float64 eval, batch one, matching the verifier clean forward.
Raw CIFAR test-batch hash, model state/checkpoint hashes, per-input represented
center/lower/upper hashes and all scanned clean predictions are retained.
No new router-feasibility, complexity, support or verification query ran during
selection. This is endpoint-independent of schedule design, not "never seen"
images: earlier test telemetry was available, and the models are already fixed.

The raw exclusion inventory binds 1,643 files, union 442 dataset indices:
sample-index lists, selection manifests, staged/paired terminal ledgers (including
outer kills without packages), and saved evidence. The parser extracts explicit
dataset indices, not expert/factor `indices`. This coverage is bounded to the
recorded project artifacts, not a claim about unrecorded human inspections.
The independent process rebuilt the selection from clean forwards and source
hashes with zero issues: `results/schedule_confirmation_selection_review_20260912_r1.json`.
The large provenance inventory remains local, hash-referenced by the manifest.

## Frozen methods

1. Adaptive with scoped interval reuse, unchanged single-pair direct path and
   multi-pair 25%-of-remaining Tier-1 allocation.
2. **Primary comparator**: matched monolithic, same independently computed,
   charged common interval facts and new remaining-budget rule.
3. **Secondary strong reference**: historical best monolithic configuration,
   no common-prelude computation/reuse; property limit 300 seconds under the
   outer cap. Exact old settings are pinned, not replaced with new matched settings.

All method JSON hashes are hard-pinned in `schedule_confirmation.py` and
`configs/schedule_confirmation_r1.json`. Epsilon is 2/255, all tie-legal pairs
are obligations, selected-softmax weights stay input-dependent. Optimal-status,
dual-bound and positive-margin numerical policy remain unchanged. SAFE retains
the HZ/HiGHS acceptance scope, not a fully independently checked network proof.

## Runtime, snapshots and smoke gate

Each worker has a 300-second hard subprocess cap including import, loading,
analysis, propagation, optional snapshot publication and solving. All requests
remain in the denominator; an over-cap process or late package is TIMEOUT.
Old times, route censuses and the other arm's facts are never reused. Post-worker
structural audit/replay is outside measured time for every method. One worker
runs at a time with one-thread BLAS/OpenMP. Record versions/load; this is a shared
server, not isolated timing. Model/arm order rotates by input, giving ten first,
ten middle and ten last arm positions per model in the full cohort.

The two scheduled arms publish their common-fact snapshot before arm-specific
solves. Terminal records bind its file hash even when the worker is killed.
The auditor reconstructs expected request identity from frozen model state,
checkpoint and input tensor hashes, checks source intervals and final-package
agreement when available, then compares paired facts. An absent snapshot is
unavailable, not equal. Legacy snapshot fields are not applicable, never a missing
feature charged to that arm. Snapshot presence cannot promote a TIMEOUT.

The separate old-input smoke needs all nine terminal rows, successful structural
audits, full-model UNSAFE replay, at least one complete package from each arm and
at least one matching common-fact pair. Full entry re-audits it and matches source
and config hashes. No all-timeout smoke, fresh-directory `--resume`, overwritten
roots or silent recovery. Errors stop the chain with retained row/log evidence.
The same lock is shared with the prior scheduling runner; a dirty/source-changed
checkout stops before another request. Do not edit code while this chain runs.

## Analysis frozen before outcomes

Primary output: **per-model gained/lost SAFE** for adaptive versus matched on all
30 inputs. Solved (SAFE plus replayed UNSAFE) is reported separately, together
with all four states and gained/lost identities. Adaptive versus legacy is a
fixed secondary comparison; never replace the primary comparator after results.
Report means and medians of paired observed time on all capped requests, not
solved-only costs or an estimate of uncensored time-to-solve.

For each comparison/outcome, also report the mean of the three model differences
within each input and a percentile bootstrap interval (10,000 draws, RNG seed
20260912, 95%). Resample input blocks, retaining all models together. These are
descriptive, unadjusted uncertainty intervals, not family-wise hypothesis tests.
A degenerate interval does not prove population equivalence. Per-model net gains
must remain visible even if the input-clustered average is positive; neither a
positive average nor this same-family sample establishes universal superiority.
No post-hoc effect-size or p-value gate is introduced. Negative or mixed results
are valid endpoints. Single/multiple-route strata are explanatory and use any
completed exact analysis; unavailable counts are preserved and not imputed.

## Execution and limits

Prelaunch validation: 62 focused tests pass, including equal arm-position
counts, forbidden config/comparator changes, selection exclusions, missing or
source-mismatched smoke, incomplete/reordered job streams, request-bound timeout
snapshots, literal weighted legacy conformance, and clustered-summary conflicts.

After commit/push and resource check:
`python -m act.pipeline.moe.schedule_confirmation --pipeline`
in act-py312. It runs smoke, gates full, executes all 270 and saves final audit.
Independent review: `python -m act.pipeline.moe.schedule_confirmation --audit ROOT`.
No automatic commit/push occurs in the worker chain; final outcome archival is
a separate reviewed stage. Maximum worker time is 45 minutes smoke plus 22.5 hours
full, with additional audit/launch overhead. Actual time is not yet measured.

Roots: `data/moe/results/schedule_confirmation_smoke_20260912_r1` and
`data/moe/results/schedule_confirmation_full_20260912_r1`. Planned tmux:
`moe-schedule-confirm-r1`; log `data/moe/results/schedule_confirmation_pipeline_20260912_r1.log`.
Inspect live state before saying started or complete. External-tool integration,
candidate-superset fallback, relation-only ablation, request-level LP certificates
and high-accuracy architecture changes are excluded from this run.
