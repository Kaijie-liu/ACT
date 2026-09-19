# Supervised LP diagnostic: controls and freeze, 2026-09-19

Status: **FROZEN_NOT_EXECUTED**. Zero real-request solver calls. The execution
directory does not exist. This delivery closes the outer-supervision and cost
accounting prerequisite; it does not resolve any of the 30 nonpositive rows.

## Completed implementation

New `lp_diagnostic/` namespace wraps the unchanged `lp_sandwich` component.
Loading, one-shot native proposal, retained internal exact check, packaging,
isolated full check and terminal admission share one original 300-second clock.
Native cap60; load/proposal cutoff218; work watchdog298; late publication300
invalidates acceptance. Owned process-tree cutoff does not touch other jobs.
Native values are preserved before checking. No primal repair or extra solves.

Four immutable terminals remain in the batch denominator even if ERROR stops
the run. TIMEOUT and checked-unresolved cases continue. Independent summary
reconstruction rejects dropped rows, promoted timeout outcomes and changed
cost aggregates. Full phase windows and residual overhead reconcile with the
publication clock; nested native/component seconds are never double counted.
Missing/censored durations remain null. Resource waiting, preflight and
post-terminal/final audits are separately labeled, not free hidden work.

This full clock is for a **supplied LP diagnostic**, not full network verification.
Historic network propagation and F0 construction are not rerun, and cannot be
credited as zero-cost end-to-end verification. No speed claim is made here.

## Controls and preserved receipts

| Receipt | Result | Coverage |
| --- | --- | --- |
| `lp_diagnostic_controls_attempt001.json` | 38/38 PASS | Initial supervised controls and frozen component regressions |
| `lp_diagnostic_controls_attempt002.json` | 41/41 PASS | Added batch reconstruction, missing-check and resigned-import-flag controls |
| `lp_diagnostic_controls_attempt003.json` | 41/41 PASS | Final code; also rejects changed saved aggregate costs |

The latest suite includes 15 new supervisor/batch tests plus 26 previous
component/analysis/archive tests. Actual analytic subprocess chains check a
negative optimum and a non-exact float 1/3 equality point: the latter is not
accepted as a feasible LP upper-bound witness. Other controls cover relocation
with `python -I -S`, original-clock inheritance, proposal reserve, owned cutoff,
late publication, source/checker/native-record tampering, missing evidence,
no overwrite, error-stop and complete batch accounting. Tests use analytic LPs;
their running times are not real-model performance evidence. All earlier
frozen source identities pass unchanged. No dependency installation occurred.

## Frozen diagnostic roster

| Observed input | Legal pair | Property index | Frozen job |
| --- | --- | --- | --- |
| 220 | {1,2} | 0 | `input220_p0` |
| 222 | {0,1} | 1 | `input222_p1` |
| 230 | {0,3} | 2 | `input230_p2` |
| 232 | {0,1} | 0 | `input232_p0` |

Selection is the first nonpositive obligation in original pair/property order
per observed input, not a new holdout. Each export is byte-bound to the original
sealed archive and request identity. Read-only reconstruction in a fresh
process passed with zero issues and zero solver calls. It is a source/selection
review, not an independent network proof or a real LP feasibility result.

Protocol: [lp_diagnostic_v1.md](lp_diagnostic_v1.md).
Frozen sources, policy and jobs: [freeze](lp_diagnostic_v1_freeze.json).
Fresh-process selection review: [review](lp_diagnostic_v1_selection_review.json).
Latest controls: [attempt003](lp_diagnostic_controls_attempt003.json).

## Next bounded action

Only after this code/freeze is committed and pushed, execute the four frozen
one-shot diagnostics if requested. Report exact checked L/U or the absence of
either witness, full diagnostic cost, termination state and retained native
metadata. Do not re-solve the other 26 nonpositive obligations, tune bounds,
retry approximate points, change caches/order, or alter production gates.

A checked feasible U<=0 establishes an obstruction for this LP, not model
UNSAFE. Positive L is LP-only evidence, not complete MoE SAFE. An inexact or
missing primal witness leaves candidate-vs-relaxation unresolved. Network→HZ,
guard, route exclusion and F0 lowering remain upstream trusted assumptions.
