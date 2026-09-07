# Common-task cross-model performance bundle

## Question

This experiment asks whether the complete ACT/HybridZ staged method retains a
registered combination of properties across three independently trained
`bal010` checkpoints.  It is not an attempt to turn a failed historical
conjunction into a pass.  It uses a new common fixed task, the production
verifier entry point, a separately measured structural census, and thresholds
frozen before any endpoint is queried.

All three models receive the same 100 ordered CIFAR-10 images and the same
`2/255` box.  Selection uses only joint clean correctness, ordered index, and
exclusion of earlier HZ cohorts.  Candidate sets, route stability, width,
support elimination, solver status, runtime, and certificate outcomes are
forbidden selection predicates.

## Two separately costed executions

The verdict execution calls the production staged verifier.  It performs exact
candidate and route-set analysis, guarded Tier 1 verification, and invokes the
frozen F0 weighted-range fallback only for semantic incompleteness.  It does
not search for a route boundary and does not execute the matched no-support or
unguarded-accounting controls.

The structural census independently measures IBP, ordinary zonotope, and exact
router candidates; exact unordered top-2 sets; structural monolithic and
route-conditioned widths; and guarded-support binary elimination.  Census
runtime is neither charged to nor subtracted from verdict runtime.  This split
prevents paper-only controls from consuming the deployed verifier's budget.

## Frozen conjunction

For each checkpoint, success requires all of the following:

- zero audit issues, every UNSAFE full-model replayed, exact candidates always
  a subset of ordinary abstractions, and every guard accounting identity closed;
- at least one route-changing SAFE request and at least 50% complete semantic
  outcomes over the full 100-request denominator;
- exact-HZ strictly reduces the ordinary-zonotope candidate set on at least 20%
  of route-unstable census rows;
- route-unstable route-conditioned width ratio has median at most `0.7` and
  90th percentile strictly below `1`;
- F0 resolves at least 25% of the Tier-1 semantic-incompleteness requests on
  which it is invoked; and
- guarded support eliminates at least one expert binary.

The cross-model claim passes only if every checkpoint passes the entire
conjunction.  Individual mechanisms and failures remain visible if the joint
gate fails.  No threshold is weakened after observing results.

The scope is stability across three registered training runs of one model
family.  It is not certified accuracy and not evidence of stability across
architectures.  The separate strict AdvMoE experiment addresses the
high-accuracy, real-scale certificate question.

## Frozen result

All three census and verdict executions completed.  The independent bundle
auditor reconstructed the shared selection, checked checkpoint and artifact
identities, recomputed every exact-candidate subset and guard-accounting
identity, audited 297 emitted evidence packages, and replayed all 67 UNSAFE
witnesses.  It reports zero issues.  The audit status `PASS` means that the
result is internally consistent; it does **not** mean that the preregistered
cross-model scientific conjunction passed.

| Model | SAFE / UNSAFE / UNKNOWN / TIMEOUT | Complete | Route-changing SAFE | F0 resolution | Exact < zonotope | Width median / p90 | Bundle |
|---|---:|---:|---:|---:|---:|---:|---:|
| seed 0 | 30 / 24 / 39 / 7 | 54/100 | 8 | 27/62 | 37/43 | 0.432 / 0.538 | PASS |
| seed 1 | 26 / 21 / 38 / 15 | 47/100 | 8 | 19/50 | 16/40 | 0.379 / 0.467 | FAIL |
| seed 2 | 44 / 22 / 25 / 9 | 66/100 | 7 | 31/53 | 30/38 | 0.348 / 0.453 | PASS |

Seed 1 fails only the frozen minimum-complete-outcome rate (`47% < 50%`).
No threshold, denominator, solver budget, or model is changed after observing
that result.  Consequently two of three models pass, and the registered rule
returns `stable_complete_bundle_supported=false`.  The correct interpretation
is that every individual mechanism in the bundle is observed on all three
registered runs, including 8, 8, and 7 route-changing SAFE model--request
pairs, but their complete performance conjunction is not stable across all
three runs under the frozen budget.

The immutable audited result is
`results/staged_verifier_multimodel_bundle_20260906_r1.json`.  These 100 inputs
were selected to be clean-correct for all three low-accuracy verification-scale
models; the table is neither certified accuracy nor evidence about a
high-accuracy or cross-architecture model.  Runtime is descriptive because the
census is separately costed and this experiment was not designed as a paired
speed comparison.
