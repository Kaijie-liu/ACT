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
