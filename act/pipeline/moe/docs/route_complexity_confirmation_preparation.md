# Independent scheduling confirmation: preparation, not a launched experiment

2026-09-12. The observed ten-input schedule run is sealed. No fraction tuning,
candidate-superset fallback, external backend, training or LP acceptance change
is included here. Preparation is not a frozen selection or a result.

## Three solver configurations

- Primary: `configs/route_complexity_reuse_v1.json`, unchanged 25% fraction.
- Primary comparison: `configs/monolithic_matched_reuse_v1.json`, unchanged.
- Strong legacy reference: `configs/monolithic_legacy_reference_v1.json`, exactly
  the JSON settings of the audited four-arm `monolithic_f0.json`. Original SHA256
  `e00c4dcd2ac2b3c0497b1b16a04f8b8bd7204026d842880754ffdc426849cc02`;
  tracked copy SHA256 `bb05702ebcd61ae47db28f2d35fe03d2e84803b23fa3e2cc4328a954e00ff756`.
  The only byte difference is the tracked copy's final newline.
  It has no scoped reuse/common prelude and allows a property up to 300 s,
  subject to the outer cap. Do not charge it for unused new common facts.

Rerun these configurations; do not reuse historical times. Shared implementation
changes still need regression checks before claiming legacy execution equivalence.

## Durable common facts implemented now

The API accepts optional `common_fact_callback`; the CLI accepts
`--common-fact-snapshot PATH`. Publication occurs after common fact extraction,
before any arm-specific expert solve. It binds request/model/checkpoint,
represented center/bounds, config, property, tie policy, local frame, exact
legal-pair inventory, source intervals, fact count and elapsed completion time.
Copying/checking/writing consume the same budget; no extra solver query is made.

Flushed/fsynced temporary contents are atomically linked without overwrite;
directory fsync follows. A killed writer cannot publish half JSON under the
final name. A pending temporary file can survive interruption but is not an
accepted snapshot. Existing final paths fail closed. Detached callback data
cannot mutate live facts. The real SIGKILL control retains a readable snapshot
without producing a final package or successful verdict.

`common_fact_snapshot.check_snapshot` checks hashes, request/config identity,
canonical inventories, interval arithmetic/counts and optional agreement with
the final package. The next runner must supply expected task/config identities:
a self-consistent hash alone is not independent model binding. `fact_view`
ignores only local IDs; pair comparisons must also match literal request/model
identity. Route feasibility and network-to-HZ propagation remain trusted.

Incomplete preparation emits nothing. Missing means unavailable, never equal.
TIMEOUT remains TIMEOUT even if its prelude completed. No historical unavailable
facts are fabricated. Existing runners do not enable the new flag; the next
three-arm runner must explicitly collect/validate snapshots from both scheduled
arms, including killed requests. The legacy arm is not applicable.

## Selection and launch pending

Keep the three checkpoints and 2/255. Select new ordered jointly clean-correct
inputs excluding all previously analyzed HZ endpoint cohorts, including the old
common 100 and development subsets. Prior clean telemetry means these are not
"never seen" inputs; the claim is that their verification endpoints did not
inform the schedule. No route-complexity, bound, status or timing prefilter.

Before solving: settle sample count, audit the exclusion inventory, freeze
selection/primary contrast/statistics, implement/test a separate three-arm
runner/auditor, commit/push, then an observed-input smoke with the same code.
Exclude smoke inputs from confirmation. Rotate arms within input/model blocks,
retain all states/costs, separate SAFE and solved, and cluster models by input.
Route-complexity strata are explanatory; no post-outcome comparator replacement.

Sample count awaits PI choice: 30 inputs = 270 requests / at most 22.5 worker
hours; 100 = 900 / at most 75 hours, plus smoke/audit. No new endpoint is run
by this stage. External-tool compatibility and request-level LP proof packages
remain separate, unimplemented/unlaunched follow-ons, not this ablation.

## Validation

54 focused tests pass, including callback isolation, final-package comparison,
scope/hash/inventory tampering, no-clobber publication, incomplete-prelude
absence, real SIGKILL retention, and legacy configuration identity. A separate
process reproduces the complete derived phase review from raw package hashes.
An initial byte-identity assertion caught the copied config's added final
newline; both hashes and this formatting-only difference are disclosed above.
No configuration value changed to make the test pass.
