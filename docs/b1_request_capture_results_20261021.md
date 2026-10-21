# B1 same-invocation request capture — archived control results

Execution `0cc9a4f9db5449b17a1a6444bbbc7cb5db5589b9`; one frozen batch,
no retry. [Protocol](b1_request_capture_protocol_20261021.md),
[freeze](b1_request_capture_freeze_20261021.json),
[lossless records and saved-only audit](b1_request_capture_records_20261021.json).

**PASS:4 complete synthetic requests,17/17 fixed controls,2 selected actual-source
records.** Both paths preserve their off/on status, reason, decision stage,
route coverage and absence of a replayed full-model counterexample. No training
checkpoint, dataset, GPU, stopped fixed-tilt strategy or external backend ran.
Real-stage consumption remains0 stages/0 calls/0 hours.

| Path | Observer | Complete verdict | Observed request time |
| --- | --- | --- | ---: |
| staged | off | UNKNOWN_WEIGHTED_RELAXATION |0.321816s|
| staged | on | UNKNOWN_WEIGHTED_RELAXATION |0.189040s|
| monolithic | off | UNKNOWN_MONOLITHIC_RELAXATION |0.234715s|
| monolithic | on | UNKNOWN_MONOLITHIC_RELAXATION |0.133863s|

These ordered, tiny synthetic timings are **not a speedup experiment**. Import,
model creation and reception are also charged in the complete batch clock.
The fixture has constant gate0.5 and no expert-private factors; these results
do not establish variable-gate/private-factor behavior on trained models.
Every on-request retains3 tie-legal pairs ×9 properties =27 obligations:
25 reused,1 selected and1 other returned. Staged registers2 F0 slots; monolithic
registers1 slot containing2 residual pairs. Selection is always pair(0,1),
property0, not the native active branch. Both selected records report an actual
native call; this does not turn all registered slots into native-call counts.

## What is now captured

The two selected records contain the actual guarded entry, conditioned router,
joint expert output, F0 source, native matrix/objective/bounds/integrality,
raw point and mapping metadata from the same invocation. All10 class rows
are retained. Published inventory, request evidence and selected native payload
are hash-bound. Router→entry→joint→F0 stored constraint prefixes are checked
by exact column sums; noncanonical duplicate CSR control passes and nonzero
appended-column pollution rejects without rewriting original matrices.

The metadata explicitly records margin support with binaries retained and
difference support with binaries relaxed. This matters to B1: a range valid
only on integer assignments cannot be silently used to describe the unchanged
continuous relaxation. No unique source of relaxation error is diagnosed by
these synthetic data.

## Fault and budget evidence

The17 fixed controls cover inventory completeness/identity, duplicate reuse,
property/deadline mismatches, absent native returns, discarded references,
first-ordinal missing-source lock, altered guard prefix, reception/publication
failure, alias mutation, CSR representation/padding and request-level skip
reasons. Expiry during terminal publication is rejected under the same clock;
an unavailable first source is not replaced with an easier later one. These
state-control fixtures are not additional native solver results.

- Controller elapsed before terminal export:6.583128s; through export:6.585035s.
- Controller structural reception:0.009761s, included above.
- Cumulative diagnostic Work:3047 at worker completion →4493 after reception.
  This counts diagnostic checking, not every native optimizer instruction.
- Owned cleanup:0.040087s; direct leader reaped, no live session members at
  receipt. Final descriptor balance passed. This is not an escaped-descendant
  or all-zombies proof.
- The log includes a CUDA-availability probe warning from a dependency; models
  and solves used CPU and no GPU experiment ran. Do not claim zero CUDA API
  contact merely from the disabled-device configuration.
- Direct-child `wait4.ru_maxrss`:873,193,472 bytes, telemetry only, not group
  simultaneous physical-memory peak. Each process retained8GiB AS hard limit.
- Saved run520,657 bytes + lossless archive527,183 bytes =1,047,840 bytes,
  within the128MiB total. No artifacts were removed.

The once-frozen saved-only audit passed using `python -I -S -B`; it does not
rerun models or solvers, replenish an online deadline, or prove native-point
feasibility. Pre-interpreter startup and final driver flush/exit tail are not
reported as complete cold-start cost.

A separate read-only AI review matched all44 saved file digests/sizes, four
evidence receipts, both final inventories, scope/deadline and obligation counts.
It is not the independent human technical review required by G6.

## Scientific boundary and next task

This closes B1's request-level source/denominator interface gap. It does **not**
establish a real-model gain, LP optimality, guard lowering, network conversion,
complete same-source robustness or a new algorithm. Structural source reception
does not independently replay the producer's exact-prefix check.

Next finish the prospective dataset-aware input/history manifest and fixed
source-matched query recipes, then freeze the first real B1 diagnostic stage.
Its output should distinguish unfinished search, continuous/integer differences,
gate refinement and genuine full-model witnesses, retaining “not separated”
when the evidence is insufficient. Do not repeat the passed interface controls
or extend the stopped fixed-tilt mechanism. B2's failed deployment audit remains
stopped pending its separately scoped decision. G1–G6 remain OPEN.
