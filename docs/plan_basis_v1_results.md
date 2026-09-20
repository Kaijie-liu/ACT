# Cross-prime plan reuse: control results, not a real-LP speedup

Date: 2026-09-20. Starting branch `feat/moe-route-verification`, clean HEAD
`8e4f30e47ca8f9af3f89cc7f0f17c4f64283664b`.

## Delivered scope

Separate `plan_basis/` implements candidate-only symbolic schedule reuse.
Frozen modular/primitive implementations and the four real diagnostic records
were not changed or rerun. No native solver call, network training, new sample,
cap extension, production hook or unified-supervisor integration was introduced.

The first usable prime records initial row structure and pivot row/column,
pivot support and affected rows. Later primes recompute every modular numerical
value. A schedule is immutable and bound to exact source coefficients, RHS and
scope (the LP adapter includes statement, original LP and basis-hint identities).
Plan integrity and schema errors fail closed. Initial cancellation, intermediate
zero planned pivots, changed support or wrong modular residuals invalidate the
plan; a dynamic factorization of the same prime is then charged to the same
budget. This is a new registered research fallback, not a retrofit to old runs.

Both on/off paths check original modular equations before CRT. Full rational
vectors still require exact original-equation residuals, followed by the
unchanged standalone original-LP checker. Even a solved basis can have a
nonzero original equality residual or illegal slack and must be rejected there.
No partial modular coordinate is a certified feasible point.

## Controls and independent review

- Current [control receipt](plan_basis_controls_attempt004.json): **86/86 PASS**,
  comprising 22 new controls and 64 frozen arithmetic/checker regression tests.
- [Fresh review](plan_basis_review_attempt002.json): **PASS, zero issues**,
  228 artifacts, 60 exact saved-system residual checks, 11 unresolved systems
  retained, 49 saved new-arm differential comparisons and three reviewer mutation
  rejections. These counts include explicitly identified frozen regression fixtures.
- Nine freshly relocated `python -I -S` original-LP checks: seven exactly
  feasible and two expected rejections. The new multi-round fixture uses three
  primes and two plan replays; its checked upper bound is
  `-1099511627776/7`. This is an analytic LP diagnostic, not a MoE certificate.
- No new elimination or native call in the fresh review. Source and artifact
  identities, including all previously frozen real records, were checked again.

Controls cover initial modular cancellation, a zero planned pivot despite an
invertible matrix at that prime, matrix/RHS/request-scope contamination, mutable
or malformed plan data, rehashed malicious schedules, corrupted replay values,
on/off/frozen differentials, deadlines inside replay and reconstruction, and
operation/bit/plan/live-storage limits. Fallback retains the old plan's storage
while constructing the replacement; counters and deadlines are never reset.

Receipts 001 (82 tests), 002 (83), 003 (85) and their available earlier review
are retained as historical development revisions. They are not substituted for
receipt004's binding to the current source. Incremental additions corrected
fallback retained-storage accounting and strengthened deadline, schema and
actual multi-round original-LP coverage; no acceptance threshold was changed.

## Diagnostic additions

`costs.operations` and `costs.seconds` distinguish:

- `finite_field/map`, `finite_field/elimination`,
  `finite_field/back_substitution`, `finite_field/residual_check`;
- plan binding, validation and publication;
- prime generation, CRT, rational reconstruction and exact residual checking.

Observed zero-operation phases have explicit zero counts. Unreached phases are
absent rather than fabricated as observed zero duration. Count sums equal the
shared counter, including the operation that trips the cap. Plan capture/index
work inside mapping or elimination is included in those buckets; this is not a
pure numerical-kernel profile. Timings are local instrumented arithmetic, not
complete production-request cost.

Each attempted reconstruction records successful prefix length, first failed
coordinate, bound size and reason, or an interrupted status. A complete vector
awaiting exact residual checking is distinct from a residual-checked vector.
The prefix control fails coordinate1 after one successful coordinate, records
that failure, and only a subsequent fully checked vector returns. The deadline
control retains the interrupted coordinate without publishing a solution.
These new diagnostics do not backfill missing fields in the four historical LPs.

## Cost result: correctness passed; net operation reduction did not

The registered 1024-dimensional sparse synthetic control has the same exact
solution and three prime rounds in both arms. Reuse builds one plan and replays
it twice. This is a descriptive control, not a paired real timing experiment.

| Control arm | Counted operations | Observed arithmetic seconds |
| --- | ---: | ---: |
| No reuse | 212,648 | 0.052968 |
| Reuse | 249,499 | 0.062709 |

The operation increase is **36,851 (+17.33%)**, decomposed exactly as:

| Added charged work | Operations |
| --- | ---: |
| Repeated source/scope binding | 15,354 |
| Plan validation | 14,332 |
| Mapping / initial-plan structure work | 5,118 |
| Elimination including plan capture/checks, net | 2,047 |
| Total | 36,851 |

CRT, prime generation, reconstruction, back-substitution and modular/exact
residual operation counts are unchanged in this comparison. Avoiding repeated
heap-based pivot selection does not by itself pay for binding, validation and
plan construction on this control. We therefore do **not** claim acceleration,
real-LP benefit or likely closure of the four original LIMIT results.

## Next boundary

This controls-only stage is complete. Keep the interface isolated and optional.
Before a real diagnostic, the most directly motivated follow-up is separately
studying immutable per-request source/plan validation amortization, preserving
all binding and residual checks. That is not permission to remove checks or
raise the operation cap. Any later unified-budget supervisor must separately
validate deadlines, partial evidence and full request costs before a new real
freeze; the existing frozen supervisor does not accept this new stats schema.
