# Validation amortization: fixed controls and limitations

Date: 2026-09-20. Started on clean `feat/moe-route-verification` at
`21484b21556de00e0d9dc87f9a2382da0458d57a`.

## Result

The isolated `amortized_basis/` implementation passes **106/106 controls**
(20 new +86 regressions). In the unchanged structured1024-dimensional,
three-prime analytic fixture, amortizing both validations saves **24,568
counted operations (9.496%)** relative to this version's repeated-validation
arm. The saving is entirely attributable to repeated source binding and plan
validation; numerical computation and residual-check operation counts agree.

This is a bounded mechanism result, **not** a new real-LP result, a matched
wall-time speedup, or proof that symbolic plan reuse beats no reuse. No real
diagnostics, native solver calls, training, samples or budgets were added.

## What is checked, and when

All four modes perform the same owned-source creation and full admission.
Source rows, RHS and scope are copied into nested tuples of integer rational
pairs; no caller dictionary or mutable Fraction internals remain aliased.
Every new plan receives the full index/schema/permutation/checksum/binding
check before an immutable owned-plan receipt is admitted. External imports
always receive that full check again; a serialized `validated` flag is rejected.

Source and plan validation facts are invariant for these owned immutable
payloads. The optional modes reuse those facts, not unverified caller data.
Every use still checks source/scope and request ownership, with plan generation
and receipt identity checks. Foreign, forged, replaced, stale and closed handles
are rejected. Mutating an exported wrapper or the original source cannot alter
owned payloads. Late admission timeout rolls back the new receipt while retaining
all incurred costs; it does not leave a usable partially published receipt.

Every prime still recomputes coefficients and numerical factors, checks
structural compatibility and zero pivots, and verifies the original modular
equations. All complete reconstructed vectors still require exact original
rational equation residuals. The unchanged standalone original-LP checker still
checks feasibility and the claimed objective. A basis solution is not itself a
feasible LP point. All of these checks remain in both arms.

This is an API-level immutability/ownership argument, not a machine-checked
Python execution proof or protection from arbitrary code/private-memory
replacement. The external exact LP checker remains an independent gate.

## Four-mode attribution

Mode order and fixture were fixed in the [protocol](amortized_basis_v1.md).
Every arm has three prime rounds, one plan construction and two plan replays,
with the same exact solution. Source copy and per-use checks are charged in all
four arms; only repeated full validation is amortized.

| Source validation | Plan validation | Full source / plan checks | Operations | Observed seconds |
| --- | --- | ---: | ---: | ---: |
| Repeated | Repeated | 3 / 3 | 258,718 | 0.098093 |
| Amortized | Repeated | 1 / 3 | 248,482 | 0.082337 |
| Repeated | Amortized | 3 / 1 | 244,386 | 0.085681 |
| Amortized | Amortized | 1 / 1 | 234,150 | 0.079308 |

The operation saving separates exactly:

- Source binding: **10,236**, eliminating two repetitions of5,118.
- Plan validation: **14,332**, eliminating two repetitions of7,166.
- Total: **24,568**; no change to counted mapping/elimination/back-substitution,
  modular residuals, prime generation, CRT, rational reconstruction or exact
  residual work in this comparison.

All modes retain four request guards (three field requests plus one admission)
and two plan-receipt guards. The source snapshot has3,071 logical units. These
are registered counters, not CPU instruction counts or exact Python RSS.
Timings are single instrumented controls on a shared CPU machine, not a formal
timing experiment; no acceleration factor is claimed from them.

For context, frozen V1 no-reuse on the same synthetic equations counted212,648
operations, below this version's234,150. V1 lacks the new owned admission and
common resource repair, so it is a descriptive reference, not a same-version
factorial arm. **There is still no evidence here that the whole plan-reuse path
is preferable to no reuse.** Do not use the9.496% localized saving to claim
otherwise or to justify immediate real retries.

## Resource repair and preserved limits

The new versions retain4096 bits,20M operations,128 primes,1M plan units and2M
live units. Both account for the owned source plus old/new plan indices during
replacement. No deadline or operation counter is reset on invalidation.

A small analytic control exposes a V1 accounting inconsistency: with live cap3,
the old dynamic path returned while recording peak7 because its cap comparison
omitted plan units. Both new arms use the inclusive cap expression; the control
now stops at observed4. This is a resource-policy repair, not evidence of an
incorrect mathematical certificate. Frozen V1 code/results were not rewritten.

The new interface still lacks the production outer supervisor and durable
partial-publication protocol. Its local deadline/cost controls cannot be claimed
as completed end-to-end request integration.

## Evidence and independent review

- [Current controls](amortized_basis_controls_attempt002.json):106/106 PASS,
  including four-mode/frozen differentials, alias/residual rejection, cache
  contamination, immutable payloads, invalid imports and handles, cancellation,
  zero pivots, failed-prime recovery, caps/deadlines and publication rollback.
- [Fresh review](amortized_basis_review_attempt001.json): PASS,0 issues;
  341 artifacts,92 exact saved-system residual checks,13 unresolved systems
  preserved,28 new-mode saved differentials and3 reviewer mutation rejections.
- Four-mode cost attribution and validation/guard counts are recomputed from
  raw control records by the review, not copied from this prose.
- Fourteen freshly relocated `python -I -S` original-LP checks:11 feasible and
  3 expected rejections. These include frozen regression fixtures, not14 new
  independent LPs. Each new mode's multi-round analytic LP checks the same
  feasible upper bound `-1099511627776/7`; no network verdict follows.
- The review performs no new elimination or native solve. Its scope is exact
  saved-system and full LP checking, not an independent proof of every internal
  field operation. All previously sealed source and real artifacts hash-match.

Attempt001 (105 passing tests) is retained; attempt002 adds a late-publication
rollback repair/control and binds the current source. No outcome threshold was
changed. Raw synthetic artifacts remain under the recorded `data/moe/results`
directories; only source, compact receipts/hashes and documentation enter Git.

## Next boundary

The requested amortization research/control stage is complete. Keep it optional
and isolated. Do not expand real experiments or remove checks. If proceeding,
separately validate integration under one outer request budget, deadlines and
partial evidence, and retain a same-version no-plan reference before claiming
net plan-reuse value. The four original real diagnostics remain4LIMIT and zero
checked feasible endpoints; this control result does not change them.
