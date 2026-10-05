# Next same-structure implementation: owned normalized precision statistics

C64's actual code is now qualified on1466 inherited/new tests; the complete
source-cost checker passes its four new tests. The entire actual source still
fails the256M construction gate: upper257193403, optimistic lower257132585.
Its full100965 boundary and all37376 complete row gauges match the independent
source proof. Do not narrow the boundary, raise a cap or retry unchanged v1.

## Specific implementation opportunity, NOT yet executed

The complete profile has216448 non-sum external coefficient occurrences and
100608 sum/dyadic occurrences. C64 calls the general c62 odd_significands reader
for the first group, although every row was just emitted by C31's exact owned
normalizing encoder. That reader re-extracts exponent bits and checks zero,
subnormal and nonfinite encodings on every value. The fresh encoder has already
proved the stricter nonzero normal[2^-20,2^40] domain before publishing its row.

A NEW source-specific primitive may calculate the exact odd significand
directly from the mantissa bits of such an owned, normalized row. Its contract
must be tied to actual successful source-row emission, never an arbitrary
external vector, deserialized flag, hash, or old proof receipt. Keep the general
external reader and the independent complete source/domain oracle unchanged.
The new primitive is a producer-specific algebraic specialization, not weaker
overall acceptance of invalid numeric inputs.

The concrete removable operations in the current general reader are exponent
shift/mask, zero comparison, all-ones comparison, OR and any reduction: six
vector operations per non-sum hit. If (and only if) the new implementation
actually removes these operations and all producer/proof bookkeeping is paid,
the existing profile suggests6*216448=1298688 work can be removed. Prospective
construction upper255894715 leaves only105285 headroom. This is a hypothesis,
not measured work, a waived check or permission to discount unrelated prices.
Any necessary additional binding cost must fit that headroom or the version
fails before an expensive real generator. Whole native/LIVE costs are not
included and must never be called paid by this number.

Do not simply lower the inherited8*row-width maximum tariff: v1 already reuses
its hit positions, so relabeling an unchanged body after seeing a failure is
not a newly removed operation. C19 also documents why merely reviving sparse
changed-only sorting was not profitable; its old boundary census is not a new
universal row-order theorem. Keep the focus on this ordinary fresh-owned
coefficient structure, not an unrelated sorting/corner-case campaign.

## Completion obligations

1. New versioned files and explicit default-off owned normal primitive; no
   frozen C64 edit. Exact ordinary domain theorem and reference-reader equality,
   real source-row producer binding, unchanged upstream invalid-input guards.
2. Derive a new complete prepaid work bound with all new operations, preserve
   the full100965 optimum, all EQ/INEQ/radix/owner deltas and local inverse tags.
3. Keep all57 original qualification files and the four source-cost tests;
   rerun relevant changed-source qualification. No skipped failures, favorable
   subset, new arbitrary test framework, unavailable reader, or old receipt.
4. Finish the new full-source/box/local-inverse/owner/UID checker and same-C31
   reachable numeric+Python storage metric before claiming an admitted state.
   New C64 tags are NOT legacy aliases()/Closed tags. The separate source
   expressions may have different object IDs; old opaque-ID cancellation is
   insufficient for Python metadata. Include new proof/gauge/report storage.
5. Register and execute the real fresh original-expression generator only once
   its complete bound fits; require actual operations <= each bound group,
   measured both1GiB,64M entries and all graph-retirement/source-preservation
   obligations. A count or model is not physical HZ publication.
6. Source construction and offline qualification may use only C31's explicit
   boundary with all diagnostics separately reported. They do not prove the
   combined LIVE/native/source-first path, terminal/network witness, same-
   structure shadows or the full2413 and separate400 retention replays.

No next-candidate source, preregistration or job has been created. Every C64 job
is terminal with retained output. No confirmation is needed for this scoped
next implementation; no new permission or broader objective is inferred.

CPU1/GPU0,AS16GiB,60s tests/240s worker,64M entries,256M whole/200M nested,
both1GiB transient,radix16384/131072/16M remain. Genuine nonconvex HZ and every
original invariant remain; no convex replacement/binary pivot/solver rescue/
iid menu. Formal1870/all13 families and E0 CIFAR25/Tiny36 remain unchanged.
All history read-only; new exclusive source/results, no commit/push. Goal ACTIVE.
