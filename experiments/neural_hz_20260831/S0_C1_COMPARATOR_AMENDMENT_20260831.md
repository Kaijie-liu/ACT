# S0-C1 Comparator Amendment

Locked on 2026-08-31 before any composed-stencil production run. This file
supplements, but does not overwrite, the hashed
`S0_C1_COMPOSED_STENCIL_PREREG.md`. It separates score, external-retention,
representation, and implementation-performance comparators so that a later
result cannot select whichever baseline is favorable.

## Four disjoint comparators

1. **Formal-score comparator.** The only 13-family score baseline is
   1,870/2,413. No S0 intermediate-layer or external-family result changes it.
   Promotion requires one full 2,413-case candidate replay with every old
   CERT/validated ADV and every per-family solved count retained.
2. **External-outcome comparator.** CIFAR100/TinyImageNet use E0 v2: 61
   independently validated historical-origin ADV plus 339 UNKNOWN. A full
   400-row single-path candidate must retain all 61 and can gain only on the
   339 UNKNOWN rows. E0 is never added to the formal score.
3. **Representation comparator.** S0-C1 compares the complete combined
   `phase-sliced + implicit-Conv + composed-stencil` candidate with the same
   phase-sliced HZ mathematics and the same target, row selection, checkpoint,
   budgets, cache policy, and materialization boundary, but with its exact Conv
   maps represented through the unfused expanded/support-sliced path. This is
   the `phase_selective_expanded_v1` comparator intended by “the candidate's
   unfused phase-selective baseline” in the original preregistration.
4. **Implementation-performance comparator.** Trial 9's
   `phase-sliced + implicit-Conv + unfused row-oracle composition` path is a
   diagnostic implementation comparator only. A composed descriptor may claim
   exact work compression relative to Trial 9, but it may not claim HZ
   simplification merely because it runs faster or avoids scalar products.

The representation comparator was fixed before the first real-network run of
the composed-stencil candidate. Trial 8/9 capability measurements motivated
the structure but cannot be unioned into a candidate score.

## Whole-state physical transaction

At each registered checkpoint, both sides charge the same live-state boundary:

- HZ center/value maps and all equality/inequality predicate matrices;
- continuous and binary slot metadata, bounds, phase data, frame and witness
  ledgers;
- every uniquely reachable exact linear operator, kernel, coefficient, index,
  mask and bias buffer;
- retained materialization/suffix caches and their CSR buffers; and
- controlled construction temporaries plus measured process peak RSS, with
  allocator/workspace omissions stated explicitly.

Objects shared by identity are charged once; logical expanded nnz, unique
resident bytes, controlled transient peak, materialized nnz, wall time and
peak RSS are reported separately. S0-C1 passes the representation gate only if
the candidate's unique reachable physical state is strictly smaller than
`phase_selective_expanded_v1` at every registered comparison boundary needed
for advancement. A local descriptor reduction, a lower logical work count, or
a lower wall time cannot substitute for this inequality.

The composed descriptor may temporarily use more memory than Trial 9. That is
permitted only inside the frozen transient cap and must be reported honestly;
it does not invalidate a whole-candidate reduction against the registered
expanded comparator, and it does not create a standalone simplification claim.

## Advancement rule

After default-off equivalence/resource tests, the fixed order remains Tiny
ReLU36, ReLU63, ReLU71 and terminal; registered CIFAR targets and guards;
same-structure residual/plain-CNN shadows; the formal Conv cohort; and only
then the full 2,413 replay. Every result records both comparator identities,
candidate source/configuration hashes and all four ledger namespaces. Failure
of exactness, witness reconstruction, any frozen resource gate, physical strict
decrease, E0 retention, or formal per-family retention closes that candidate
version without rewriting this amendment.
