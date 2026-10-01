# Sparse row enclosure with exact residual accumulation

This design follows the [block capacity assessment](hz_block_capacity_20261002.md)
at `3e5d524a3e0ec0aeda6629d65069d08516b8a97d`. It freezes one given-reference
mechanism before implementation: replace row-by-row whole-state reparsing and
the eight-factor local projection with a canonical sparse-row pass. It does not
change the enclosure mathematics, input semantics, support algorithm, numerical
acceptance gate, or any production default. No real model or output solve is
admitted. Old finite implementations and their limits remain unchanged.

## Mathematical contract

Keep every original continuous factor in [-1,1] and binary factor in {-1,1}, with
its original ID, owner, frame and column. Sparse omission means a zero coefficient,
not permission to omit a source coordinate from the identity inventory. Shared
factors stay shared; private expert factors stay distinct.

For an exact source output c+g z and finite binary64 target coefficients cbar,gbar,
independently compute

    delta = abs(c-cbar) + sum_j abs(g_j-gbar_j).

The sum covers the union of original source and target nonzero columns, including
target-only original columns. A finite stored rho >= delta gives the enclosure
`cbar+gbar z+rho eta`, where eta is a fresh continuous factor. With rho=0, no new
factor is allowed. For a source equality e z=h, use the same rule with
`abs(h-hbar)` and store `ebar z+sigma eta=hbar`. Each such slack appears only in
its own equality. Output errors appear only in their own output row and in no
constraint or other output row. All new error/slack IDs are mutually distinct
and disjoint from every original continuous/binary ID. For an inequality,

    bbar >= b + sum_j abs(a_j-abar_j).

Use the exact source b, not an already rounded RHS. This covers a reference
feasible assignment by extending the **same** original z; it does not assert
equality of sets, network correctness or floating-execution safety.

The proposed producer uses nearest finite coefficients and one outward rounding
of the complete exact residual/RHS per row. It adds at most one compensation
factor per output/equality, never one per chunk. Splitting the sum into finite
work chunks must not change rho or any factor identity. The checker independently
recomputes these inequalities; it cannot trust claimed chunk totals or call the
producer's accumulation routine. NaN/Inf, unrepresentable outward endpoints,
overflow and overlarge rationals cause rejection, not clipping or fallback.

## Parsing and resource contract

Introduce separate opt-in producer/checker modules and a new proof schema. Each
checker call anchors canonical source and target bytes and parses each once,
then visits rows using owned structures decoded from those anchored bytes, not
borrowed mutable row dictionaries. It still verifies the caller's source and
proof identities before returning. There is no cross-request cache, alias reuse
or skipped check. Hash and canonicality work stays visible;
this is not a constant-memory streaming proof format.

Use canonical sorted sparse coordinates and bounded work chunks of at most 256
entries. Every union entry is checked exactly once in its row residual. Check
deadlines inside these chunks as well as at phase boundaries and after final
identity verification. No dense scan over all global columns per sparse row.
The implementation may sort actual sparse keys; count that work rather than
claiming strict linear wall time. Target-added error columns are handled by a
separate complete layout check, not hidden in the original-column residual.

The new **given-reference** admission is at most 4096 original factors, sixteen
outputs, sixteen equalities, sixteen inequalities and 262144 stored nonzero
coefficients in each reference or target; each stored rational numerator and
denominator retains the 4096-bit limit. Factor IDs are unique strings of at most
200 characters and owner labels retain the old 48-character limit. Exact
intermediate numerator/denominator growth beyond 8192 bits is a recorded resource
refusal, never a rounded residual or an accepted partial sum.
The target may add at most output_count+equality_count continuous factors. These
are explicit bounds for this row-only mechanism, not changes to existing source,
block, support-query or portable limits. Validating dimensions does not admit a
whole model. Partial output or a reached deadline is never an accepted enclosure.

All work uses CPU and the existing act-py312 environment, one cooperative deadline
of at most 300 seconds per complete control: construction, conversion to actual
SparseHZono, serialization and independent checking are charged. Imports and test
orchestration are separately reported. No optimizer, model/data loading, GPU,
hard-supervision claim or performance confirmation belongs to this stage.

## Frozen controls

Use the eight existing valid exact references in `test_hz_binary64.patterns()`
and both existing overflow references without changing their coefficients.
Compare the old and new target coefficients, ownership, maps and residual bounds
exactly; schema/receipt labels may differ. Both targets must instantiate actual
SparseHZono without coefficient loss. Keep the existing exact point embeddings
as supplementary checks, never as the universal inclusion argument.

Add one fixed wide supplied reference, not a network or a support query. It has
1536 continuous factors `input/0` through `input/1535` owned by `shared`, and 1536
binary factors `expert0/b/0` through `expert0/b/1535` owned by `expert0`. Use frame
1, enclosure owner `wide/row`, two outputs, one equality and one inequality.
In each factor namespace use

    v_j = (-1)^j * (1 + 2^-52)^2 / 2^(j mod 3),  j=0,...,1535.

The first output has c=1+2^-54 and both coefficient rows v; the second has the
same original factors with both rows -v and c=0. Equality rows are v in both
namespaces, h=0; inequality rows are the same with b=1-2^-54. The all-one original
assignment is feasible because each six-term coefficient block sums to zero.
Check its exact extension and unchanged original coordinates. This is a wide-row
inclusion control; its outputs are not MoE certificates. Do not tune values or
add more widths after observing results.

An analytic expectation, fixed before execution, is total coefficient residual
`D=7*2^-96`, output radii `2^-54+D` and `D`, equality radius `D`, and inequality
RHS rounded outward to `1`. The all-one extension has new factors
`2^42/(2^42+7), 0, 0`; the binary flat offset moves from 1536 to 1539. These are
algebraic expectations for a control, not already executed results.

Required groups are:

1. Eight old valid references, exact old/new target and bound differential.
2. Wide mixed continuous/binary row, complete sparse union and exact embedding.
3. Same wide reference at chunk sizes 1 and 256, identical target and residuals;
   these are deterministic differentials, not a timing search.
4. Shared/private ID, frame, owner and flattened binary-offset preservation.
5. Source binding, missing/duplicate rows, malformed/duplicate sparse coordinates,
   target-only original columns and compensation-zero-layout mutations.
6. Inward output/equality compensation, inward inequality RHS and wrong signs;
   choose concrete mutations with insufficient compensation, not arbitrary
   coefficient changes that could still form a valid outer enclosure.
7. Both original overflow refusals, subnormal preservation and finite resource
   admission/refusal; resource refusal does not establish an unsafe model.
8. Producer/checker expiry during a chunk and final identity work, source/target
   mutation attempts, and incomplete proof refusal.
9. Checker independence from producer/numeric libraries, source/target parse
   counts, complete coefficient visit counts and missing inventory detection.

Save actual references, proofs, embeddings, concrete mutation inputs and ordered
cost intervals. An independent audit reconstructs the frozen control inventory
and rechecks it without the producer or solver. Unexpected timeout is not a
successful corruption rejection. Preserve failed attempts and original limits.

## Stop and next gate

Stop and repair if coefficient/ownership differential or inclusion fails; never
weaken the gate to make a control pass. On success, report only a checked wide-row
primitive and removed repeated parsing, not a new tightening theorem, speedup or
real-model proof. Do not repeat timings or introduce a wider/easier fixture.

Only then decide a separately scoped source integration: every affine/ReLU/guard
reference must still be checked, all fresh matrices need fresh output evidence,
and global dimensions, property streaming, source storage and complete-budget
admission remain unresolved. This protocol does not authorize another supervisor
layer, sealed input, physical GPU retry or full-size model execution.
