# C118 conditional exact sum-factor lemma

This is an algebraic statement, not a native candidate, stored-data observation,
completed source-resource proof or new solve. The census does not emit this HZ.

Let x_j be DISTINCT retained continuous coordinates, each in [-1,1], occurring
in an otherwise arbitrary hybrid zonotope with continuous/binary variables,
equalities and inequalities. Correlations between x_j are unrestricted. Let
their physical values be v_j = 2^e_j x_j. Assume selected raw operator columns
are the same complete vector w, so their output contribution is

    w * sum_j 2^e_j x_j.

Set m = min_j e_j, T = sum_j 2^(e_j-m), and

    s = m + (T-1).bit_length().

This computes ceil(log2(sum_j 2^e_j)) without a rounded logarithm or sum.
Introduce one new continuous z in [-1,1] and the exact equation

    2^s z - sum_j 2^e_j x_j = 0.

Replace the selected output contribution by w * 2^s z. Retain every original
coordinate, binary variable, original predicate and output pivot.

## Equivalence and reconstruction

For every original feasible assignment, the displayed equality determines one
and only one z. Its magnitude is at most sum_j 2^e_j / 2^s <= 1, so the added
box is redundant. Every original feasible point therefore extends uniquely.

Conversely, substituting the new equality into the transformed output recovers
the original contribution exactly. Projection onto the old coordinates
preserves all old predicates and outputs. The original-input inverse is the
unchanged inverse on those old coordinates; z itself is reconstructed by the
displayed sum. No latent identities are merged and no nonconvex binary choice
is relaxed. The proof does not require independent parents.

## Conditional cost

For q columns with d nonzero output incidences each, q*d coefficients become
q+1+d: q parent terms, one new pivot and d routed output terms. The reduction
is (q-1)*d-q-1. Existing output pivots appear on both sides. Zero or negative
reductions are rejected; q=2,d=3 is neutral, q=2,d=4 saves one coefficient.

This raw count is not the final native bill. Compose the original and candidate
equations with the inherited quotient map first: anchors can coalesce, signs
and dyadic multipliers can differ, and removed factors cannot be resurrected.
Recount all changed rows and retain their existing logical output pivots.

Native admission additionally requires exact finite forward/backward ldexp
checks, complete-row positive dyadic gauge selection, and the unchanged
nonzero coefficient window [2^-20,2^40] on new definitions and changed outputs.
The entire changed row, including untouched terms and pivots, must qualify.

Shared whole-source reserves are 16,384 auxiliaries, 131,072 positive extra
entries and 16M emission work, with old usage included. The historical
conservative emission tariff includes all new circuit rows, not just sum
definitions: 64R +16(N+R). Planning, binding, inverse and full physical proofs
still fit the independent whole/branch caps. C96 credits cannot be transplanted
unless its corresponding actual emitter/proof is reused.

Consequently, even a positive exact-column census only creates a candidate for
further proof. It does not establish new source/native/LIVE admission, memory
reduction, runtime improvement, replay preservation or benchmark gain.
