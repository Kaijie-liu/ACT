# Checked binary64 outer enclosure for HybridZ

This is a finite mechanism design frozen after the [metadata and arithmetic
diagnosis](hz_real_intake_20261001.md), before implementation. It addresses a
specific obstacle to the real-model goal: an exact reference HZ can have rational
coefficients that do not fit the live binary64 sparse representation. It does
not admit larger models, change current caps, retry a GPU, or replace any old
proof or production default. No speed or certificate gain is the acceptance gate.

## Scope and inclusion contract

Let the exact reference use original factors `z`, continuous in `[-1,1]` and
binary in `{-1,1}`. The binary relaxation, when used for support, remains explicit.
Its output is `c+Gz`, with constraints `Az≤b` and `Ez=h`. Reference identities,
factor types, shared input prefixes and expert-private ownership are immutable.
The actual sparse target stores finite binary64 coefficients interpreted exactly
as their binary rational values. It adds only fresh continuous error factors.

The required theorem is an **outer inclusion**: for every feasible reference
assignment, an extension of the same original factors is feasible in the target
and has the same output. It is not equality of the two sets or proof of native
floating execution. Neither exact equality acceptance nor old matrix certificates
may be reused for the new enclosure matrix.

For an output row, choose finite stored `c̄,Ḡ` and a stored nonnegative radius

    ρ ≥ |c−c̄| + Σ_j |G_j−Ḡ_j|.

Represent it as `c̄+Ḡz+ρη`, `η∈[-1,1]`. The exact residual determines a valid
η; with ρ=0, require the residual identically zero and add no factor.

For an equality row, choose finite stored `ē,h̄`, and

    σ ≥ |h−h̄| + Σ_j |e_j−ē_j|.

Store `ēz+σζ=h̄`. At a reference feasible point, choose
`ζ=(h̄−ēz)/σ`. Its absolute value is at most one. With σ=0, the equality is
unchanged exactly. New equality slack factors are distinct from all output
errors and all old continuous/binary factors.

Output-error columns are zero in every constraint. Each equality slack is zero
in every output and every other constraint. Original continuous and binary
coordinates have explicit separate maps: appending continuous columns changes
the binary offset in a flattened `[continuous,binary]` vector. It must not be
treated as an unchanged flat prefix. A shared entry is lifted once; expert-owned
new factors cannot be shared merely because their numerical coefficients match.

For an inequality row, choose finite stored `ā` and a stored RHS satisfying

    b̄ ≥ b + Σ_j |a_j−ā_j|.

Then `āz≤b̄` holds for every reference feasible point. A weakened guard remains
an outer approximation of that declared legal branch; it must not be used as an
exact route-feasibility witness or as authority to discard other tie-legal routes.
The RHS calculation uses exact original `b` plus the exact coefficient error;
rounding `b` first and forgetting its error is forbidden.

The first producer uses deterministic nearest finite coefficients and the least
available binary64 outward radius/RHS. The checker need not trust that rounding
routine: it recomputes the rational residuals and verifies inclusion independently.
No finite coefficient/enclosure, malformed bounds, NaN/Inf or overflow means
explicit rejection, not clipping or dtype fallback. Signed-zero canonicalization
must preserve identities consistently; subnormal underflow needs compensation.

## Frozen finite controls

Controls use small supplied exact-reference HZs only, at most eight original
factors, four output rows and four rows of each constraint kind. The following
source patterns are fixed; no trained object or full output query is run:

1. Exactly representable identity, no new factors and unchanged constraints.
2. The diagnosed affine error `2^-104+2^-200`, requiring outward output error.
3. The diagnosed ReLU range endpoint `2+2^-54` in an output expression.
4. The diagnosed equality RHS `2^-56−1/2`, requiring equality compensation.
5. The diagnosed guard difference `1−2^-54`, requiring an outward inequality.
6. Half-minimum-subnormal output/row coefficients, including negative values.
7. Shared input with two separately owned expert factors: distinct ownership and
   an explicit embedding, no identification of private factors.
8. Zero-width/exact rows, zero coefficients and empty equality/inequality blocks.
9. Overflow/no finite outward endpoint, rejected without partial acceptance.

For the finite valid patterns, exact assignments and their calculated extensions
must satisfy target constraints and preserve outputs. They supplement, rather
than replace, the algebraic inclusion checker. Independently vary output signs,
RHS, radii, row inventory, factor maps, kind/owner, reference binding and source
identity; insufficient compensation, collisions or missing rows must reject.
Expired cooperative deadlines and attempted source mutation must not accept.
The checker must not call the producer or numerical optimizer.

The actual target must instantiate SparseHZono and round-trip without coefficient
loss. For exactly representable controls, its support representation is unchanged;
no solver or candidate search is needed. Existing mathematical modules and
accepted source/HZ/endpoint tests must remain unchanged and pass. All new files
are opt-in and versioned. Preserve any failed attempt and report all controls.

## Integration and stopping boundary

The new kernel first connects **given exact reference** to actual HZ only. A
later separately scoped source integration must check the original affine/ReLU
or conditional-guard reference first, then this lift, with new hashes and fresh
output evidence. Do not replace `radius == needed` in the old affine checker
with an unchecked larger float and call it equivalent.

Extra factors and weakened constraints may reduce useful bounds. Successful
inclusion does not establish real capacity or improved coverage. Report those
costs when a subsequent full-path comparison is frozen; do not tune this finite
control until it gets a positive output. Shared-block/template factoring and
query batching are separate mechanisms. No new supervision layer, real request,
larger timeout, holdout or sealed input is part of this control stage.
