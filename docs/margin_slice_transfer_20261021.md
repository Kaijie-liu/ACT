# Margin supports and strong slicing comparisons

The current scalar margin certificate can be reproduced by transferring its
checked duals to the same closed margin slices. Rechecking the combined duals
can produce a higher bound without new support queries. Consequently A2 must
test a bound-versus-cost tradeoff, not claim that the current representation is
inherently stronger than unrestricted slicing.

This relationship follows from the current rational residual formula and
closed-cell aggregation at `198cd4fbfd0ed443c025e5fff2f4dd93ba0a8c26`.
It does not establish an experimental advantage, a new LP-duality theorem or
full-model safety. The diagnostic below only reads the archived A1 evidence.

## Exact dual transfer

Let P retain the original shared/private factors z in [-1,1], inequalities
A z<=u and equalities E z=v. Write m=m0+M z. A support certificate for
o=b+t d+nu m uses y<=0, free equality multipliers eta, and residual
r=o_vec−A^T y−E^T eta. Its checked lower is

    L = o0 + u^T y + v^T eta − ||r||_1.

For a closed cell C=[x,ycell], append both exact inequalities, without deleting
or merging any original row:

    M z <= ycell−m0,       −M z <= m0−x.

For the new objective b+t d, retain the old duals and assign the two new
multipliers as follows:

| Tilt | Upper-row multiplier | Lower-row multiplier | New checked lower |
| --- | ---: | ---: | --- |
| nu>=0 | −nu | 0 | L−nu*ycell |
| nu<0 | 0 | nu | L−nu*x |

The objective residual is exactly unchanged. The affine m0 term cancels in
the offset, so no zero-center assumption is used. This is exactly the current
cell rule `L+min(−nu*x,−nu*ycell)`. Taking the strongest saved support at each
anchor t yields A_t(C); ties use the first saved record. The construction is
valid on a continuous HZ outer approximation even when the original network
has binary factors. It does not prove an integer optimum or exact lowering.

## Combining before residual compensation

For two anchors l<h and a local gate endpoint g in[l,h], put
theta=(h−g)/(h−l). Combine the complete transferred dual vectors at l and h
using nonnegative weights theta and1−theta. These remain valid duals for the
same P and slice, now with objective b+g d. If their residuals are r_l,r_h,
the recomputed bound is

    L_transfer(g,C) = theta*A_l(C)+(1−theta)*A_h(C)
      + theta*||r_l||_1+(1−theta)*||r_h||_1
      − ||theta*r_l+(1−theta)*r_h||_1.

The final correction is nonnegative by the triangle inequality. Minimizing the
two checked local endpoint bounds certifies the entire local gate interval;
minimizing over all original closed cells covers the same P. It is therefore
no weaker than the current scalar-only interpolation, and may be stronger when
residual signs cancel. No candidate optimizer or new sigmoid evaluation is
needed for this transfer. Obtaining the original tilted certificates is not
free: a competitive sliced-reuse path must pay for those same queries.

Exact fresh endpoint support on the same cells also dominates the current
scalar bound, but may require up to two goals per cell. The current A1 record
uses eight cells and at most four global objectives; a same-cell fresh sliced
reference may use sixteen local objectives. Counting only global queries and
ignoring scalar cells, or giving a derived-certificate baseline free global
queries, would conceal this tradeoff. Neither exact dominance result implies
finite-budget performance dominance by an arbitrary numerical solver.

## Existing baseline capabilities and guarantee mismatches

N1 already accepts arbitrary cut points and tightens difference bounds within
each slice. A comparator with only global difference bounds would weaken it.
M3 already adds sign/Lipschitz gate–margin rows; these are not a new mechanism.
Native N1/F0 uses floating projections and ranges, whereas the margin path
checks its given continuous HZ. In particular the native gate helper requests
`relax_binaries=False`; integer-only ranges cannot silently be used for a
same-continuous-P comparison. M3's row checker also trusts slice coverage and
original F0 lowering. Run those original paths only under their actual guarantee
labels, or supply independently checked exact slice construction/ranges.

Direct rational McCormick construction and its independent checker already
exist in `act/back_end/solver/rational_mccormick.py` and
`check_rational_mccormick.py`. They expect the expert-only2C output view;
the native margin source has2C+2 outputs. A future baseline adapter must bind a
lossless expert view, retain P and the router binding, and continue the same
deadline/Work. It must not present these existing formulas as new research.

## Frozen saved-evidence diagnostic

Read only the three scalar records in
`margin_scalar_records_20261021.json`. Recheck their original source/property
and certificates. For each of their eight cells, select the current maximum
saved line at each anchor, with first-record ties; transfer and blend its duals
at both existing gate endpoints. There are48 local endpoint receipts,48 anchor
transfer checks, zero new support calls and no parameter or source change.
Keep every original row, nonzero margin constant, exact bit check and binary
relaxation declaration. Independently reproject original expert outputs and
recompute every residual against P plus the two exact slab rows.

Record each cell's selected support identities, combined duals, lower, residual
cancellation and original lower; record the complete minimum and costs. Do not
search all certificate pairs, tighten ranges, change cuts or add solves. The
input archive, diagnostic code, existing checker dependencies and this protocol
are hash-bound before one invocation. The result directory is
`/data1/Kane/MOE/results/margin_slice_transfer_20261021_r1`.
Single stdlib process, at most300s, RLIMIT_AS8GiB, original rational bit checks,
development Work accounting and128MiB artifacts including archive. No children,
new supervisor, checkpoint/data, native library, model or GPU. All work is
offline diagnostic cost; it cannot repair or upgrade the earlier online result.

Unexpected rejection, formula mismatch or weaker cell bound stops the attempt
and preserves FAIL; no repair/retry. A strict increase identifies a specific
loss from scalarizing residuals early, not a unique real-model bottleneck.
Exact equality still establishes reproducibility by slicing, not source
infeasibility or lack of possible cost benefit. Both outcomes require A2 to
include a priced shared-certificate reference and a genuinely strong fresh-slice
reference. No saved positive result permits automatic external deployment.
All G goals and the real-development budget remain unchanged.
