# C46: compose exact half-gauge inversion with the old witness lineage

This is a conditional algebra/implementation lemma, not a new source or native
admission certificate. The full1870/2413 goal,13-family retention and separate
E061 ledger remain unchanged. All symbols below are global latent identities.

## Required independently established inputs

The enclosing transaction must establish the original C32 post-HZ and its
paired exact old lineage, the complete original half-factor incidence degrees,
and the C40 transformation preconditions. C40 removes definitions
`p*z - s*p*u/2 = 0`, with p an admitted power of two and s in{-1,+1}. Selected
parents are distinct and outside the selected cohort; every gauged original
row excludes ALL selected parents. Selected factors are output-dead, binary-
free in their definitions, and the complete inverse covers their full original
incidence. No binary factor or input coordinate is eliminated.

C46 additionally checks that half children and parents do not name a variable
already reconstructed by the older unit/alias lineage. This follows from
proper full incidence for a valid current state, but is checked explicitly.
Original input coordinates precede every reconstructed factor.

## Old-row oracle

For an original post-C32 EQ row r, let k be the number of half definitions
strictly before r. A surviving row lives at new rank r-k. A deleted definition
is synthesized from its compact (column,parent,power,sign) record. INEQ row
numbers do not change. Gauge intervals are interpreted in the OLD post-C32
rank space, never reused as current ranks.

On a gauged surviving row, a coefficient at selected parent u maps back to
child z with old value s*new_value; every other coefficient, binary coefficient
and RHS maps back by division by2. The exclusion of ALL selected parents in
the original row makes this inverse unambiguous. On an ungauged row values are
unchanged. Deleted definitions have their original two continuous entries,
empty binary row and zero RHS. C40's full original-byte inverse additionally
checks the signed-zero descriptor; C46's rational row values treat both zeros
as the same exact real zero, as required by witness arithmetic.

Thus each emitted old coefficient is the exact real value of the corresponding
original post-C32 coefficient. The row reader need not sort the mapped terms:
Fraction addition is exact and associative, and prefix selection tests the
RESTORED original column identity, not its new parent index.

## Witness composition and order

1. Retain every original input/retained latent value. Restore each half child
   as `z=s*u/2`. Its latent box follows from |u|<=1.
2. Iterate the old C32 unit columns in their original order. First apply the
   old lineage's EQ-rank deletion to its saved pre-C32 consumer index; then
   use the C46 oracle to read that old post-C32 row from the new HZ. Use the
   unchanged C32 formula `(old_D_rhs - old_sign*prefix)/old_pivot`.
3. Reconstruct old aliases in increasing global-column order using their
   original exact ratios, including aliases of restored unit/half factors.

Because the oracle supplies exactly the coefficients used by the original
reader, each unit result equals old_lineage.reconstruct_fraction applied to
the original post-HZ after step1. By induction the same holds for every alias.
No original input coordinate changes. Binaries are not an argument to this
continuous reconstruction and are never rewritten. Feasibility preservation
still relies on the independently proved set transformations and a feasible
new point; C46 alone does not certify either native feasibility or a concrete
network/property violation.

## Ordered concrete symbolic family used for independent tests

Let p in{1/2,1,2}, s,sigma,t in{-1,+1}, and d in{-1/8,0,1/8}. Per block the
pre-unit equations are:

    -s*p*x/2 + p*u = p*d
    -sigma*p*u + p*z = -sigma*p*d
    z + r/4 = 0
    z - r/4 + b/8 <= 1,  b in{-1,+1}
    a = t*u/2   (old alias metadata)

After the unit splice the first two rows become one half definition:
`-sigma*s*p*x/2 + p*z = 0`. C40 deletes THAT SAME old unit consumer, and gauges
the remaining rows to `sigma*s*x+r/2=0` and
`sigma*s*x-r/2+b/4<=2`. The old unit witness must now read a row that is absent
from the new matrix. C46 synthesizes its original prefix, giving
`u=d+s*x/2`, then `z=sigma*s*x/2` and `a=t*u/2`. All equations, inequalities,
boxes and binary identities agree as exact-real identities; no sampled attack
or solver repair is involved. Test points additionally check full pre/post/new
constraints at literal zero tolerance.

## Implementation and remaining obligations

c46_half_lineage_extension_v1 uses only a new actual SparseHZono, packed half
metadata, intervals and old lineage. It verifies the complete inverse against
the SUPPLIED original anchor and brackets reads with full new-HZ and lineage
hashes. Supplied anchors/degrees are not self-issued source authority. The
reader retains no old post-HZ object and never constructs an old full CSR/HZ.
Tests/diagnostics intentionally retain original matrices outside the reader
for independent comparisons; their entire construction is measured.

Runtime integration still needs the independently bound real source/lineage
pair, original full LIVE roots, complete source/native payment, actual native
coordinate/input-recovery wiring, concrete network validation, shadows and
the full2413 replay. This lemma does not promote the candidate or close any
of those higher-level obligations.
