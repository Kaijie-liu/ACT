# Rank obstructions to existing shared secant generators

D016 needs existing source arguments satisfying g_r=g_s+c*g_h. A cheap
necessary-condition test can prevent investing in GPU compression of a motif
absent from the selected network. This note proves conditional exclusions; it
does not report that any actual model satisfies their premises or lacks D016
motifs. Date 2026-09-30, redu-hz at
f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac. No numerical census accompanies it.

## Affine independence obstruction

Let g_i(x)=b_i+w_i*x on a source set containing an open subset of its free
coordinates. If the gradients of three distinct arguments have rank three,
no identity g_r=g_s+c*g_h with c nonzero holds on that source. Differentiating
the identity would give w_r-w_s-c*w_h=0. Biases cannot repair independent
gradients. Conversely, dependent gradients do NOT prove a valid identity:
the coefficients and constants must still agree.

Fixed box coordinates and source equalities matter. First project to the actual
free affine hull, or decline this exclusion. Rank computed using a coordinate
that is fixed in the admissible source is not a valid negative certificate.
Pair independence and nonzero gradients handle repeated-role identities;
g_r=g_s with c=0 is a zero secant, not useful new nonlinear structure.

## Dense convolution supports

Assume a group-one 3x3 convolution, stride and dilation one, pad one, and spatial
height/width at least four. Every relevant raw coefficient and preprocessing or
postprocessing scale must be certified nonzero. Actual nonzero support then
equals the clipped geometric patch. Nonzero post-BN scalars and input-channel
scales are invertible row/column scalings, so gradient rank can be checked on
the appropriately clipped raw rational kernel coefficients. Do not approximate
an irrational BN scale into a modular matrix; establish that it is nonzero and
strip the scaling algebraically.

For three source rows there are three support cases.

1. Same spatial position: certify rank three for their channel rows restricted
   to the corresponding clipped kernel mask. With at least three channels,
   certificates for every triple also imply every pair is independent.
2. Two rows at support S, one at support T: if T minus S is nonempty, its nonzero
   singleton coordinate forces that row's relation coefficient to zero. Pair
   independence on S finishes the proof. If T is properly contained in S,
   certify rank two for the pair restricted to S minus T instead. Empty or
   insufficient difference support is unresolved, never a passed certificate.
3. Three distinct spatial positions: a pixel belonging to exactly one support
   forces its row coefficient to zero. Distinct dense supports make the two
   remaining rows independent.

Under the geometry above, the exclusive pixel needed in case 3 always exists.
In one dimension write I(t)=[max(0,t-1),min(H-1,t+1)]. For t1<t2<t3, either
I(t1) has a uniquely smallest endpoint or I(t3) has a uniquely largest endpoint.
Both failures would force t2=1=H-2, hence H=3, contrary to the premise.

In two dimensions, three distinct row indices use this fact on rows; one row
index uses it on columns. With two row indices, either the singleton row has an
exclusive endpoint, or it is boundary-nested in the doubled row's support. In
the latter case the doubled group has an exclusive row; its two different column
supports provide an exclusive column for one member. Any input channel with
the certified nonzero coefficient supplies the required source coordinate.

Thus a cached finite set of channel-rank certificates plus support geometry can
cover all spatial/channel triples, even across a union of windows. For 64
channels, an analytic sufficient bound is nine clipped masks times C(64,3),
or 374976 rank-three tests. Proper nesting needs at most sixteen difference masks
times C(64,2), or 32256 pair tests: six horizontal stripes, six vertical stripes
and four L-shaped differences. These are upper bounds from geometry, NOT measured
successful certificates. Unsupported stride/padding, zero weights/scales and
deleted fixed coordinates require their own proof; this theorem cannot be used
to omit them from a census denominator.

## Exact modular certificates suitable for GPU batching

Let W have exact rational entries whose denominators are invertible modulo the
fixed prime p=32749. Reducing n/d to n*(d modulo p)^(-1) modulo p is exact. For
any integer projection P, a nonzero rank-r minor of W*P over the finite field
implies rank_Q(W)>=r. A projected nonzero determinant therefore certifies
independence; a zero determinant proves nothing about dependence or usefulness.

A fixed three-column projection may use P_j=(1,j,j*j) modulo p. Both projection
and reduction can lose rank, so failed certificates remain unknown. Before GPU
integer operations, enforce D*(p-1)^2 <= 2^63-1 and
6*(p-1)^3 <= 2^63-1, along with shape, residue and index checks. Then the intended
int64 products/sums have no overflow. Parallel execution changes wall time,
not the charged scalar work. The separately preregistered GPU fixture checks
arithmetic readiness only; it does not certify original weights.

## Consequences for the next domain decision

If all premises and ranks are certified on a complete preregistered first-bank
population, D016 has no nontrivial literal existing-gate anchor triples there.
That would be evidence to revise its source-identity hypothesis rather than
optimize that absent pattern. It would NOT exclude later nonlinear prefixes,
paid construction of additional anchors, or a different Neural-HZ definition.

No such empirical conclusion has yet been obtained. The proof and GPU readiness
step are kept separate so a synthetic arithmetic pass cannot become a claimed
negative census or a Neural-HZ verification gain. The previous D016 turn was
progress through its exact and compositional theorem; this turn adds a testable
applicability obstruction rather than reclassifying that theorem as a real solve.
