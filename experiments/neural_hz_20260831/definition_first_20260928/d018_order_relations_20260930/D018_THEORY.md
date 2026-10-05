# Anchored order relations and their limits for Neural-HZ

This paper investigation removes D016's exact three-source identity requirement.
Certified orders between otherwise generic affine preactivations support an
exact, phase-preserving replacement of the ReLU lower predicates. Two anchored
forests retain the original row budget and cannot weaken the original LP in
exact arithmetic. A mixed-sign forward composition rule is also available.
However, these results currently describe a stronger HZ predicate basis, not an
established new nonconvex domain. The known common-source disjunctive lift is a
stronger comparator, not an authorized implementation or a novelty claim.

Date 2026-09-30; branch `redu-hz`; commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, with existing dirty work preserved.
Configuration: paper derivation, independent mathematical audit and primary
literature review only. No candidate import, test, model execution, LP/MILP,
GPU kernel or physical-cost experiment. No implementation is enabled.

The [goal amendment](../../GOAL_DEFINITION_FIRST_AMENDMENT_20260928.md) remains
authoritative. This follows the [cross-domain supplement](../../literature_cross_domain_20260930_followup/REVIEW.md),
[D002](../D002_PHASE_FIBER_AND_RANK.md),
[D011](../d011_phase_value_audit_20260928/D011_RESULTS.md), and
[D014's source-gluing audit](../d014_guarded_amplitude_20260928/D014_FACTOR_AND_LIMITS.md).
Their negative results are not overwritten or presented as new discoveries.

## Semantic contract and candidate element

Let S be the owned, exact upstream HZ relation in its continuous coordinates,
all original binary coordinates and EQ/LE predicates. Its original input
decoder is retained. New preactivations f_i are affine forms on this SAME
source frame. Let l_i <= 0 <= u_i be the same valid gate bounds used in the
original comparison. The original gate has output r_i, bit beta_i and rows

```text
r_i >= 0
r_i >= f_i
r_i <= u_i beta_i
r_i <= f_i - l_i (1-beta_i)
```

An order-annotated element stores S, the same r and beta, affine readouts, a
certified preactivation-order graph and two anchored forests as specified below.
Its strong concretization contains original inputs, every original bit, and
visible values satisfying the predicates. Comparison aligns all shared source
and binary identities before taking inclusion. No best abstraction or complete
lattice is claimed. Empty forests mean every vertex is an anchor, reproducing
the original HZ exactly; arbitrary other HZ predicates and consumers remain.

The retained integer gates make this relation nonconvex: the one-gate graph
over [-1,1] contains (-1,0) and (1,1), but not their midpoint (0,1/2).
This is not a replacement by an order cone or a convex zone. Nevertheless the
element uses the original affine readout language, so its annotation alone does
not establish the requested definition-level innovation.

## Pair and chain theorem

Suppose f >= g on every exact source tuple. No strict gap or proportionality
is required. For q=ReLU(f), r=ReLU(g), keep all FOUR original upper rows and
replace the four lower rows by

```text
r >= 0
q >= f
q-r >= 0
q-r <= f-g
```

These imply q>=0 and r>=g. Every new relaxed tuple is therefore an old relaxed
tuple, even if an enlarged relaxed source did not independently imply f>=g.
For the reverse INTEGER inclusion, true ReLU is monotone and 1-Lipschitz, so
every old exact tuple satisfies the new rows. Thus the full same-source,
same-bit relation is unchanged and the LP projection cannot weaken.

No bit-order constraint is added. At f=g=0 every original combination of the
two bits is retained. This avoids the unsound inference that non-strict input
order implies ordered original bits. The exact source order must be certified;
testing a few input points is not a certificate.

For f_1 >= ... >= f_m, keep the original 2m upper rows and use

```text
r_m >= 0
r_1 >= f_1
0 <= r_i-r_(i+1) <= f_i-f_(i+1)       for i=1,...,m-1
```

The lower row count is exactly 2m. Telescoping from r_m proves every r_i>=0;
telescoping from r_1-f_1 proves every r_i>=f_i. The same exactness and
LP-containment conclusions follow. This is a parallel-bank relation, not
D002's serial positive-threshold-chain identity. Unlike D011's discarded
difference-only representation, absolute anchors and original uppers remain.

## Two forests cover partial orders at the same row budget

A certified graph edge i -> j means f_i>=f_j on S. Choose two separate acyclic
functional forests; every pointer path must terminate at an anchor.

1. Nonnegative-output forest: each vertex i has either the anchor r_i>=0 or
   one pointer to a lower ordered vertex j with r_i-r_j>=0.
2. Nonnegative-deficit forest: each vertex i has either the anchor r_i>=f_i
   or one pointer to a higher ordered vertex h with
   r_h-r_i<=f_h-f_i, equivalently r_i-f_i>=r_h-f_h.

Each forest has exactly m rows. Induction along pointers proves all original
2m lower inequalities. Every chosen edge follows from the true ReLU sector,
so strong integer equivalence and LP containment again hold. Cyclic unanchored
components are not accepted: they would leave an uncontrolled common offset.
The construction neither identifies nor eliminates any binary coordinate.

This theorem does not prescribe a tested discovery or selection algorithm.
Any eventual rule must select solely from certified mathematical structure,
not instance identities, labels, known outcomes, margins or solver states.
Forest metadata and order certificates must be charged, not treated as free.

## A strict relaxation control with nonparallel weights

Reuse the supplement's paper control, not a new network experiment:

```text
x,y in [-1,1]
f_1=x+y/4+1/2       in [-3/4,7/4]
f_2=x-y/4-1/2       in [-7/4,3/4]
```

Both gates cross zero. Their gradients have determinant -1/2, and
f_1-f_2=1+y/2>=1/2. At x=0,y=-3/4, set
r_1=5/16,beta_1=1,r_2=7/20,beta_2=1/2. The first gate is exact. The second
tuple is the equal mixture of its exact source/gate tuples
(1,-4/5,7/10,1) and (-1,-7/10,0,0). Hence even the individual full retained-bit
ideal hulls, and therefore the old big-M rows, admit this point. The new
nonnegative difference row excludes r_2-r_1=3/80>0.

This is a separation of formulations, not an error in integer HZ and not a new
monotonicity theorem. Ordinary HZ supplied the same joint relation excludes
the same point. No structural prevalence or property solve follows from it.

## Forward closure through mixed-sign affine layers

For an ordered nonnegative vector r, set z_j=r_j-r_(j+1) for j<m and z_m=r_m.
Then z>=0 and

```text
w*r = sum_j (sum_(i<=j) w_i) z_j.
```

Consequently w*r+b>=0 on the ENTIRE unbounded order cone exactly when b>=0
and all prefix sums of w are nonnegative. Sufficiency follows from z>=0;
necessity follows by taking z=0 or increasing one coordinate of z. On a
bounded, predicate-restricted neural source the test is only sufficient.

For next preactivations t_i=a_i*r+b_i, apply this test to adjacent row and bias
differences. It certifies t_i>=t_(i+1) without requiring each coefficient of
a_i to be nonnegative. Since the order is implied by the preceding new LP,
the next row replacement composes with the old LP-containment mapping.
Same-frame Affine/Conv, Add and Concat use ordinary aligned affine readouts;
new ReLUs without an order certificate keep their original predicates.

For the two-source control, consider

```text
t_1=2r_1-r_2-1/2
t_2=r_1-r_2-1/2.
```

Here t_1-t_2=r_1>=0. Both rows have mixed signs and a negative bias. At
x=-1,y=0 both t values are -1/2; at x=1,y=0 they are 2 and 1/2. Thus the
second layer also has genuinely unstable gates. The result is not limited to
stable pruning or zero-bias source identities. It provides an exact two-layer
composition, not continuous-factor elimination.

For partial forests, only output-order edges actually implied by retained
predicates may be used for relaxed-source propagation. The full original
preactivation-order graph is not automatically an enforced output-order graph.
Shared residual branches must keep the same source identity. Using separate
independent copies is not a sound substitute for aligned composition.

## Complete symbolic cost and the compression limit

Assume distinct owned source/output coordinates, with s_i nonzeros in f_i and
s_ij in f_i-f_j after exact coefficient combination. Count nonnegativity as
a row, not as free storage. Source predicates, readouts and variable bounds
that are identical in both arms are shared comparison costs, not omitted
from a future physical measurement.

Original lower rows use sum_i s_i+2m nonzeros. Original upper rows, unchanged
here, use sum_i s_i+4m nonzeros when their bound coefficients are nonzero.
Both formulations have m output continuous coordinates, m original bits and
4m gate rows. In the forest variant the LOWER-nnz change is exactly

```text
number_of_nonnegative_edges
  + sum_(i -> h in deficit_forest) [1+s_ih-s_i].
```

For a chain this becomes sum_(i=2..m)[s_(i-1,i)-s_i]+2(m-1). With generic
dense source rows of support d and equally dense differences, this is an
INCREASE of 2(m-1), not a saving. Sparse differences can save nnz, but their
actual occurrence has not been measured. A nonnegative edge alone costs one
more nonzero than its original anchor.

In the displayed two-layer, two-input example, both stages have old 8 rows/
20 predicate nnz and new 8 rows/21 nnz. Across those four gates the comparison
is 16 rows/40 nnz versus 16 rows/42 nnz, with the same four output auxiliaries
and four original bits. Certificates, metadata, RHS and source storage are
additional. These are hand-derived counts, not host/GPU memory measurements.

Changing coordinates to output differences does not generically remove
continuous factors. It can densify suffix-sum readouts and upper predicates.
D002's retained-input/output graph rank applies when its crossing-hyperplane
premises hold. Full guards, terminal lowering, witness decoding and any EQ-only
backend's bounded slack factors remain payable.

## Rejected seven-row aggregate

For a strict ordered pair with original bounds L_f,U_f,L_g,U_g, retain the
four strengthened lowers but replace the four uppers by three rows:

```text
q <= U_f beta_f
r <= g-L_g(1-beta_g)
(q-f)+r <= (-L_f)(1-beta_f)+U_g beta_g.
```

This is integer-exact: 00 forces q=r=0; 10 forces q=f,r=0; 11 forces r=g
and then q=f; 01 is impossible under strict order. It nevertheless fails
LP containment. On the source -1<=f<=2,-2<=g<=1,f-g>=1, the tuple

```text
f=1/2, g=-3/5, q=9/10, r=1/10, beta_f=1, beta_g=1/2
```

satisfies all seven rows but violates the old q<=f upper row at beta_f=1.
Adding beta_f>=beta_g does not remove it. This rejects this particular
one-row aggregation, not every possible seven-row formulation. No experiment
or implementation of the rejected encoding was run.

## Novelty and the next definition question

The sector facts come from standard monotone/slope-restricted activation
geometry; see [Fazlyab et al., section III-C3](https://arxiv.org/pdf/1903.01287).
Neural phase dependencies and dependency cuts also have established prior
art; [Botoeva et al., AAAI 2020, section 3](https://www.doc.ic.ac.uk/~alessio/papers/20/aaai20-BKKLM.pdf)
uses them inside a branch-and-bound workflow, which is NOT adopted here.
This note has not established that the exact forest packaging is a new
literature result. Elementary cone duality is not presented as new either.

The decisive comparator is ordinary HZ with the SAME certified sectors and
the SAME deletion of implied lower rows. Its terminal relation and symbolic
cost match. Therefore this construction is retained as a phase-safe,
fixed-row relational component, not promoted as the requested Neural-HZ
definition breakthrough. It can strengthen a later candidate's fair baseline.

The [common-source comparison](COMMON_SOURCE_COMPARISON.md) explains why a
known joint lift can be stronger but pays for source copies and explicitly
represents phase regions. Neither simply adding that lift nor accelerating
its construction on GPU closes the definition gap. The open research task is
a source/phase interface with a genuinely useful composition or elimination
theorem, without source-copy explosion, phase-region expansion or merely
renaming the original guarded graph. No general impossibility is claimed.
