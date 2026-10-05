# Joint witnesses and the cost of exact neural projection

D155 is paper research, not an executable or enabled Neural-HZ candidate.
It repairs D154's loss of shared realization by exact joint projection, then
measures the resulting cost. A narrow interval-fiber transfer can save rows,
but a new applicability argument prevents treating its positive-width example
as evidence about an exact deterministic CNN. No formal score changes.

## Protected semantics and the question being tested

Keep the same original inputs, continuous source coordinates, all original
binary identities, EQ/LE predicates, shared frame and input decoder. Fractional
bits below refer only to an LP comparison; the represented domain keeps its
bits integer and preserves both legal labels at zero.

An interface may hide a continuous amplitude only after accounting for every
consumer, predicate, property and decoder use. A relation on retained coordinates
is exact when it is the existential projection of the original SAME-source,
SAME-bit relation, not an intersection of projections with independent witnesses.
The complete source relation remains nonconvex through its retained bits.
Using a convex fixed-LP slice for a proof does not replace the domain with CZ
or zonotopes. Empty projection metadata leaves the original HZ representation.

This is a relational interface specification, not a claim of a new set class,
best abstraction, complete lattice or useful global query algorithm. Ordinary
HZ with the same exact projection can express it. D004 already established
the distinction between integer aggregate equivalence and exact LP projection.

## Exact joint projection of the four-gate shared-phase control

Use D154's four gates and complete mixed consumer C=(2,-1,1,3). Retain their
source forms f_i and bounds l_i<0<u_i. Define

```text
L_i=max(0,f_i),
U_i=min(u_i*beta_i, f_i-l_i*(1-beta_i)),
a=t1+f1=q1+q3,
b=t2+f2=q2+q4,
z=4*q4-q3.
```

Unlike D154's independently lowered max and product graphs, these are the
LINEAR quotient coordinates of the entire original four-gate LP. Let h=q4.
The inverse fiber is

```text
q1=a+z-4*h, q2=b-h, q3=4*h-z, q4=h,

h_lower=max(L4, (z+L3)/4, b-U2, (a+z-U1)/4),
h_upper=min(U4, (z+U3)/4, b-L2, (a+z-L1)/4).
```

The exact projected relation is h_lower<=h_upper. A witness h=h_lower
simultaneously reconstructs ALL four old amplitudes. This holds for fractional
bits as well as integer bits. No source or phase search is performed.

Expanding the sixteen comparisons before expanding the L/U expressions gives
four own-interval feasibility conditions L_i<=U_i and six two-sided strips:

| Readout | Lower bound | Upper bound |
| --- | --- | --- |
| a | L1+L3 | U1+U3 |
| b | L2+L4 | U2+U4 |
| a+z | L1+4L4 | U1+4U4 |
| 4b-z | 4L2+L3 | 4U2+U3 |
| z | 4L4-U3 | 4U4-L3 |
| a+z-4b | L1-4U2 | U1-4L2 |

The four own-interval conditions are precisely the eight original relaxed
sign guards l_i*(1-beta_i)<=f_i<=u_i*beta_i. They cannot be omitted because
another gate's interval slack might otherwise hide an infeasible gate.

The complete consumer becomes

```text
Y=2*f1-f2+2*t1-t2+z.
```

Every additional consumer must factor through the same retained packet or be
included in the projection. In particular an identity use of the whole q
vector is not covered by merely listing this single Y.

## Linear row and coefficient costs

If f1>=f2>=f3>=f4 holds throughout the COMPARED LP source, the first four
positive-sum lower bounds each have three rather than four affine pieces.
For example L1+4L4=max(0,f1,f1+4f4). Each corresponding upper bound has
four pieces. The final two difference strips each have four lower and four
upper pieces.

Using physical f_i columns, inlining a=t1+f1 and b=t2+f2, and counting
nonnegative rows, the direct formulation has this complete predicate bill:

| Row family | Rows | Predicate nnz |
| --- | ---: | ---: |
| a strip | 7 | 21 |
| b strip | 7 | 21 |
| a+z strip | 7 | 28 |
| 4b-z strip | 7 | 28 |
| z strip | 8 | 24 |
| a+z-4b strip | 8 | 40 |
| Own-interval guards | 8 | 16 |
| Total | 52 | 178 |

The original four gates have four amplitudes, sixteen rows and 32 nnz under
the same convention. The quotient has three amplitudes but 52 rows/178 nnz.
Source construction, bounds, readouts, RHS, bit bounds, proof records,
decoder, possible inequality-slack factors and coexistence memory are further
costs on BOTH sides, not free work. No physical memory or timing was measured.

An order certified only on integer source states is not enough to delete
those four LP pieces. Without LP-valid order, restore

```text
a>=f3, b>=f4, a+z>=4*f4, 4*b-z>=f3.
```

They add respectively 3,3,4,4 nnz, giving 56 rows/192 nnz. Alternatively give
both comparison arms the same valid explicit order rows and charge them:
three adjacent orders cost three rows/six nnz with physical f columns.
D154's ordinary source-box control already implies the order on its LP source.

These counts describe a sufficient direct formulation, not a proof of minimum
global row count. They do not prohibit a better specialized joint lowering.
Reintroducing h makes a compact extended formulation, but restores the fourth
continuous amplitude. Storing only the max/min expression makes conditional
membership cheap; it does not solve global optimization over free source/bits
with the original LP/MILP terminal for free. Batched pointwise max/min arithmetic
is likewise not a GPU verification result.

## A fixed fiber has twelve necessary facets

Take the D154 control

```text
f_i=x+v_i/8-theta_i, x in [-2,2], v_i in [-1,1],
theta=(-3/4,-1/4,1/4,3/4),
x=v_i=0, all beta_i=1/2.
```

The four old amplitude intervals are

```text
[12/16,23/16], [4/16,19/16], [0,15/16], [0,11/16].
```

They all have positive width. In (a,b,z) coordinates the four generating
directions are (1,0,0), (0,1,0), (1,0,-1), (0,1,4). Consider the six normals

```text
(1,0,0), (0,1,0), (0,0,1),
(1,0,1), (0,4,-1), (1,-4,1).
```

Each is orthogonal to exactly two independent generating directions, and
nonorthogonal to the other two. Maximizing its linear functional fixes those
other two interval endpoints; the orthogonal pair ranges over a nondegenerate
parallelogram face. The negative normal gives another face. These six distinct
normal pairs therefore supply twelve necessary facets of the three-dimensional
image, compared with eight facets of the original fixed four-dimensional box.

This explains some real projection complexity, rather than just an inefficient
expansion. It is ONLY a nonextended, fixed-fiber fact. It does not imply 52
necessary rows globally, does not exclude a sixteen-row parametric formulation,
and is not a lower bound for all extended/domain representations.

## Stronger order comparator and exact downstream composition

D018's order-enhanced chain is not the independent four-gate LP above. Put
Delta_ij=f_i-f_j, A_i=u_i*beta_i and B_i=f_i-l_i*(1-beta_i). Substitution of
the same inverse fiber gives these eight affine lower endpoints for h:

```text
0,
(a+z-b-Delta12)/3, (b+z-Delta23)/5, z/3,
(a+z-A1)/4, (a+z-B1)/4, b-A2, b-B2.
```

The eight affine upper endpoints are

```text
(a+z-f1)/4, (a+z-b)/3, (b+z)/5, (z+Delta34)/3,
(z+A3)/4, (z+B3)/4, A4, B4.
```

All lower<=all upper gives a direct 64-row exact projection of THAT comparator.
Its minimum size and coefficient bill have not been established; do not attach
the preceding 52/178 count to it.

For either exact projection, downstream Affine/Conv, aligned Add/Concat and
ordinary ReLU predicates compose exactly when they use only retained readouts.
In logic, projecting an internal h commutes with conjoining a downstream
relation that does not mention h. The same h witness works for the whole
upstream block. Original-input decoding is unchanged; integer reconstruction
gives the unique original q at that input. This preserves the matched LP's
precision, rather than creating new precision by projection alone.

## A general common-fiber ReLU transfer

Suppose a non-input amplitude eta has no omitted consumer or decoder use and
its entire parent relation is

```text
S(theta), L_i(theta)<=eta<=U_k(theta), i=1..p, k=1..n.
```

All endpoints are affine in the retained source and original bits. For s
actual child gates g_j=a_j*eta+b_j(theta), a_j nonzero, keep each original
bit gamma_j and the same sound crossing bounds ell_j<=0<=u_j. The old rows
are r_j>=0, r_j>=g_j, r_j<=u_j*gamma_j,
r_j<=g_j-ell_j*(1-gamma_j). Define

```text
a_j>0:
  A_j=(r_j-b_j+ell_j*(1-gamma_j))/a_j,
  B_j=(r_j-b_j)/a_j;
a_j<0:
  A_j=(r_j-b_j)/a_j,
  B_j=(r_j-b_j+ell_j*(1-gamma_j))/a_j.
```

The two rows involving eta are exactly A_j<=eta<=B_j. Let the complete lower
list be {L_i} union {A_j}, and the upper list {U_k} union {B_j}. The precise
projected child relation keeps S, 0<=r_j<=u_j*gamma_j and EVERY lower<=EVERY
upper. Reconstruct eta as the maximum of the complete lower list. This proves
integer and LP equivalence simultaneously, including original zero labels.
Gates with a_j=0 do not participate in elimination and retain their own rows.

The old local row count is p+n+4s. The raw new count is (p+s)*(n+s)+2s;
each A_j<=B_j is a tautology, leaving the sufficient count

```text
pn+s*(p+n)+s*s+s.
```

Cross-child comparisons cannot be deleted. For p=n=s=1 this is five versus
six rows. If the parent L<=U is already implied by S, it need not be repeated,
giving four versus six. Endpoint coefficient fill and all shared-system costs
remain chargeable. This is classical Fourier-Motzkin elimination in a useful
shared-fiber form, not a newly invented elimination principle.

## Positive abstract-parent control and its precise limitation

Let x,y in [-1,1], retain original bit beta, and take the parent interval

```text
L=x/4-y/8+beta/16, L<=eta<=L+1/4.
g=2*eta-x/3+3*y/4-1/4, q=ReLU(g), original bit gamma.
```

Valid tight bounds are ell=-11/12 and u=25/24. Put

```text
Hminus=x/6+y/2+beta/8-1/4,
Hplus =x/6+y/2+beta/8+1/4.
```

The exact projected LP/native relation is only

```text
q>=0, q<=(25/24)*gamma,
q>=Hminus, q<=Hplus+(11/12)*(1-gamma).
```

Counting affine coefficients directly in x,y,eta,q and the original bits,
the old fiber plus gate has two amplitudes/six rows/20 nnz; the new relation
has one amplitude/four rows/12 nnz. L<=L+1/4 is a tautology, not an unpaid
feasibility test. Original source, bits and decoder stay. Any next gate using
only retained values, e.g. ReLU(q-y/3+x/5-1/8), composes exactly.

However this is a control for a GENERAL HZ or an already abstracted remainder.
It is not demonstrated to arise as the exact activation graph of a CNN. The
following functionality argument explains why that distinction is necessary.

If theta uniquely fixes the original concrete input and eta is a real
intermediate value in a deterministic network, an EXACT graph representation
has exactly one eta at each feasible integer theta. Therefore any exact full
interval-fiber description must obey

```text
max_i L_i(theta) = min_k U_k(theta)
```

at every such state. Otherwise two distinct interval points would describe
two different intermediate values for the same input. This is incompatible
with deterministic evaluation. Legal zero-phase ambiguity does not create
ambiguity in the activation's value.

For p=n=1, this reduces to eta=L=U on the integer relation, returning an affine
identity-elimination situation. Equality only on integer states does NOT imply
that the source LP also has zero width. Ordinary exact LP substitution needs
the equality on the compared relaxed relation as well; an integer-valid
identity can instead strengthen that relaxation, with its other feasibility
constraints still accounted for.

The argument does not apply to a nondeterministic abstract remainder, a free
coordinate gauge rather than an actual neuron value, or retained keys that do
not fix the input. Those cases require their own provenance and decoder proof.
It does not ban useful approximate Neural-HZ domains. It DOES prevent counting
the positive-width toy parent as a discovered exact-CNN structure or launching
a model census under that unsupported premise.

## Independent child projections break the common witness

Use the preceding parent and the interior source x=y=0, beta=0, so eta is in
[0,1/4]. Take two biased mixed-source gates

```text
g1= eta+x/5-y/7+1/16,
g2=-eta-x/6+y/5+3/16.
```

Both have reliable bounds [-2,2] on the whole parent. Set their original
bits gamma1=gamma2=1. Separately, r1=1/4 has eta1=3/16 and r2=1/8 has
eta2=1/16. Both witnesses are strictly inside the SAME parent fiber and both
gates are strictly active. Yet no single eta realizes the pair. The joint
projection includes A1<=B2, here 3/16<=1/16, and rejects it.

This is a same-source, non-boundary witness counterexample, not a concrete
network ADV or formal regression. It demonstrates why all consumers must
share a fiber and why the quadratic cross-comparison term in the direct
formula cannot simply be erased.

## Research decision and qualification boundaries

The direct joint projection repairs D154's precise semantic gap. For its dense
four-gate control the paid native formulation is larger. The cheap one-child
fiber lemma has a genuine positive count, but lacks a new domain principle
and an exact deterministic-CNN applicability witness. Neither is selected for
implementation, metadata scanning or a new test framework in this turn.

Exact LP preservation is a sufficient comparison property being investigated,
NOT an extra hard permission gate added to the user's goal. A sound candidate
with a different LP remains eligible for research if its definition, realistic
structure and complete cost justify it. Promotion still requires the unchanged
full replay preserving every one of 1870 and all family counts, plus validated
new solves; no paper LP inclusion theorem replaces that empirical requirement.
Nothing here relaxes testing, safety, source integrity or default-off rules.

The next definition must supply a nontrivial joint neural transfer on actual
source-linked states, not rely on free hidden amplitude width that an exact
network does not have. New exact coordinates, useful controlled approximation
and other structures remain possible; this paper does not establish a general
impossibility. GPU set queries, smooth/Transformer mechanisms, real-family
coverage and the full goal all remain unachieved.
