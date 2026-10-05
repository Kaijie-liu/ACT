# D007: conservation-coupled Neural-HZ candidate

2026-09-28; branch `redu-hz`; commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper mathematics and primary-source review only. No implementation, numerical
imports, tests, model runs or solver calls. Baseline remains1870/2413
(1063CERT+807validatedADV); separate E0 remains CIFAR10025/TinyImageNet36.
The active definition-first Goal, all original resource/replay gates, default-off
rule, no-rescue boundary and read-only historical custody remain unchanged.

Previous turn classification: PROGRESS. D006 established the fiber structure
and oracle obstruction of the complete guard quotient, changing the next
action from canonical quotient engineering to a different domain hypothesis.

## 1. Hypothesis, not a declaration of achieved novelty

Treat a collection of phase-dependent generators and its shared affine
conservation relations as a semantic bundle. Derive a paid, compositional query
envelope from those relations, while retaining the exact nonconvex bundle.
The intended contribution is not changing ReLU into absolute-value notation:
that is established abs-normal-form machinery. Nor is it merely adding arbitrary
solver cuts. The research question is whether a forward, proof-producing bundle
calculus can exploit ordinary affine-expansion/mixed-residual structure at a
cost and precision unavailable from independently analyzed generators.

A positive paper result below gives a complete continuous hull for one
unbounded conservation relation, with an explicit known edge-cone connection.
It is NOT a new theorem of convex geometry, an ideal binary formulation,
complete bounded-network hull, or proven PLDI-level contribution. Discovery,
composition, actual prevalence and end-to-end performance remain open.

## 2. Candidate elements and exact concretization

An element D consists of:

- one owned original HZ frame: bounded continuous xi, original binary identities
  b, original EQ/LE predicates P0, and original-input affine decoder iota;
- a topologically ordered collection of bundles, with preactivations f affine
  in that SAME frame and earlier bundle magnitudes;
- one explicit original ReLU bit beta_i per activation and its magnitude m_i;
- further EQ/LE predicates and affine visible-output decoder o;
- a finite ledger C of certified same-frame relations or bounded defects.

The defining nonconvex factor relation for each activation is

    beta_i in {0,1},  m_i = (2 beta_i - 1) f_i,  m_i >= 0,
    r_i = (f_i + m_i)/2.

Thus m_i=|f_i| and r_i=ReLU(f_i). Both beta values remain legal when f_i=0.
Do not replace this factor relation by its convex envelope. All original
continuous factors, free bits, predicates and shared identities are retained.
Magnitudes are dependent semantic coordinates, not independent box errors.

Strong concretization is the relation

    Gamma(D) = {(iota(xi,b), b, beta, o(xi,b,m)) :
                 P0 and all retained predicates and bundle relations hold}.

Only internal continuous magnitudes are existentially hidden in this display;
original bits are not projected away. Semantic order on a common frame is
Gamma-inclusion. Incompatible frames may not be silently joined or compared.
No general join, widening or efficient semantic-inclusion decision is claimed.

Original HZ embeds with no bundles: retain its generators, predicates and input
decoder, recoding signed bits as2b-1 only if necessary. Affine/Conv composes
visible affine decoders. ReLU adds the exact bundle relation above. Shared
Add/Concat unifies the common frame/owned ancestors and combines decoders;
it must not independently duplicate their latent assignments. Extra predicates
restrict the same relation. Induction proves exact feedforward semantics and
input reconstruction for these operators. Unsupported operations remain out
of scope and fail closed.

This exact grammar alone is closely related to known abs-normal forms and the
D003 reference. It is not the claimed innovation. The extra candidate invariant
is a certified, finite conservation-envelope closure of each admitted bundle.

An initial phase-only attempt was also checked. For sum lambda_i f_i=c with
lambda_i>0, c>0 implies sum beta_i>=1; c<0 implies sum beta_i<=k-1. At c=0
no phase-only exclusion follows: f=0 admits every bit pattern. Signed
coefficients can be oriented by aliasing beta as1-beta on negative terms,
without introducing or deleting any bit. For example, mixed rows
f1=x1+x2, f2=x1-2x2, f3=1-2x1+x2 have sum1. On[-1,1]^2 the old four-row LP
admits x=0,r=(0,0,1),beta=(0,0,1/4), excluded by the valid phase clause.
That is a useful dependency cut, not by itself a new domain. The magnitude
proposal below also works for homogeneous relations where phase clauses fail.

## 3. Query invariant: magnitude conservation

For a certified same-frame identity sum_i a_i f_i=0, with nonzero coefficients
on its k-member support, impose

    |a_i| m_i <= sum_{j != i} |a_j| m_j,  for each i.                 (C)

Triangle inequality proves validity for every exact point, regardless of other
predicates, bounds, phases or residual consumers. Thus adding these predicates
preserves Gamma exactly, including every original zero-phase witness.

For a bounded ordinary terminal four-row encoding, retain ALL old rows and
integral bits, substitute m_i=2r_i-f_i into(C), and add the resulting linear
predicates. If R_old denotes its existing LP query envelope, then

    Gamma projected to query coordinates subset R_new subset R_old.

This is a mathematical envelope comparison, not a change to the permitted
terminal solver boundary or a binary relaxation of the exact domain. It gives
no wall-time or finite-budget solved-count guarantee. In particular it does
not bypass the every-old-solve/full-replay gates.

For a FIXED finite ledger C, closure means checking each certificate, emitting
every declared member row and anchor, and exact canonical deduplication of
those rows. It is idempotent, Gamma-preserving and monotone in admitted valid
certificates. This is not closure under all possible circuits or all semantic
consequences. Incomplete source/bound/bit-length evidence cannot authorize a
row. No fall-through positive verdict is allowed on a failed check.

## 4. Complete single-relation theorem

For the FULL UNBOUNDED hyperplane a^T f=0, the convex hull of

    {(f,r): a^T f=0, r=ReLU(f)}

is exactly

    a^T f=0, r>=0, r>=f, and(C) with m=2r-f.

Proof. Let g_i=a_i f_i, t_i=|a_i|(2r_i-f_i),
u=(t+g)/2 and v=(t-g)/2. The lower ReLU inequalities are equivalent to u,v>=0;
the conservation relation gives sum u=sum v=T. At an exact graph point,
u_i v_i=0. Its cone is generated by the off-diagonal rays

    (u,v)=(e_i,e_j), i != j.

A nonnegative off-diagonal transport matrix with row margins u and column
margins v exists iff u_i+v_i<=T for each i. Necessity follows because row-i and
column-i edges are disjoint. For sufficiency, the neighbor set of a single row
i omits only column i, while any set of at least two rows reaches all columns.
The flow-cut/Hall conditions therefore reduce to precisely these inequalities
and equality of totals. These are(C), since sum t=2T.

If T>0, transport entries pi_ij/T form a finite convex combination of exact
graph points g=T(e_i-e_j), with corresponding u=T e_i and v=T e_j. If T=0,
the point is zero. Inverting the coordinate change proves finite convex-hull
equality, not merely equality after closure.

This is the crown-graph edge cone K_{k,k} minus its matching. The edge-cone
halfspace description is classical; see Corollary3.8 of
[Valencia--Villarreal](https://arxiv.org/pdf/math/0506281).
Our specialization and its neural use are the object of study, not a claim of
inventing that cone. The hull theorem projects away beta: it is NOT an ideal
description of the extended (f,r,beta) hull.

## 5. Ordinary mixed/residual paper control

Let x in[-1,1]^2 and

    f1=x1+x2/2, f2=-x1/2+x2, f3=f1+f2=x1/2+3x2/2,
    r_i=ReLU(f_i), y=r1-r2-r3+f2.

All three rows are mixed and distinct, with no duplicate/opposite pair.
Bounds are[-3/2,3/2],[-3/2,3/2],[-2,2]. The relation is f1+f2-f3=0.
The bundle rows simplify to

    r3 <= r1+r2,
    r1 <= r2+r3-f2,
    r2 <= r1+r3-f1.

At x=0, the old four-row LP admits r=(1/2,0,0), beta=(1/2,0,0), but the second
row excludes it. The same tuple survives even the intersection of the THREE
INDIVIDUAL ideal neuron hulls over the shared input box: for neuron1 take the
midpoint of its exact points at x=(1,0) and x=(-1,0); for neurons2/3 use their
exact zero-input, zero-output, beta=0 point. The means share x=0 but their
independent mixtures need not describe one joint network state.

The actual output satisfies y<=0 and attains0 at x=0. The new linear row proves
this bound, whereas the displayed old query point gives y=1/2. This is an
analytic synthetic precision separation, NOT a benchmark CERT or ADV.
All eight INTEGRAL bit choices remain legal at zero with actual r=0.

Native direct inequality bill:5 continuous variables,3 bits,12 gate rows and
30 matrix nnz before;5 continuous,3 bits,15 rows and43 nnz after. Original input
bounds, RHS, certificates, metadata, source and witness cost are additional.
No count is a measured runtime or total physical-storage result.

## 6. Bounded defects: sound use without exact nullspace arithmetic

Exact rational nullspace bases may have excessive coefficient bit lengths.
Do not treat a numerically proposed dependence as exact. For same-frame
f_i(x)=w_i x+b_i and sound coordinate bounds x in[L,U], a proposed dyadic a,c
can instead be checked through

    d=sum_i a_i w_i, d0=sum_i a_i b_i-c,
    emin=d0+sum_j min(d_j L_j,d_j U_j),
    emax=d0+sum_j max(d_j L_j,d_j U_j),
    epsilon=max(|emin|,|emax|).

Exact rational or rigorously outward computations establish
|a^T f-c|<=epsilon. Then every exact point satisfies

    |a_i|m_i <= sum_{j != i}|a_j|m_j + |c|+epsilon,
    sum_i |a_i|m_i >= |c|-epsilon.

Proof: isolate a_i f_i and apply the triangle inequality; the last row is the
reverse triangle bound on a^T f. These are valid additional constraints only:
do not change the original weights, exact equations, phases or witnesses.
Checking a k-row/d-port proposal costs O(kd+d) arithmetic operations, plus
bit-length, source-identity, bound and retained-evidence checks. A large defect
may render every row useless. Proposal discovery itself is not free, and no
numerical linear-algebra proposal method is qualified in this checkpoint.
There is no LP dual, infeasibility oracle, backward rescue or phase search here.

## 7. Full cost, common structures and strong comparators

For k members, direct(C) has k rows but up to2k^2 local(f,r) coefficient
occurrences. Flattening f over d source ports gives at most k(k+d) matrix nnz,
before extra predicates, bounds or certificates. Affine/defect anchor adds a
row. Counting just rows as O(k)-size would be misleading.

An alternative stores one paid hub T=(sum |a_i|m_i)/2, then k inequalities
|a_i|m_i<=T for the homogeneous case. With m aliases=2r-f, this uses one
additional continuous scalar, one equality and k inequalities, with O(k) local
coefficient occurrences (and source expansion still payable). If the backend
requires equality-only HZ, EACH additional inequality needs its own bounded
slack and normalization proof; do not hide those continuous factors.

Affine channel expansion is a plausible ordinary target: m output rows over a
rank-r source admit fundamental affine dependencies of support at most r+1. This is
an algebraic availability observation, not evidence of useful tightness on a
saved network. For dense64->128 with rank(W)=64,64 supports of65 can add4160
MEMBER rows and up to536640 flattened nnz, versus512 standard gate rows/17152
nnz. With nonzero biases the relations need not be homogeneous:64 affine
anchor rows add up to8256 nnz, for4224 extra rows/up to544896 nnz. These are
symbolic upper counts before other costs, not observed ranks or measurements.
This example warns AGAINST indiscriminate all-circuit construction.
Spatial replication, proposal discovery, certificate bit growth, bounds,
lowering, slack variables, solver overhead and four-concurrent memory all count.

### Saved structural evidence: wide convolution is actually present

Read-only jq inspection of two frozen graph-descriptor results found the same
ordinary structural class: the initial convolution has64 output filters over
3x3x3=27 patch coordinates. CIFAR's descriptor has Conv2->ReLU3. Tiny's has
Conv2, channel Scale/Bias descriptors and ReLU5. Thus a faithful first-filter
matrix is64-by27 and has rank at most27, so affine dependencies exist without
requiring duplicate or specially chosen trained filters. Full row rank27 is
NOT established by shapes; actual coefficients, defects and usefulness remain
unmeasured. These saved records are NOT a target qualification or an instance
selection menu; the rule is the structural filter-bank relation.

Sources, relative to experiments/neural_hz_20260831:

- results/trial6_cnn_census__cifar100_2024__iid166__v1.json,
  SHA256 03aff8b0cb3aaa80a15ff27774cb74dfef599f180849878a098c807b30f66ce4;
- results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json,
  SHA256 f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274.

Both are historical incomplete-HZ descriptors, not completed exact network
states. In particular Tiny's saved Scale/Bias edges have the previously
diagnosed graph-faithfulness defect. Do NOT use those edges to certify an
effective filter or reuse historical bound hashes as universal bounds. A new
source-faithful original-network read/preflight is required before numerical use.

**Shared-filter transfer.** If f_t=F p_t+b is the faithfully decoded effective
filter bank at spatial position t, a single checked coefficient residual
d=a^T F and bias residual d0=a^T b-c applies at every position. A sound patch
box produces its own epsilon_t by the same interval formula. Thus proposal and
coefficient checking can be shared once; patch-bound checks, emitted predicates,
query matrices and all evidence remain payable at every position. Padding,
stride and overlap change p_t, not this algebra. The p_t are references into
the SAME input frame, never independently chosen patches. This preserves
spatial correlations and allows the structural rule to apply uniformly.

This gives a concrete next research target: bounded-budget conservation
bundles of a wide convolutional filter bank, including faithful channel-affine
operations. It does not justify constructing every fundamental circuit: with
rank27 there can be37 supports of28 and large replicated matrix overhead.

Known nonnegative inputs cover a common marginal, but not a novelty claim.
For g=b+sum w_i h_i with h_i>=0, the same triangle rule yields

    ReLU(g) <= sum_{w_i>0} w_i h_i + max(b,0).

This is dominated by the known indicator inequality
r<=sum_{w_i>0}w_i h_i+b*beta on zero-lower-bound input boxes, a fixed instance
of Proposition1(6b) in
[Anderson et al.](https://arxiv.org/html/1811.08359v2).
The fixed known facet belongs in the comparator; rediscovering it is not a
joint-bundle contribution. Ordinary HZ/MILP supplied the SAME certified bundle
rows has the same terminal relation. Any actual advantage must include their
discovery/propagation/cost, not falsely deny that comparator these rows.

## 8. Limits: intersection is sound, not globally complete

Bounds: intersecting the unbounded hull with a box need not give the bounded
graph hull. Affine anchors likewise require caution. For f1+f2=1, the fixed
anchor triangle system admits(f1,f2,r1,r2)=(0,1,10,11), outside the finite convex
hull of the actual graph. The saturated row forces every contributing exact
point to have f1<=0 and r1=0, contradicting r1=10. A limit of convex combinations
can approach it. Only sound envelopes, not affine/boxed hull exactness, are
claimed here.

Multiple relations: even ALL individual circuit hull constraints need not
describe the joint continuous hull. Take five potentials x_i and the ten
preactivations f_ij=x_i-x_j, i<j. At x=0 choose magnitudes m_ij=2 within the
partition {1,2,3}|{4,5} and1 across; let r=m/2. These distances satisfy every
triangle, hence every cycle inequality. Minimal affine dependencies are graph
cycles. A conformal cycle decomposition extends validity to inequalities
from arbitrary circulation relations as well.

Yet every actual magnitude vector, and every convex combination of them,
satisfies the pentagonal inequality

    sum_{within {1,2,3}} m_ij + m_45 <= sum_{across the partition} m_ij.

To prove validity, decompose real line distances into nonnegative threshold
cut metrics. For b=(1,1,1,-1,-1), a cut's hypermetric sum is t(1-t)<=0, where
t=sum_{i in cut}b_i is integer. The proposed m has8 on the left and6 on the
right. This is the classical metric-versus-cut-cone distinction; see section3.1
and equation(3.4) of
[Deza--Laurent](https://pure.uvt.nl/ws/files/1215796/Facets.pdf).
With ALL five x_i in[-1,1], beta_ij=1/2 gives a valid ordinary four-row LP
witness too (each f_ij bound is[-2,2]). Do not additionally anchor x5=0 while
retaining those tight bounds without rechecking them.

This is a scope counterexample, not a new target family or an invitation to
pursue exotic cases. Original integral bundle semantics remains exact because
no bits/guards are removed; incomplete query closure is explicitly permitted.

## 9. Prior art and research disposition

[Abs-normal forms](https://arxiv.org/pdf/1701.00753), section2, already express
piecewise-linear maps using shared affine switching variables and magnitudes.
[Exact ReLU HZ](https://arxiv.org/pdf/2304.02755), sectionsII-B/III, already
preserves shared factors and exact binary graph semantics. Neither notation
nor exactness alone distinguishes D007.

[PRIMA](https://arxiv.org/pdf/2103.03638), sections3/5, already generates
multi-neuron relational constraints, using octahedral projections and
Split-Bound-Lift. The paper's group construction includes activation-region
partitioning; D007's single-relation formula uses no such construction.
This is a restricted analytic comparison, NOT a blanket complexity dominance
claim over PRIMA's hull approximation algorithms or all group methods.
[GCP-CROWN](https://arxiv.org/pdf/2208.05740), section3, already accepts general
indicator/value cuts. Its dual propagation/MIP generation is not adopted.

Disposition: retain the conservation-bundle hypothesis for further research,
with a proved restricted transfer and synthetic precision separation. Do NOT
declare an innovative domain achieved or start broad replay. The next concrete
question is a bounded forward certificate-selection/normalization rule on
ordinary affine expansion, with measured full costs against standard HZ,
known single-neuron facets and ordinary HZ with identical bundle predicates.
Qualification requires default-off mathematical tests first, then faithful
same-structure source evidence; no source/label/solver-status menu is allowed.
The proposal method, paid limits and native lowering must be preregistered
before execution. All historical files remain read-only. Goal remains ACTIVE.
