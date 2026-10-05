# D004 results — original-subset obstruction and aggregate tradeoff

2026-09-28; branch `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Paper proofs, independent reviews and saved-JSON topology only. No new candidate
code/import, numerical test, model/HZ decode, property solve or benchmark run.
The earlier sealed `D004_QUESTION.md` is preserved unchanged.

## 1. Original-gate subset interfaces fail generically

Let U be a nonempty open box in R^p; f_j=w_j*x+b_j with w_j nonzero;
r_j=ReLU(f_j). Suppose their geometric zero hyperplanes are pairwise distinct
and each cuts U. Let h=C*r+D*x+e include ALL specified affine consumers.
Then a constant-affine identity

    h = A*r_K + B*x+d  on U

using only a subset K of the original ReLU values exists **iff** every column
C[:,j] with j outside K is zero.

Proof: choose a point on the deleted j's hyperplane in U, away from all other
hyperplanes. Such a point exists because finitely many proper intersections
cannot cover its relatively open portion. On a short transverse line choose
the parameter so f_j=t. All other gates are locally affine. The derivative
jump of the left side is C[:,j]; the right side has no jump. Hence C[:,j]=0.
Sufficiency follows by dropping exactly those zero columns. Apply componentwise
to vector consumers. Residual D*x terms and a small rank(C) do not remove this
obstruction. This is a proof, not a phase-enumeration algorithm.

For coincident hyperplanes write f_j=a_j*f_G. Then

    r_j=|a_j|*ReLU(f_G)+min(a_j,0)*f_G.

A wholly deleted group requires sum_j |a_j| C[:,j]=0; one retained member can
supply its hinge. This is the familiar signed/proportional exception, not a
new population of generic mixing reductions.

Explicit predicates restrict the domain: the argument requires an isolated
two-sided crossing inside the actual feasible domain, or its relative interior
after affine-hull parameterization. Ambient distinctness alone is insufficient.
Example: f1=x1+x2 and f2=x1-x2 become identical under predicate x2=0. Fixed
original-free-bit fibers are treated separately only in this proof; fixing all
ReLU bits erases crossings and cannot establish one phase-independent interface.
For deep layers use a qualifying open prefix cell in original-input space,
not an unjustified full feature box. Additive phase-bit readout terms do not
help in the unconstrained full-phase relation: at a gate's zero, both bit
choices have identical values, forcing that decoder column to vanish.

Disposition: close **generic deletion of live original-gate subsets** as the
main dense-mixing route. This does not rule out other coordinates, nonlinear
decoders, contextual equivalence or domain restrictions.

## 2. Aggregate-only linear lowering can need exponentially many rows

Consider the ordinary mixed family f=W*x, with W invertible m by m,
x in [-1,1]^m, and z=sum_j ReLU(f_j). The consumer rank is only one.
Every sign pattern occurs in the box interior: choose a sufficiently small
signed f and use x=W^-1*f. The graph's open-cell slopes beta^T W are all distinct.

Any finite exact linear formulation with continuous variables ONLY (x,z) and
the retained bits needs at least 2*2^m inequality constraints, INCLUDING
active variable-bound inequalities in this count. Fix an interior point
of a cell and its bits. Exactness requires an active upper row and active lower
row in z; otherwise a small upward/downward z perturbation remains feasible.
An active valid row must annihilate that full-dimensional graph-cell tangent.
Its ratio of x coefficients to the nonzero z coefficient therefore fixes one
cell slope. One upper/lower row cannot serve two distinct slopes. A global
equality involving z cannot fit distinct slopes. This proves the row lower
bound even without requiring an ideal continuous relaxation. If native bounds
are counted separately, z>=0 may supply one of these inequalities; the explicit
matrix-row count can then be 2*2^m-1. The exponential obstruction remains, but
the total-inequality lower bound must not be mislabeled an explicit-row bound.

This is a formulation-counting argument, not an allowed implementation that
enumerates phases. It does not prohibit extended formulations with retained
continuous auxiliaries, or all bounded rank gaps. For rank(C)=m-1, eliminating
one kernel coordinate by ordinary Fourier–Motzkin can use O(m^2) rows. The
statement concerns generic aggregate-only compactness, not every reduction.

Prior work already distinguishes compact extended formulations from large
nonextended formulations. Anderson et al.'s result concerns an **ideal single
ReLU formulation**, not the multi-ReLU retained-phase lower bound above; it is
relevant comparison, not a citation proving our specific derivation.
[Anderson et al.](https://arxiv.org/abs/1811.08359).

## 3. A small exact aggregate is possible, but the full bill matters

For z=c1*r1+c2*r2 with both c_i nonzero, define

    L_i=max(0,f_i), U_i=min(u_i*beta_i, f_i-l_i*(1-beta_i)).

For integral bits and sound l_i<=f_i<=u_i, L_i>=U_i, with equality exactly
when the original gate guard is legal: beta=0 gives U_i=0 and deficit
max(0,f_i); beta=1 gives U_i=f_i and deficit max(0,-f_i). For c_i>0 use
contribution endpoints c_i L_i,c_i U_i; for c_i<0 reverse them. Their deficit
is |c_i|(L_i-U_i). Thus deficits cannot cancel at integral bits.

Four lower pair sums and four upper pair sums yield eight linear inequalities
that preserve the entire continuous-input/integral-bit/z relation, including independent
zero-phase choices. These are algebraic bound combinations, not a performed
phase search. A zero coefficient does NOT preserve that gate's guard implicitly.
Every outside consumer, including later preactivations and EQ/LE predicates,
must factor through this z. Equivalently complete consumer columns are v*c_i.

Take f1=x1+x2, f2=x1-x2, c=(1,1), and bounds [-2,2]. Exact integral rows:

    -z                             <= 0
     x1+x2-z                       <= 0
     x1-x2-z                       <= 0
     2*x1-z                        <= 0
     z-2*beta1-2*beta2              <= 0
     z-x1-x2+2*beta1-2*beta2        <= 2
     z-x1+x2-2*beta1+2*beta2        <= 2
     z-2*x1+2*beta1+2*beta2         <= 4

Simply summing corresponding original gate rows is UNSOUND: x=(0,1/2),
bits=(1,0) satisfies the original guards, but those four aggregate rows admit
z=1 while the actual sum is1/2. This point is not a network ADV.

The eight-row version is **not the exact LP projection**. At x=(1/4,1/4),
bits=(0,1/2), the first interval is inverted [1/2,0], the second is [0,1],
and the aggregate admits z=1/2. The original first gate is infeasible. A
fractional interval's positive width can hide another gate's deficit.
For crossing bounds l<=0<=u, retaining each gate's two own-interval feasibility
rows restores the exact LP projection:

     x1+x2-2*beta1 <= 0;   -x1-x2+2*beta1 <= 2
     x1-x2-2*beta2 <= 0;   -x1+x2+2*beta2 <= 2.

Stable bounds require checking all own-interval conditions, not blindly using
this crossing-only reduction. The distinction was caught and corrected during
paper review, before any implementation or numerical run.

| Full formulation | Continuous variables | Original bits | Rows | Row nnz |
| --- | ---: | ---: | ---: | ---: |
| Original two gates | 4 | 2 | 8 | 20 |
| Integral-phase aggregate | 3 | 2 | 8 | 26 |
| Exact LP projection | 3 | 2 | 12 | 38 |

Each row has one RHS. Original continuous bounds have8 endpoint scalars;
projected bounds have6; both have4 bit-bound endpoints. Use z in [0,4] in the
LP comparison: z<=2 is a valid INTEGER strengthening but changes the original
LP projection, which permits r1=r2=3/2 at x=(1,0), bits=(3/4,3/4). All witness,
source, bounds, projection-certificate and any backend slack costs remain due.

Ordinary inequality-HZ/MILP can perform exactly this projection. It is useful
supporting algebra, not evidence for new-domain novelty or net improvement.
No implementation is selected merely because one continuous variable disappears.

## 4. Saved real-topology evidence and limitations

Both saved JSONs were reauthenticated against D002:

- Tiny trial8: SHA256
  `f1f6abcdc87d96e9d8898d0f667d8087b2a72a9898f44d61b4e123426d45b274`;
  experiment-root `results/trial8_phase_selective__tinyimagenet_2024__iid143__relu63_v1.json`.
  81 descriptors:19Conv,19Scale,19Bias,10ReLU,8Add,6others.
- CIFAR trial6: SHA256
  `03aff8b0cb3aaa80a15ff27774cb74dfef599f180849878a098c807b30f66ce4`;
  experiment-root `results/trial6_cnn_census__cifar100_2024__iid166__v1.json`.
  44 descriptors:20Conv,10ReLU,8Add,6others.

The exact fields are `layer_hz_structure[id].kind`, `.weight_shape` and
`.successors`. Clean tail blocks are Tiny Dense77[200,6272]->ReLU78->
Dense79[200,200], CIFAR Dense40[100,4096]->ReLU41->Dense42[100,100]. Residual
fanout is explicit (CIFAR ReLU3->Conv4 and Add7; Tiny ReLU5->Conv6 and Conv13).
Neither graph has any pooling descriptor: their other nodes are INPUT,
INPUT_SPEC, FLATTEN, two DENSE and ASSERT. Thus they provide no pooling source
for the aggregate pair, though coefficients might contain other proportionality.

Shapes/edges establish neither nonzero columns nor rank, exact proportionality,
attainable isolated crossings or a full-dimensional reachable feature domain.
Both old runs report missing_hz_state/output_hz_exact:false. Historical Tiny
stability totals are not current certified per-neuron bounds. The old Tiny BN
edge defect and clone-only correction described by D002 remain relevant.
No theorem premise was falsely declared verified on these targets.

## 5. Decision and next definition question

D004's exact-preactivation interface condition is too restrictive for generic
mixing, and unrestricted aggregate-only linear lowering has a real row barrier.
The pooling-pair projection is archived as reusable algebra, not adopted as a
presumed CIFAR/Tiny breakthrough or an LP-neutral shortcut.

The next question changes the semantic equivalence at a **ReLU-only consumer**:
may preactivations differ on their strictly negative region while preserving
their rectified value and exact phase/zero relation? This is a contextual
quotient, not equality of every intermediate real value. It requires its own
consumer/predicate and reconstruction theorem, strong prior-art comparator,
ordinary mixed positive control, real-source evidence and full cost analysis.
See the separate D005 draft; no new runtime rule has been authorized/enabled.

Formal1870/2413 and separateE0=61/400 unchanged; no model, solver, test or
background numerical process was started this turn. The research goal is ACTIVE.
