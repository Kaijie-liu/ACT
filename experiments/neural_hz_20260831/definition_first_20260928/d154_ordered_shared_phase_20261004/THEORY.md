# Ordered shared-phase amplitudes: an exact interface, not yet a stronger domain

D154, 2026-10-04 Australia/Sydney. Paper derivation and independent review only.
No executable candidate, numerical experiment, benchmark gain or novelty claim.
The objective remains a powerful nonconvex Neural-HZ defined from HZ semantics,
not a helper or merely cheaper storage. [Research record](RESEARCH_RECORD.md)
contains provenance and the implementation decision.

## Semantic contract

The source is one owned HZ relation S, including its continuous frame, ALL
original signed bits, EQ/LE predicates, shared identities and input decoder.
Write an original bit as beta=(1+bit)/2 only for notation. The represented
semantics keeps beta binary; fractional beta below is solely a terminal-LP
comparison. No bit is removed, pivoted, replaced or cloned.

Let f_1,...,f_m be affine readouts on that same source with certified order
f_1>=...>=f_m and reliable bounds l_i<0<u_i. A strict order is useful in the
ordinary controls, but the exact folding theorem also preserves ALL original
zero labels under weak order. In particular, weak order does NOT justify
imposing beta_1>=...>=beta_m when several f_i are zero.

An ordered-fold element retains S, the original bits and guards, exact shared
amplitudes defined below, and phase-dependent visible readouts. Its strong
concretization is the relation of original inputs, original bits and declared
values satisfying these relations. Original HZ embeds with no folded blocks.
Order is relation inclusion after aligning identities; no best abstraction or
complete lattice is claimed. The retained exact ReLU graphs remain nonconvex.
This is not replacement by a zonotope, CZ or a convex order cone.

Any predicate or consumer of an eliminated intermediate value must be
substituted consistently. No output dimension is called dead merely because
one next layer does not use it. Original-input decoding remains unchanged;
an alleged ADV still requires the original concrete-network validation.

## Exact common-median-phase folding

First take m=2r, pair i with r+i, and use the ORIGINAL central bit rho=beta_r.
Set

```text
t_i=max(0,-f_i,f_(r+i)),     i=1,...,r,
q_L=f_L+(1-rho)*t,
q_R=rho*t.
```

If rho=0, its original guard gives f_r<=0 and hence every right f<=0.
Therefore t_i=max(0,-f_i), the left formula is ReLU(f_i), and right q=0.
If rho=1, f_r>=0 and all left f>=0. Then t_i=ReLU(f_(r+i)), giving the
same exact outputs. At f_r=0 either legal label works. This proof needs
neither runtime phase enumeration nor a new phase variable.

For odd m, pair floor(m/2) left and right gates and retain one unpaired right
amplitude, giving ceil(m/2) amplitudes before terminal lowering. The m=1 case
is the original gate. All detailed cost statements below concern even m.

Thus amplitude width can genuinely change in the exact representation. That
alone does NOT prove a new mathematical principle, a useful terminal LP, or
an inexpensive subsequent ReLU. D014/D137 already supply disjoint-amplitude
identities; this is their ordered, common-original-phase organization.

## A sharp algebraic width fact and its scope

For prefix masks D_s=diag(1_(i<=s)), s=0,...,m,

```text
min_B max_s rank(D_s-B) = ceil(m/2).
```

For a proposed maximum r, the endpoints imply
m=rank(I)<=rank(B)+rank(I-B)<=2r. Conversely choose the median prefix B;
rank(D_s-B)=abs(s-floor(m/2))<=ceil(m/2).

To interpret this as a necessary continuous residual width in
q=B*f+c_beta+E_beta*eta, the source must supply a common full-dimensional f
chart and the relevant endpoint phase cells must have open interiors there.
Finite differences inside a phase imply range(D_s-B) is contained in
range(E_beta), regardless of differentiability of eta. The matrix identity
alone is NOT a dimensional lower bound for a one-dimensional threshold chain.
It is also not an operation-count or every-possible-domain lower bound.

An ordinary full-row-rank control is

```text
f_i=x+z_i/8-theta_i,
x in [-2,2], z_i in [-1,1],
theta=(-3/4,-1/4,1/4,3/4).
```

Adjacent gaps are at least 1/4. Independent z columns give row rank four;
all prefix cells have open source neighborhoods, including both endpoints.
No trained-model prevalence is claimed. D153's all-independent-phase premise
does not apply to this ordered family; its earlier theorem is not contradicted.

## The complete consumer difference rank pays for terminal products

Stack EVERY remaining consumer, predicate readout and required value decoder
of this block into

```text
Y=C_L*q_L+C_R*q_R+H*s+b,
D=C_R-C_L,
Y=C_L*f_L+C_L*t+rho*D*t+H*s+b.
```

This stack excludes the original gate-defining rows only if an exact new
block graph replaces them. Source predicates remain. If D=L*R is an EXACT,
certified rank-d factorization, introduce z=rho*(R*t); then

```text
Y=C_L*f_L+C_L*t+L*z+H*s+b.
```

There are r shared amplitudes and d scalar binary-continuous products, not
necessarily r products. Thus r+d<m is possible when d<r. A full-rank D
recovers m amplitudes under this lowering. An identity consumer of the whole
q vector forces full column rank of D; an original-source skip does not.

For each row v=(R*t)_j with certified bounds a<=v<=b, the standard exact
binary-product lowering is

```text
z >= a*rho,       z <= b*rho,
z >= v-b*(1-rho), z <= v-a*(1-rho).
```

Inlining v uses d new variables and 4d rows. Only TWO rows per product contain
R: their repeated coefficient cost is 2*nnz(R), plus z/rho coefficients.
Materializing v instead adds d variables and d equations. Reliable bounds,
factorization certificates, L readouts and any coefficient fill also cost.
Approximate floating-point rank or deleting small singular values is not
licensed by this exact theorem.

The d=0 case includes equal paired columns; scaled clique normalization gives
the corresponding D137 common-normalized-column case. The range 0<d<r is
a broader useful algebraic interface, but low-rank factoring a common binary
product is not claimed as a new principle. It is not D143's product of two
continuous aggregates: rho here remains one original binary, so the native
product lowering IS integer exact.

## Six-row max graph and symmetric costs

For one ordered pair f>=g, with the same bounds and original pair bits, use

```text
t>=0,  t>=-f,  t>=g,
t<=(-l_f)*(1-beta_f)+u_g*beta_g,
t<=-f+(u_f+u_g)*beta_f,
t<=g+(-l_f-l_g)*(1-beta_g).
```

For bits 00 the second upper forces t=-f>=0 and order gives g<=0. For 10,
the first upper forces t=0, f>=0 and g<=0. For 11, the third upper forces
t=g>=0, with f>=0. The remaining upper inequalities follow from the reliable
source bounds. For 01 the second and third uppers give t=-f=g>=0; with order
this forces f=g=t=0. That is a legal weak-order zero-label combination, not
a reason to delete it. Strict order excludes it. This is a proof by cases,
not an executable phase split.

The six rows imply the original sign guards for INTEGER bits. They do not
automatically imply all the old relaxed guards. Explicitly retaining
l_i*(1-beta_i)<=f_i<=u_i*beta_i adds 2m rows unless separately certified
redundant in the LP. All counterexamples below also satisfy those guards.

For m=4, sole complete consumer C=(2,-1,1,3), D=(-1,4) has rank one:

```text
Y=2*f1-f2+2*t1-t2+z,   z=rho*(-t1+4*t2).
```

Counting physical f columns, nonnegative rows, and nonzero product-bound
coefficients, but excluding the common source system and separately declared
scalar bounds/readouts:

| Formulation | New continuous | Original bits | Graph/product rows | Row nnz |
| --- | ---: | ---: | ---: | ---: |
| Four original gates | 4 | 4 | 16 | 32 |
| Two folded pairs, one product | 3 | 4 | 16 | 40 |
| Folded form plus explicit relaxed sign guards | 3 | 4 | 24 | 56 |

Each max block has 14 nnz; the two-nonzero signed product has 12 nnz. These
are symbolic counts, not complete bytes or timing. Both sides still pay for
source equations or affine expansion, source bounds, all retained predicates,
amplitude upper bounds, consumer coefficients, RHS, original-bit bounds,
metadata, witnesses, and any backend inequality-slack conversion. The folded
readout has more entries than the original four-coefficient readout here.
No unconditional end-to-end saving is inferred from the one-coordinate gain.

## Cheap max lowering loses old LP inclusion

An ordinary nonparallel two-source control is

```text
f=x+y/8+1/4, g=x-y/8-1/4, x,y in [-1,1],
[l_f,u_f]=[-7/8,11/8], [l_g,u_g]=[-11/8,7/8].
```

Its order gap is at least 1/4. At x=y=0, beta_f=1/4, beta_g=1/10, the six
rows allow t=1/4. The exact integer sum readout f+t is therefore 1/2 in
this LP, whereas the old same-source, same-bit LP requires
q_f+q_g<=(11/8)/4+(7/8)/10=69/160. Original sign guards and the strict-order
bit relation both hold. Their addition does not close the gap.

On the full-row-rank four-gate control above, take x=z_i=0, all beta_i=1/2,
t1=t2=13/8. Bounds are

```text
l=(-11,-15,-19,-23)/8, u=(23,19,15,11)/8,
t_i<=15/8,
v=-t1+4*t2=39/8 in [-15/8,15/2].
```

All three max uppers for each pair equal 13/8, and all original sign guards
hold. The four product rows admit z=15/4. Hence the folded Y=53/8. The old
same-labelled LP instead has

```text
Y<=2*(23/16)-1/4+15/16+3*(11/16)=45/8.
```

This is a gap of one on a mixed consumer, despite integer exactness and one
fewer amplitude. It rejects this particular box-bounded lowering, not every
phase-aware joint lowering or the exact folded relation itself.

## A positive projection belongs to the existing order component

There is a narrower way to remove a continuous value without losing the old
relaxation. If the ONLY complete consumer of an ordered pair is d=q_f-q_g,
put delta=f-g. Projecting D018's order-enhanced eight rows gives exactly

```text
0<=d, d<=delta,
d<=u_f*beta_f,
d<=f-l_f*(1-beta_f),
d>=f-u_g*beta_g,
d>=delta+l_g*(1-beta_g),
f<=u_f*beta_f,
g>=l_g*(1-beta_g).
```

To verify the projection, recover a relaxed q_g in the interval

```text
[max(0,f-d),
 min(u_f*beta_f-d, f-l_f*(1-beta_f)-d,
     u_g*beta_g, g-l_g*(1-beta_g))],
q_f=q_g+d.
```

The eight displayed inequalities are precisely its nonemptiness conditions
after removing conditions already implied by beta in [0,1] and l<=0<=u.
Thus this is an exact LP projection of D018, not the weak independent product
construction. Integer semantics reconstructs q_g=ReLU(g), with original
bits and source preserved. Extra consumers of the individual q values would
invalidate the premise unless also paid for in the complete projection.

This keeps eight predicate rows and reduces two amplitudes to one. It has
20 predicate nnz with physical f,g columns, versus 16 for the unenhanced two
original gates and 19 for D018's ordered pair. Readout/storage costs differ
as well. It is a useful known order-component projection, not a new domain
principle or proven runtime/capability gain. Positive rescaling extends it to
opposite-sign unequal weights ONLY if the rescaled preactivations are still
certifiably ordered; the original order does not imply that condition.

For a general positive weighted residual sum c1*(q_f-f)+c2*q_g, c1,c2>0,
direct interval elimination gives three lower sum bounds, four upper sum
bounds and four source-feasibility guards (eleven rows before scalar bounds).
The nominal fourth lower is redundant: f>=g prevents both -f and g from
being positive. Some special coefficients/source relations can make further
rows redundant; this count is a direct construction, not a minimality bound.
It does not rescue a generic cheap six-row projection claim.

## Even a complete max-block hull does not glue to an independent product

A stronger paper control isolates composition from the cheap six-row graph.
Let x,y in [-1,1] and

```text
f=(x+3*y/20+3/4, x+y/20+1/4,
   x-y/20-1/4, x-3*y/20-3/4).
```

Every gate crosses, adjacent gaps are at least 2/5, and all four normals are
distinct. Use the same C, pairing, rho and v. Put b=1/4+y/20 in [1/5,3/10].
Then f=(x+3b,x+b,x-b,x-3b). On the five ordered intervals in x, the exact v
is respectively

```text
-3*x-b;  -4*x-4*b;  0;  -x+b;  3*x-11*b,
cut at -3b,-b,b,3b.
```

This direct paper calculation yields tight global bounds [-3/5,14/5],
and tight conditional intervals

```text
rho=0: v in [0,14/5],
rho=1: v in [-3/5,4/5].
```

Now average these two TRUE full source/ALL-bit/max-block tuples with weight
one half each:

```text
A: x=1/5,y=0,  beta=1100, t=(0,0),    rho=1,
B: x=-3/10,y=0,beta=1000, t=(0,1/20), rho=0.
```

All eight preactivation values in A and B are nonzero. The resulting point is

```text
x=-1/20,y=0, f=(7/10,1/5,-3/10,-4/5),
beta=(1,1/2,0,0), rho=1/2, t=(0,1/40), v=1/10.
```

It lies in the COMPLETE joint source-labelled max-block convex hull, not
merely an intersection of separate gate hulls. Giving that stronger hull
to a comparator is a mathematical stress test, not authorization to build
a hull-enumeration helper.

This same source also explicitly separates six-row max graphs from old
relaxed sign guards: at x=y=0, beta=(3/10,1/5,0,0), t=(0,0), all twelve
max rows hold but f1=3/4>(19/10)*(3/10)=57/100. Integer implication of guards
is therefore not a license to omit their fractional-strength cost.

Even use the tight conditional scalar product hull

```text
(-3/5)*rho <= z <= (4/5)*rho,
0*(1-rho) <= v-z <= (14/5)*(1-rho).
```

It admits z=1/10. The folded consumer gives Y=51/40. In the old LP at the
same source and ALL original bits, q1=f1, q3=q4=0 and q2>=f2, so
Y<=2*f1-f2=6/5=48/40. The loss is 3/40 even after using the stronger joint
max hull and tight rho-conditioned v intervals.

With only global McCormick bounds it admits z=2/5 and Y=63/40, a still
larger gap. The problem is the independently convexified product losing
which source/phase mixture produced v, not simply bad max-block bounds.
For example, these controls also have valid all-bit relations
-(4/5)*beta3<=z<=(8/5)*beta4, which exclude the point. This illustrates what
information was lost; it neither proves that this cut family fully repairs
the lowering nor proposes a cut/helper implementation.

At this fixed labelled source, a following ReLU(Y-5/4) has folded value
1/40 while the old parent LP puts its input below zero. This is NOT a
global property certificate, actual benchmark regression, or valid ADV:
the comparison fixes source and original phases. No ordinary-network score
is inferred. A general no-regression claim nevertheless cannot follow from
these independent component hulls.

## Composition and research decision

Affine/Conv readouts, aligned Add and Concat can stack and transform the
complete consumer packet exactly. They can also increase its difference rank.
An arbitrary new mixed/bias ReLU is not proved closed in the same small
folded language: restoring its ordinary exact graph adds the original bit,
continuous value and predicates. That is an exact fallback representation,
not a successful compact cross-layer calculus or a result-dependent menu.
No smooth/Transformer transfer or useful GPU set-query kernel is established.

D018 with the same certified order is a stronger fair comparator than the
unaugmented four-row LP. Since the demonstrated formulation already loses
the latter's same-labelled constraints, it has no proved inclusion in D018.
Merely retaining every old q and row alongside the folded interface would
abandon the claimed reduction and turn this into an auxiliary formulation.

Known formulation research separates size and LP strength: Vielma studies
embedding complexity for ideal disjunctive formulations, and Anderson et al.
contrast compact-variable ReLU formulations with extended ones. Their
abstracts were checked here as context, not as proofs of D154 or authority
to add their separation/search algorithms. [Vielma](https://arxiv.org/abs/1506.01417),
[Anderson et al.](https://arxiv.org/abs/1811.08359).

Decision: retain the exact shared-phase/consumer-rank theorem as supporting
mathematics, but do NOT implement this independent max-plus-product lowering
as the next Neural-HZ candidate. There is no proven all-cost advantage,
old-LP preservation, measured real-structure population, new solved case or
established literature novelty. This is not a general impossibility theorem.

The open definition question is a source/phase/value interface with a useful
joint elimination or cross-layer transfer law that does not discard the
coupling when queried. Another product factorization plus independent hulls,
or appending helper cuts to the unchanged old graph, does not answer it.
Real-model screening and numerical implementation should follow such a
positive, fully paid theorem, not substitute for it.
