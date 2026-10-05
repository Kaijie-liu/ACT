# Shared curvature relations for neural forward transfer

This paper result gives a finite joint-error interface for a group of existing activations whose preactivations lie on, or near, one common affine segment. Its ReLU instance has a linear-size description and supports mixed readouts by a prefix scan. A biased, genuinely two-source control separates it from the intersection of every source-labelled two-gate convex hull, even when the next ReLU has exact global scalar bounds. A bounded third-source perturbation preserves that separation. These are paper results, not trained-model measurements or a completed new abstract domain.

The underlying interpolation, convexity and curvature mechanisms are classical. The research opportunity is a certified forward relation generator with useful composition and full-cost behaviour, not a claim to have invented the kernel or to be stronger than HZ supplied exactly the same rows.

## Carrier and protected semantics

Write a research element as (H, C, decoder). H retains the original continuous factors, every original signed binary factor, EQ/LE predicates, common latent/frame identity and output readouts. C records finite source-bound curvature relations, their applicability certificates and their installed consequences. All relations refer to the same original state theta. Empty C embeds H exactly. Semantic inclusion is inclusion of concretizations; no complete lattice or computable best abstraction is asserted.

For the ReLU theorem, H must certify the actual original equations q_j=ReLU(g_j), including the original phase guards. A reference formula from an unbound or differently rounded network is insufficient. The original input decoder and all other consumers remain unchanged. Every added exact-real row holds for every original integer theta, so projection of the extended ReLU system back to the original coordinates is exactly H. In particular, adding the rows strengthens some continuous query relaxations; it does not enlarge the expressive power of the protected integer set.

The finite curvature polytope below is an auxiliary consequence, not a replacement of H by a convex domain. Original bits are never deleted, pivoted or made continuous in the actual domain. Fractional bits appear only in the comparison proof. Both legal original labels at a zero preactivation remain protected.

## Common segment identity

Take two actual existing gates p=ReLU(f), q=ReLU(h), and m existing interior gates. All f,h,g_i are functions of the same theta and may depend on many genuine input dimensions. Fixed coefficients satisfy

```text
0 = t_0 < t_1 < ... < t_m < t_(m+1) = 1
g_i^0 = (1-t_i) f + t_i h
e_i = (1-t_i) p + t_i q - ReLU(g_i^0)
G(t,s) = min(t,s) - t*s
```

If f and h have opposite signs, d=abs(h-f) and k is the unique segment zero. Directly evaluating the two linear pieces gives e_i=d*G(t_i,k). If f,h have the same weak sign, e is zero. Endpoint zeros and f=h are covered without division: e=0 whenever the segment has no interior crossing.

The original gates need not create a new absolute-value gate. The existing affine readout

```text
S = 2*p + 2*q - f - h = abs(f) + abs(h)
```

is nonnegative on every original state. At an interior crossing S=d; otherwise e=0. Hence the actual defect vector belongs to the complete tent family with amplitude S, with the zero vector included. This identity is the reason a source-dependent curvature budget can lower to ordinary linear predicates without new products or bits.

The hypothetical ReLU(g_i^0) used when an actual g_i has a residual is only a proof function. It is not a free new network node, reference gate or binary factor.

## Exact finite hull of the declared error family

For a fixed nonnegative B, let T(B) consist of the vectors B*G(t_i,k), for k in [0,1]. Set e_0=e_(m+1)=0, a_i=t_i-t_(i-1), and v_i=(e_i-e_(i-1))/a_i. Then

```text
conv T(B) = { e : v_i >= v_(i+1) for i=1..m,
                  v_1 - v_(m+1) <= B }.
```

Proof: define nonnegative discrete curvature masses mu_i=v_i-v_(i+1). Piecewise-linear interpolation with zero endpoints gives

```text
e_j = sum_i mu_i * G(t_j,t_i)
sum_i mu_i = v_1 - v_(m+1) <= B.
```

For B>0 this is a convex combination of the m knot vectors B*G(.,t_i) and the zero vector. B=0 forces all masses and e to zero. Conversely, each tent is concave, its sampled slope drop is at most B, and the stated rows are convex. This proves equality. It is an exact hull of this declared error family, NOT the full source-labelled neural graph or a claim that every curvature vector is reachable from the actual source domain.

For exact interior gates q_i=ReLU(g_i^0), substitute e_i=(1-t_i)p+t_i*q-q_i. The m concavity rows become ordinary convex interpolation rows on q_0=p,q_(m+1)=q:

```text
a_(i+1)*q_(i-1) - (a_i+a_(i+1))*q_i + a_i*q_(i+1) >= 0.
```

The single budget row is

```text
(p-q_1)/a_1 + (q-q_m)/a_(m+1) <= S.
```

For m>=2 and already physical distinct f,h,p,q,q_i readouts, these are m+1 LE, at most 3m+6 coefficient nonzeros after substituting S, zero new continuous quantities and zero new binary factors. This is a coordinate-level bill; latent expansion, normalization, RHS, any required slack encoding and evidence are additional.

## Mixed readouts and forward consumption

For any fixed signed row w, define K_w(s)=sum_i w_i*G(t_i,s). It is piecewise linear with breakpoints only at the fixed t_i. Let L be its minimum and U its maximum over {0,1,t_1,...,t_m}; L<=0<=U. Then exactly over conv T(B),

```text
L*B <= w*e <= U*B.
```

After sorting t once, evaluate each knot with prefix sums of w_i and w_i*t_i. For a knot s, if A=sum_(t_i<=s) w_i*t_i, C=sum_(t_i>s) w_i and T=sum_i w_i*t_i, then K_w(s)=A+s*C-s*T. One readout costs O(m) arithmetic after the O(m log m) sort. No source subdomain, phase assignment, terminal LP state or dual information is queried. A coefficient-knot scan is not an input or phase split.

Affine/Conv and shared Add/Concat combine the weights of the SAME relation first. Boxing each intermediate e_i independently would discard the point of the interface. Different source frames cannot be identified because their coefficients or tensor shapes happen to match. Multiple blocks retain their shared original H; their independent outer bounds are sound but are not claimed jointly exact.

Writing A_w=sum_i w_i*((1-t_i)p+t_i*q), the actual mixed output F=sum_i w_i*q_i has the source-dependent envelope A_w-U*S <= F <= A_w-L*S. A residual affine readout is added using the same theta. For the next original r=ReLU(F+b), a certified affine upper envelope V of F+b, with V>=0, gives the direct row r<=V. More generally one can use a certified secant upper bound for ReLU(V). Original r, its bit and all its guards remain. This is a sound forward use, not an arbitrary-depth closure theorem for one tent family.

## Separation from every two gate source hull

Use the ordinary biased source box x,y in [-1,1], with

```text
f = x + y/5 + 1/10
h = -x/4 + y + 1/5
p = ReLU(f), q = ReLU(h)
q_i = ReLU((1-t_i)*f+t_i*h), t=(1/4,1/2,3/4).
```

The inverse source map is x=20f/21-4h/21-2/35 and y=5f/21+20h/21-3/14. Consider the fractional tuple

```text
f=h=0, x=-2/35, y=-3/14
p=q=1/5, q_1=q_2=q_3=1/20
all five original active-bit means = 1/2.
```

Here an active indicator is (b+1)/2 for an original signed bit b; its displayed mean 1/2 corresponds to signed mean zero, not to changing the domain's binary encoding.

Define the comparison explicitly: retain the original linear source equalities and the intersection of all ten complete convex hulls of TWO original labelled gates over the SAME full x,y box. A pair hull includes x,y and its two outputs/bits. The convex-mixture witnesses used for different pairs may differ; the intersection enforces their common displayed coordinate means, not one joint distribution.

Every pair admits the displayed tuple. The following table supplies its witness: take the two source points (f*,h*) and (-f*,-h*) with equal weights, then apply the inverse map. The selected gates have opposite nonzero preactivations, so each selected bit mean is 1/2.

| Selected pair | Positive representative (f*,h*) |
| --- | --- |
| p and q | (2/5,2/5) |
| p and q_i | (2/5, 2/5-3/(10*t_i)) |
| q and q_i | (2/5-3/(10*(1-t_i)), 2/5) |
| any two interior gates | (1/10,1/10) |

The relevant measured positive preactivation is 2/5 for an endpoint and 1/10 for an interior gate. All representatives and their negatives map strictly inside the source box. Over this finite list the extreme x values are -94/105 and 82/105, and extreme y values are -37/42 and 19/42. Other gate bits, if carried as unconstrained pair coordinates, can also be assigned complementary zero labels where needed; this does not assert matching the other outputs in a given pair witness.

The tuple additionally passes all three independently tight tent caps 0<=e_i<=t_i*(1-t_i)*S and every discrete concavity row: S=4/5 and e=(3/20,3/20,3/20). Nevertheless the joint budget requires 4*(e_1+e_3)<=S, while its left side is 6/5. In original coordinates this is

```text
Y = (p+q)/2 + (f+h)/4 - q_1 - q_3 <= 0.
```

The false tuple has Y=1/10. Add the actual next original gate

```text
r = ReLU(Y + (f+h)/4 + 1/2).
```

Since f+h lies in [-33/20,9/4], V=(f+h)/4+1/2 is at least 7/80. The new source row implies r<=V. The false tuple can take its exact child graph value r=3/5 with child bit 1, but the new row requires r<=1/2, excluding it by 1/10.

This is not an artifact of loose scalar bounds for the child. The real source (f,h)=(-2/5,6/5), or (x,y)=(-2/3,5/6), gives Y=0 and child preactivation 7/10. At x=y=-1 that preactivation is -13/40. Therefore even the exact true global scalar interval crosses zero and contains the false child's 3/5. Its original scalar labelled hull admits the exact child tuple. No exact global extrema are claimed beyond these two witnesses, and the comparator does not include complete cross-layer hulls involving the child.

The new row is a known multi-gate convexity/absolute-value consequence and is contained in the full group hull. For these knots it is equivalent to abs((3*f+h)/4)+abs((f+3*h)/4)>=(abs(f)+abs(h))/2, a triangle-inequality consequence of the inverse linear map. What the control proves is a strict, source-labelled, downstream relation advantage over the stated pair-hull intersection plus independent caps and concavity, generated by one explicit budget row. It does not compare against complete three-gate hulls, prove novelty or establish a trained-network gain. The four group rows use fifteen coordinate nonzeros; an additionally installed child row is a separate LE with three coordinate nonzeros in r,f,h, before source expansion.

## Residuals and a third genuine source

If an actual interior preactivation is g_i=g_i^0+r_i(theta), put eta_i=ReLU(g_i)-ReLU(g_i^0). Then min(0,r_i)<=eta_i<=max(0,r_i). For a fixed row w,

```text
-sum_i ReLU(-w_i*r_i) <= sum_i w_i*eta_i
                               <= sum_i ReLU(w_i*r_i).
```

These terms may be bounded by certified scalar secants and then merged on the original source coordinates; they need not become new ReLU gates. A simpler paid bound is sum_i abs(w_i)*rho_i when abs(r_i)<=rho_i. For any installed linear row, use its actual residual coefficients before bounding; no free independent-error replacement or residual cancellation is assumed.

The pair-hull control is robust to a genuine third source. Add z in [-1,1], perturb g_1 by eps*z and g_3 by -eps*z, and keep g_2 unchanged, with eps=1/20. The two increments have opposite signs and magnitude at most eps*abs(z), so abs(eta_1+eta_3)<=eps*abs(z)<=eps. Consequently the actual Y<=eps and r<=V+eps. The same false tuple with z=0 violates that row by 1/20. Every pair witness above also takes z=0, so it remains a legitimate witness for every full pair hull over the three-source box. The two true child witnesses likewise remain at z=0. This proves a bounded-residual structural control, not generic trained-filter near-collinearity.

## Smooth activations through the same kernel

For a C2 activation phi and u(t)=phi(f+t*(h-f)), integration by parts gives

```text
(1-t)*phi(f)+t*phi(h)-phi(f+t*(h-f))
    = (h-f)^2 * integral_0^1 G(t,s)*phi''(f+s*(h-f)) ds.
```

For convex phi this is a nonnegative shared curvature measure. A certified bound on its total mass gives the same finite sampled concavity/budget factor as above. For general signed curvature use a certified derivative-density bound, not the nonnegative-curvature cone. If abs(phi'')<=M over the entire required interval, a mixed defect has magnitude at most M*(h-f)^2*integral abs(K_w). The kernel is combined BEFORE taking an absolute value. Its piecewise-linear signed areas are computable from the fixed coefficient knots. No input subdivision or numerical quadrature is authorized here. Turning the squared source difference into terminal linear rows still requires a paid certified enclosure or an existing suitable relation.

A hand-checkable comparison uses t=(0,1/3,2/3,1), w=(1,-3,3,-1), F=sum_i w_i*phi(f+t_i*(h-f)). The affine moments cancel. The Peano kernel for F is s, 1-2s and s-1 on the three consecutive thirds. Its positive and negative areas have magnitude 1/12, hence integral abs(kernel)=1/6. Thus abs(F)<=M*(h-f)^2/6. Using the same shared first-order Taylor core but independent coordinate remainders gives M*(h-f)^2/3 at the best common expansion point 1/2. With 0<=phi''<=M the corresponding symmetric bounds are respectively M*(h-f)^2/12 and M*(h-f)^2/6. This factor-two comparison is between explicit error algorithms, not against every Taylor method: a reference that applies the same shared Peano kernel is exactly equal.

This identity is classical Peano interpolation theory, not an invented smooth domain theorem. [Fass, Section 1.4, Theorem 1.19](https://www.math.iit.edu/~fass/478578_Chapter_1.pdf) states the kernel integral and norm bounds. The displayed neural specialization and controls are derived here. Smooth soundness protects actual activation states; it does not automatically preserve every old approximation-error assignment or provide an exact finite polyhedral smooth graph. No GELU, SiLU, sigmoid, tanh, Attention or Transformer execution is qualified by this paragraph.

## Structural generation and complete cost obligations

An admissible future generator must fix grouping structurally, bind its endpoints to existing original gates and certify every actual affine residual. One possible deterministic construction selects anchors from a fixed group, projects each row onto their coefficient difference, clips the coefficient to [0,1], and retains the exact residual. This is coefficient arithmetic, not optimization of a terminal margin. It is only a possible generator: repeated knots, endpoint hits or large residuals require a separately specified conservative rule before execution; no such implementation is frozen here.

For already certified distinct knots the algebraic work is linear-size row generation, O(m log m) sorting, and O(m) per mixed readout. For k readouts this is O(m log m+km), plus all real source coefficient occurrences, residual certification and predicate expansion. An exact group implementation may install m+1 shared rows; direction-specific lowering may instead install the derived readout rows, but that policy must be fixed before a run, not chosen from solver outcomes.

GPU prefix scans and sparse three-point stencils are plausible implementation targets. They are not measured GPU acceleration. Coefficient and RHS rounding, 512-bit exact-certificate limits, physical owners, aliases, intermediate and host/device copies, original H storage, terminal queries, evidence and input reconstruction remain payable under the existing gates. Small knot spacings must not be hidden by ignoring their coefficient bit cost.

The mathematical primitive is comparable to known convex interpolation and multi-neuron relaxations. [PRIMA, Section 2](https://arxiv.org/pdf/2103.03638) motivates capturing joint activation dependencies with multi-neuron hull approximations; none of its search/refinement or split procedures is imported. A full group hull already implies our rows. Compilability into HZ with the same rows is not a rejection criterion, but a genuine method contribution still needs useful automatic generation/composition and a fair full-cost comparison.

No code, freeze, RUN, AST/import/compile, tests, solver, model or GPU execution was created for this paper. Formal scores remain unchanged. Local provenance and qualification flags are in [STATUS.json](STATUS.json); this paper uses exact-real algebra only and introduces no runtime dependency or configuration change. It was developed on 2026-10-02, branch redu-hz, commit f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac.
