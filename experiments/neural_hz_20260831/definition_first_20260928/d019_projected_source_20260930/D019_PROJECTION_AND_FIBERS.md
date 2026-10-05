# Projected source relations and their composition limits

The useful result of this investigation is a precise separation: compressing the source seen by a nonlinear block can preserve the exact integer graph, but even an ideal projected formulation need not preserve its convex relations with the retained input. A sufficient composition condition and a restricted rank obstruction identify the boundary. None of these results establishes a new Neural-HZ domain or a benchmark improvement.

This is a paper derivation, independently checked, for the definition-first Neural-HZ project. It tests whether the common-source lift in the preceding investigation can avoid copying full source coordinates. It does not implement phase-region enumeration, replace HZ by a convex domain, or change the terminal solver. The original source, all original phase bits, and concrete-input decoder remain present throughout.

## Setting and exact integer preservation

Let `u in S` be the original exact HZ relation, including its continuous factors, binary factors, equalities, inequalities, and shared identities. Let `x` be its retained continuous source port. Consider a ReLU bank

```text
f = A x + b = C t + b,     t = R x,     A = C R,
y_i = max(0, f_i).
```

Each original gate bit `beta_i` is retained: `beta_i=0` requires `f_i<=0, y_i=0`, and `beta_i=1` requires `f_i>=0, y_i=f_i`. At `f_i=0` both choices remain legal, independently for every gate. Let `Q` be a bounded convex polytope containing every `R x` from `S`. Write `G_Q` for this retained-bit gate graph over `t in Q`.

**Integer preservation.** Suppose a lifted relation `F_Q(t,beta,y,w)` is pointwise exact for `G_Q` at every binary `beta`. Then

```text
u in S,     t = R x,     exists w: F_Q(t,beta,y,w)
```

projects to exactly the original relation on `(u,beta,y)`.

Proof: an original point has `t in Q`, and exactness supplies a lift. Conversely, a lifted point retains `u in S`; at its binary `beta`, pointwise exactness enforces precisely the original gate values and guards. No source coordinate or original bit has been reconstructed from an average or deleted. The existing concrete-input decoder still applies to `u`.

In particular, an extended formulation whose projection is exactly `conv(G_Q)` is pointwise exact at binary `beta`. Indeed, a binary vector that is the average of binary vectors forces all positive-weight terms to have the same bit vector. For a fixed bit vector the sign region is convex and the gate outputs are affine. Their average remains in that same graph region, including its legal zero boundaries.

This theorem says nothing about the size or availability of an ideal formulation, nor about the cost of describing `Q`.

## Conditions for keeping the old LP strength

Integer equivalence alone does not guarantee a nonweaker terminal relaxation. Two sufficient ways to retain the original gate relaxation are:

1. Keep all old gate inequalities explicitly, alongside a sound new relaxation.
2. Use a relaxation contained in `conv(G_Q)` and ensure that every old preactivation bound `L_i <= f_i <= U_i` is valid on all of `Q`.

For the second case, the old four gate inequalities are valid on `G_Q`, hence on its convex hull. If the bounds were proved only for the smaller true projected source `R(S)`, their validity on a larger outer polytope `Q` cannot be assumed. The first option avoids this issue but its additional rows and coefficients must be paid for. All comparisons retain the same relaxation of the upstream source predicates.

This is a sufficient containment statement, not a prescription to replace the nonconvex domain by its LP relaxation.

## A projected ideal hull can lose an ordinary residual relation

Take the box input and one gate

```text
P = [0,1]^2,
t = x_1 + x_2,     Q = [0,2] = R(P),
y = ReLU(t - 1).
```

The projected hull contains `(t,beta,y)=(3/2,3/4,3/4)`, because it is the mixture

```text
(1/4) * (0,0,0) + (3/4) * (2,1,1).
```

Pull it back to the retained input `x=(9/10,3/5)`. Both coordinates lie strictly inside the original box and their sum is `3/2`. Thus `x in P` and `(R x,beta,y) in conv(G_Q)` hold. Nevertheless every exact source graph point satisfies `y<=x_2`: when the gate is active, `x_1+x_2-1<=x_2`, and otherwise `y=0<=x_2`. The inequality is valid on the full retained-source hull, but the pulled-back point violates it by `3/20`.

Consequently the ordinary residual `z=x_2-y` has the spurious value `-3/20`, although the exact network has `z>=0`. This uses neither a special floating-point condition nor a zero activation at the mean. The gate bit of the spurious point is fractional; it does not contradict integer preservation or show an invalid concrete HZ point.

The lost information is which source points can jointly explain the projected mixture and the retained residual branch. Equality of their mean projected input is insufficient.

## A sufficient condition for convex hull composition

For this section only, let `P` be a convex bounded source set and `Q=R(P)`. Assume its fibers have the product form

```text
P = { s(t) + k : t in Q, k in K },
R s(t) = t,
```

where `s` is affine and `K` is a fixed convex subset of `ker(R)`, independent of `t`. Then

```text
conv { (x,beta,y) : x in P, (R x,beta,y) in G_Q }
  = { (x,beta,y) : x in P, (R x,beta,y) in conv(G_Q) }.
```

Proof: containment from left to right follows from convexity. For the reverse containment, write a projected point as a finite mixture of `(t_j,beta_j,y_j) in G_Q` with weights `lambda_j`. If `x=s(t)+k`, use the same `k` in every source point `x_j=s(t_j)+k`. Each belongs to `P`, and affine `s` gives `sum lambda_j x_j=x`. These points provide the required full-source mixture.

A more general sufficient condition is the corresponding fiber interpolation identity `P_(sum lambda_j t_j) = sum lambda_j P_(t_j)` for every mixture in use, where `P_t={x in P:R x=t}` and the right side is a weighted Minkowski sum. Product fibers imply this identity. We make no claim that product fibers are necessary.

Coordinate projection of a product box is a positive example. The sum projection in the preceding example has nonconstant fibers and fails the conclusion. Applying the theorem to a relaxed upstream polytope proves a result about that polytope only: it does not establish ideality with respect to all original upstream integer bits. A theorem about the full retained source would have to include those coordinates and their correlations.

## A scalar or independent box projection gives the old gate relaxation

For one preactivation `f in [L,U]`, with `L<0<U`, the retained-bit graph hull is exactly

```text
0 <= beta <= 1,
y >= 0,     y >= f,
y <= U beta,
y <= f - L(1-beta).
```

For `0<beta<1`, the four inequalities produce two legal phase inputs

```text
f_minus = (f-y)/(1-beta) in [L,0],
f_plus  = y/beta         in [0,U].
```

The point is their mixture with active weight `beta`. At the endpoints the gate is exact directly. This proves sufficiency; validity proves necessity.

For a bank projected to its preactivations with `Q=product_i [L_i,U_i]`, the gate graph is a Cartesian product. Convex hull commutes with a Cartesian product, so pulling its ideal hull back through `f=A x+b` gives exactly the ordinary per-neuron big-M relaxation. Changing the names of those coordinates introduces no new strength. A stronger proposal must preserve additional, explicitly paid-for source correlations.

## Rank required by a projection-only full-source ideal formulation

The following elementary lower bound concerns a restricted formulation interface, not Neural-HZ in general. Let

```text
P = [0,1]^d,     d >= 2,
f(x) = sum_i x_i - theta,     0 < theta < 1,
y = ReLU(f(x)),
```

with its original bit `beta`. Suppose a formulation equal to the full retained-`(x,beta,y)` graph hull has the form

```text
x in P,     exists w: C(R x,beta,y,w),
```

so every nonsource constraint can observe `x` only through the fixed linear map `R`. Then `rank(R)=d` is necessary. Auxiliary variables and the form of `C` do not change the argument.

For every coordinate `i`, the exact graph and its convex hull satisfy

```text
I_i: y <= x_i + (d-1-theta) beta.
```

For `beta=0` this follows from `x_i>=0`; for `beta=1` it follows from `sum_(j!=i) x_j<=d-1`. Put `a=theta/[2(d-1)]`. Choose a legal inactive point with `x_i^0=0` and `x_j^0=a` for `j!=i`; its preactivation is `-theta/2`. Choose a legal active point with `x_i^1=1/2` and `x_j^1=1` for `j!=i`; its preactivation is `d-1/2-theta>0`. Their equal mixture has

```text
xbar_i = 1/4,
xbar_j = (1+a)/2 for j != i,
betabar = 1/2,
ybar = (d-1/2-theta)/2.
```

It is strictly interior to the box and tight for `I_i`. If `rank(R)<d`, select a nonzero `h in ker(R)` and a coordinate `i` with `h_i!=0`. Perturb this mixture to `x'=xbar-epsilon sign(h_i) h` for sufficiently small positive `epsilon`. The point remains in the box and preserves `R x`, `beta`, and `y`. The same auxiliary witness must remain feasible, yet the right-hand side of `I_i` decreases by `epsilon |h_i|`. This contradicts exact hull representation.

The proof uses an ordinary box and one gate, not a numerical corner-case experiment. It requires no completeness claim about a published list of facets. The inequalities and the general projection issue are established mathematical-programming territory; independent checking of this argument is not a novelty certification.

The scope matters: this does not prohibit integer-exact low-rank encodings, sound or useful nonideal relaxations, stronger-than-big-M projected relations, or output-only hulls. It does not force full source copying if additional predicates can refer to the original source outside `R`; those extra observations simply lie outside this restricted interface. Nor does full rank guarantee an efficient formulation.

## Consequences for a definition-first candidate

The candidate cannot obtain a new domain merely by writing `t=R x` and attaching an ideal hull in `t`. There must be an explicit account of source fibers, retained branch consumers, and phase-dependent values across composition. Full-source ideality is not a project requirement; a useful candidate may instead prove a strict precision versus cost improvement over a declared comparable representation.

Future candidate comparisons must include ordinary HZ with the same source relations, strong single-neuron formulations, and small multineuron hull approximations. Product fibers provide a sufficient positive composition test, but their real-network prevalence is unmeasured. No claim about CIFAR or TinyImageNet prevalence follows from these proofs.

GPU execution changes the cost question, not these semantic facts. Parallelizing an incorrect pullback does not recover lost source information. The accompanying review identifies matrix-vector-friendly constraint processing as an implementation target while keeping domain innovation and solver acceleration separate.

## Provenance and verification

Date: 2026-09-30. Branch: `redu-hz`. Commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, with pre-existing uncommitted work. Mode: paper proof and primary-source comparison only. Dependencies are the frozen goal amendment and preceding source/order investigations, listed in the checkpoint. Root derived the interior example and rank corollary; an independent read-only mathematics reviewer checked the exactness, composition, box no-gain, interior counterexample, and rank arguments. No executable test, benchmark, full replay, or GPU experiment was run for this document.
