# Finite tilt selection and a nonconstant difference separation

2026-10-21, static follow-up at
`4218aec46d35cf4a053758630052729d557410e1`.
The [operator contract](margin_conditioned_support_contract_20261020.md)
now has a concrete, noniterative tilt-selection policy and a shared/private
affine example with nonconstant expert difference. That example separates the
candidate from fixed sign slices plus rectangular McCormick, even if the gate
graph is exact. It does not separate the candidate from arbitrary refinement,
joint product cuts or GenBaB, and is not a measured performance result.

This stage is mathematical/source analysis and documentation only. No algorithm
implementation, numerical control, native import, solver/model/GPU execution,
external adapter or resource-policy change. Old failures and all G1–G6 statuses
remain unchanged. A mathematical policy below is not an execution authorization
or a frozen experiment.

## Two anchors and one source derived tilt per anchor

Keep the supplied guarded factor domain P and its declared finite factor box
`Bbox = c + diag(h)*[-1,1]^n`, h_i>=0. The box is used ONLY to propose a tilt;
the checked support must still use the original P, not silently replace it.
Write the affine objective coefficients as `a_t = b_vec + t*d_vec`, and let
`mu0 = m(c)`. For the box oracle,

    Phi_box(t,nu) - nu*mu0
      = b(c) + t*d(c) - sum_i h_i*abs(a_ti + nu*m_i).

For h_i>0 and m_i!=0 define

    weight_i = h_i*abs(m_i),    ratio_i = -a_ti/m_i.

The tilt maximizing this particular box-center lower bound is any weighted
median of the ratios: strictly left and strictly right weight each must be at
most half the total. Select the point of the entire median interval closest to
zero. If the total weight is zero, select zero. Terms with m_i=0 do not affect
selection, but remain in the support query, its constraints and all checks.

The proof is the identity
`h_i*abs(a_ti+nu*m_i)=weight_i*abs(nu-ratio_i)` and the left/right slopes of
the weighted absolute-deviation sum. This is an elementary one-dimensional
optimization rule, not by itself a new abstract domain or novelty claim.

Use exactly the two already justified gate endpoints l,h as anchors. At each,
propose `{0, nu_median}` and deduplicate equal queries: at most four objective
slots, no conditional LP, iterative search, third anchor or outcome-driven
choice. A fixed gate l=h reduces to its one ordinary joint support obligation.
Anchor refinement is a separate factor, not an automatic rescue after failure.

All projections, ratios, weights, ordering, half-weight ties and constants use
the original bound rational coefficients. An unsupported size/bit count or an
invalid source is a declared refusal, not a rounded approximation of this policy.
Objective slots, distinct algebraic objectives and actual native support calls
must be counted separately. Constant objectives, identical duals or analytical
box supports may legitimately avoid solves on BOTH arms.

### Orientation and representation limits

Swapping experts maps
`m'=-m, d'=-d, b'=b+d, t'=1-t, nu'=-nu`.
The objective itself is unchanged. Ratio signs reverse and weights do not.
Closest-to-zero median selection respects this mapping; choosing the lower
median does not when the minimizer is an interval. Reverse anchors and bind
the property constant once.

Including h_i makes the rule invariant under consistent independent factor
translations, nonzero rescalings/reflections and permutations. It is not
invariant under arbitrary equivalent presentations of a constrained P:

    Original: P=[-1,1], m=z, objective A=-z       -> nu=1.
    Duplicate: P'={z1=z2, |z1|,|z2|<=1},
               m=z1, A=-z2                    -> box rule chooses nu=0.

The constrained support in the second presentation is still `-|nu-1|`,
maximized at nu=1. The box ignores the equality when selecting a tilt, though
the subsequent support must retain it. Freeze the source/factor representation;
do not claim an optimal tilt for P or for the full scalar infimum. Even mu0
can be unreachable in P.

## A nonconstant shared and private example

Consider the GIVEN continuous affine factor box

    P = { (m,u,v) : -1<=m,u,v<=1 },
    C = 1/32,    kappa = 1/256,
    A = C + (m+1/2)/2 + kappa*u,
    B = C - (m+1/2)/2 + kappa*v,
    d = A-B = m+1/2+kappa*(u-v),    b=B.

The shared factor is m; u and v are separate expert-private factors. The weight
of A is t=sigma(m). Optional affine router scores `r_a=m/2, r_b=-m/2` and
outsiders -1 make the unordered selected pair fixed on P. Internal ranking
changes at m=0; this is NOT a route-set-changing certificate or a trained-network
source construction.

For t in [0,1], the same feasible private corner u=v=-1 gives

    psi(mu,t) = C + (t-1/2)*(mu+1/2) - kappa.

The affine feasible path `(mu,-1,-1)` also supplies the matching conditional
upper bound throughout [-1,1]; no feasible-point search is needed in this toy.
At either anchor the median policy chooses nu=1/2-t. It cancels the shared
margin coefficient, while retaining the private coefficients kappa*t and
kappa*(1-t). Consequently

    Phi(t,1/2-t) = C + (t-1/2)/2 - kappa.

Its tilted line equals psi for all mu. With anchors 1/4 and 3/4, which cover
sigma([-1,1]), their interpolation exactly recovers this conditional value
function. These support values are analytically obtained from a box; the
current numerical candidate generator is NOT assumed to return them.

The quantity `(mu+1/2)*(sigma(mu)-1/2)` is nonnegative outside [-1/2,0].
Inside that interval, sigma(mu)-1/2>=mu/4, so it is at least
`mu*(mu+1/2)/4 >= -1/64`. Therefore

    inf_P f >= C - kappa - 1/64 = 3/256 > 0.

This is a valid lower bound, not a claim that the exact minimum equals 3/256.
All anchors share the same private minimizing corner; ordinary box support and
sign-stable reuse also obtain its contribution. This does not establish four
necessary nontrivial solves or a native runtime advantage.

## Why exact gate values do not repair this product relaxation

Compare on the same P using the default sign slices [-1,0] and [0,1].
On the negative slice the tight difference interval is [-D,D],
`D=1/2+2*kappa=65/128`, and the tight gate interval is [l,1/2],
`l=sigma(-1)`. Rectangular McCormick introduces w for lambda*d.

Take the following relaxed point:

    m=-1/2, u=v=0, d=0, b=C,
    lambda=s=sigma(-1/2),    w=-3/64.

For kappa=0, the four McCormick inequalities at d=0 reduce to

    w >= (l-s)/2,          w >= (s-1/2)/2,
    w <= (1/2-s)/2,        w <= (s-l)/2.

The elementary bounds `8/3<e<11/4` (the first four exponential-series terms,
and a geometric upper bound on their remaining tail) imply
`3/8<s<5/13` and `l<3/11`. Thus `s-l>9/88>3/32` and
`1/2-s>3/26>3/32`: both lower planes lie below -3/64 and both upper
planes are positive. For kappa=1/256, the inequalities are explicitly

    w >= D*(l-s),          w >= D*(s-1/2),
    w <= D*(1/2-s),        w <= D*(s-l).

Only the outer factor increases from 1/2 to D; the gate upper endpoint remains
1/2. This widens the admissible w interval. The point has objective

    b+w = 1/32-3/64 = -1/64.

It uses an ACTUAL sigmoid graph point (m,s), not merely a loose gate bound.
Any sound constraint involving only (m,lambda) retains it, so even an exact
gate graph cannot exclude this w while keeping this same product rectangle.
The point is not a real network violation: true lambda*d is zero.

This isolates a concrete gate–expert product gap, rather than only insufficient
sigmoid resolution. The comparison is an exact-arithmetic relaxation analysis;
it neither executes ACT's floating construction nor validates its lowering.

### The stronger baseline can change the conclusion

N1 exposes arbitrary cut_points; `(0.0,)` is a default, not its capacity limit.
For the kappa=0 version, adding the cut -1/2 already lets three-slice N1+M3
prove positivity. On the middle slice let d=m+1/2 in [0,1/2],
`s=sigma(-1/2)`, `a=1/2-s<1/8`. McCormick and the existing negative-sign
gate row imply an objective lower bound

    max(1/32-a*d, 1/32-1/16+d/8).

Its minimum is `(s-3/8)/(32*(5/8-s))>0`; the outer slices have lower
bound 1/32. The kappa>0 example cannot inherit this small positive bound
without checking the private terms. Neither example proves failure of adaptive
partitioning, extra product cuts, algebraic elimination or GenBaB.

The earlier tempting family d=m, b=C-m/2 is rejected as a separation:
sign-sliced H2/F0 already proves C because McCormick contains w>=m/2 on both
sign domains. Keep this negative finding rather than using the unsliced path
as the sole opponent.

## A bounded scalar certificate design for this example

The general [cell rule](margin_conditioned_support_contract_20261020.md)
can use 128 fixed closed cells of width h=1/64 on [-1,1]. Both 0 and -1/2
are cell boundaries. No numerical exponential or minimizer is needed here:

- for a negative cell [x,y], use the rational gate enclosure [1/2+x/4,1/2];
- for a positive cell, use [1/2,1/2+y/4].

These follow from sigma(0)=1/2, monotonicity and sigma'<=1/4. They cover the
true gate and lie inside the two anchors. For these exact support lines,
`A_l=S_l(y)` and `A_h=S_h(x)`. Replacing the two lines by their cell minima
loses at most h/4 in their interpolated value.

On the two outer sign/product regions the rectangle objective is at least
C-kappa. On a middle cell within [-1/2,0], it is at least
`C-kappa+x*(y+1/2)/4`. Since
`x*(y+1/2)/4 = y*(y+1/2)/4 - h*(y+1/2)/4`,
this is at least `C-kappa-1/64-h/8`. The cell rule therefore has the
paper guarantee

    B >= 3/256 - 3*h/8 = 3/512 > 0.

This is a finite symbolic design bound, not an executed certificate. The
128-cell choice applies to this supplied toy only, not as a guaranteed
complete discretization for a trained request or arbitrary sigmoid range.

## A lifted form that a stronger comparator can reproduce

The interpolation rule is not an irreducible new primitive. For l<h, take any
two valid support inequalities on P (l=h uses the fixed-gate branch above),

    b+l*d+nu*m >= c_l,      b+h*d+eta*m >= c_h.

For lambda in [l,h], multiply them by the nonnegative weights h-lambda and
lambda-l and add. With `Y=b+lambda*d` and `s=lambda*m`, this gives

    Y >= [h*c_l-l*c_h+(c_h-c_l)*lambda
          +(-h*nu+l*eta)*m+(nu-eta)*s] / (h-l).

This is exactly the interpolated pair of support lines. Taking all pairs of
available endpoint lines recovers the interpolation of the two maxima, because
both weights are nonnegative. Thus a comparator containing these lifted rows
and the same-strength treatment of `lambda=sigma(m)` and `s=lambda*m` can
reproduce the candidate. A loose rectangular relaxation of this new product
does not automatically reproduce it.

The candidate replaces the expert-dependent product with a remaining
one-dimensional nonlinear relation, not with universally linear computation.
Its potential contribution must be the useful, bounded construction and reuse
of such support evidence in HybridZ, tested against stronger product-aware
comparators; neither algebraic novelty nor superiority follows from this form.

## Comparison and implementation decision

The scientifically relevant new object is a conditional support envelope,
not the weighted-median heuristic on its own. Compare with:

1. zero tilts with the same anchors and scalar checker;
2. native H2/N1 with constant-objective and sign-stable reuse allowed;
3. the existing sign-sliced M3 and a predeclared stronger subdivision;
4. a semantically matched external nonlinear baseline when authorized.

Report objective slots, distinct goals, actual solves, checked bounds and
complete cost separately. Source scans, exact median sorting, source/dual
checking, scalar coverage and reception count even when no solve is needed.
No parameter sweep, LP-based tilt tuning, extra anchors or GPU belongs to this
single factor. The unchanged 300-second and resource contracts still apply to
any later executed stage, and the old observation failure remains unresolved.

There is now enough mathematical specificity to propose a small default-off
standard-library implementation/control stage, not to launch native or real
requests. Its finite roster should include the shared/private positive source,
the previously archived diamond negative example, and the equivalent-source
median failure; pair reversal, nonunique medians, constant gates, zero router
coefficients, missing cells, false sigmoid premises, incorrect sources and
deadline/capacity rejection must be covered. Actual roster, limits, reception
and authorization must precede execution; this is not a new supervision project.

Two AI read-only reviews checked the policy, orientation, source scope, relaxed
witness and stronger-baseline limits. They are not human review or executed
controls. This stage supports a finite mechanism test; it establishes neither
G1 completion, real same-source certification, external leadership nor GPU gain.
