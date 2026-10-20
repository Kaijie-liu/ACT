# Margin conditioned support and certified scalar aggregation

2026-10-20, static research at `31c3ccde12318a4b66f266270bf0ba7971707c2b`.
This candidate uses tilted support queries on the existing guarded HybridZ to
approximate a margin-conditioned value function. Its weak-dual lower-bound rule
is sound; its finite-budget benefit and novelty remain unestablished. G1–G6 OPEN.

The user approved continuing the mathematical contract and direct comparison
design after the preceding review. This stage changes documentation only: no
candidate implementation, control batch, native import, model/solver/GPU run,
GenBaB adaptation or resource-observation policy change. Prior failed attempts
remain failed. The external proposal is preserved, not edited:
`../survey/12_margin_conditioned_support.md`, SHA-256
`806d9912000c63f601f65b609e5681d1ad2c82864cb931fe497244f3f08cbc46`
(path relative to the ACT repository).

## Scientific difference from existing ACT

[F0](../act/back_end/moe/weighted_top2.py) keeps the expert base and difference
on shared factors. [H2](../act/back_end/moe/hz_endpoints.py) jointly supports
constant-gate expert combinations. [N1](../act/back_end/moe/conditioned_difference_support.py)
actually appends closed margin inequalities, not merely interval summaries.
[M3](gate_margin_rows_results_20261020.md) adds gate–margin rows; its archived
synthetic observation is not an accepted online experiment or a new envelope
theorem. These are necessary comparators, not mechanisms to rename.

The candidate difference is to reuse a tilted support bound at many margin
values, then certify a scalar lower envelope using the gate function. The
existing [support interface](../act/back_end/moe/batched_support.py) already
accepts arbitrary rational objectives; adding a term `nu*m` is not itself a new
support kernel. No explicit gate-product variable is introduced, but nonlinear
scalar minimization and high-dimensional support work remain.

## Object and orientation contract

For a canonical pair `(a,b)` and property `(q,c)`, use ONE compact guarded factor
set P and affine maps

    m(z) = r_a(z) - r_b(z)
    d(z) = q^T(E_a(z) - E_b(z))
    b(z) = q^T E_b(z) + c
    f(z) = b(z) + sigma(m(z))*d(z).

The gate is the selected-softmax weight of expert a. Source identity must bind
P, original coefficient bits, shared/private factor maps, router, pair, property
including its constant, margin coverage, and the relaxation/integrality choice.
An LP result is over the declared continuous relaxation; it is not an exact MILP
support result. A mathematical statement about supplied P does not repair
network lowering, input inclusion or route exclusion.

Let the finite rational interval I=[Mlo,Mhi] contain m(P). Empty fibers have
value +infinity. If P is empty,
handling that case requires its own checked evidence; the construction does not
infer emptiness from missing candidates. For feasible fibers define

    psi(mu,t) = inf { b(z)+t*d(z) : z in P, m(z)=mu }
    Phi(t,nu) = inf { b(z)+t*d(z)+nu*m(z) : z in P }.

Then inf_P f = inf_{mu in I} psi(mu,sigma(mu)). The equality is a conditional
decomposition, not a removal of the inner support problem.

## Checked lower bound rule

Freeze strictly increasing rational anchors t_0,...,t_J covering sigma(I), and
finite nonempty sets V_j of finite rational tilts. Include zero when comparing
against the corresponding zero-tilt rule. For every query require a checked
finite rational L_jnu <= Phi(t_j,nu), and set

    Schecked_j(mu) = max_{nu in V_j} (L_jnu - nu*mu).

For t=sigma(mu) in [t_j,t_{j+1}], set

    theta = (t_{j+1}-t)/(t_{j+1}-t_j)
    G(mu) = theta*Schecked_j(mu) + (1-theta)*Schecked_{j+1}(mu).

For any z in P with m(z)=mu, weak duality gives
`b(z)+t_j*d(z) >= L_jnu-nu*mu`. Take the maximum at each anchor and the
nonnegative interpolation weights. Since the objective is affine in t,
`f(z) >= G(mu)`. Thus every checked B <= inf_I G is a lower bound on inf_P f.
No optimal solver status is needed for this mathematical implication; this
does NOT change the production numerical acceptance policy.

Weak duality holds with binary factors too. Strong dual equality with psi is
claimed only for a continuous LP with a feasible conditional fiber and finite
optimum. For binary P, the tilt supremum can leave a convexification gap.
Neither a failed conditional solve nor a nonpositive bound proves infeasibility
or a full-model violation.

With two anchors and zero tilt, G is an interpolation, not pointwise the
constant `min(Phi(l,0),Phi(h,0))`. Its global inf equals that minimum when the
sigmoid image over I reaches both anchors AND L equals Phi at both anchors.
With inexact certificates the corresponding equality instead uses `min(L_l0,L_h0)`.
Outward-loose anchors may yield a different scalar minimum. Separate that effect
from nonzero tilts. M1 uses separate expert facts rather than joint Phi, so it is
a cheaper lower-bound oracle, not an equality with joint support.

## Error components and what can be certified

For each fixed feasible mu, psi(mu,t) is concave in t, even for nonconvex P.
If d is uniformly in [dminus,dplus], then for v>u,

    dminus*(v-u) <= psi(mu,v)-psi(mu,u) <= dplus*(v-u).

Let Delta=h-l, A=t-l, B=h-t, and J be the linear interpolation of the exact
values at l,h. Concavity and the two slope inequalities give

    0 <= psi(mu,t)-J
       = (B/Delta)*(psi(mu,t)-psi(mu,l))
         + (A/Delta)*(psi(mu,t)-psi(mu,h))
      <= (A*B/Delta)*(dplus-dminus)
      <= Delta*(dplus-dminus)/4.

This is a bound on ANCHOR interpolation only. It needs a valid difference range;
it is not a bound on finite-tilt or solver error.

Define Sexact_j=max_nu(Phi(t_j,nu)-nu*mu). The pointwise gap decomposes into:

1. anchor chord error between psi at sigma(mu) and its exact anchor chord;
2. finite-tilt gaps `psi(mu,t_j)-Sexact_j(mu)`, interpolated between anchors;
3. support-certificate gaps `Sexact_j(mu)-Schecked_j(mu)`, also interpolated.

A lower certificate alone does not give a numerical upper bound for components
2 or 3. Checked feasible points for each global support query can bound component
3 by `max_nu(U_jnu-L_jnu)`. A point at one conditional mu cannot bound component
2 over the whole interval. Without such evidence, report those gaps as unknown.

A separate scalar certificate can enclose `inf_I G` within eps_scalar. A total
error claim about inf_P f additionally needs uniform bounds on the above gaps
and feasible fibers throughout the minimized domain (or checked removal of
infeasible parts). An outward margin interval may include unreachable mu;
minimizing there is sound but can add an unquantified domain gap. LP relaxation
versus the original binary/network object is another separate gap.

One possible certificate for the combined tilt/support error is a feasible
conditional path. For continuous convex P, checked feasible points at margin
nodes can be linearly interpolated: the interpolated point stays in P and has
the interpolated margin. At fixed t_j its objective is an affine upper bound
U_j(mu) on psi. Check `U_j-Schecked_j` over each segment, including all envelope
breakpoints, to obtain a uniform eps_conditional. Checking only segment endpoints
is insufficient. This certificate is optional for proving a positive lower
bound, and is not yet constructed. It must cover the whole minimized domain;
binary assignments cannot generally be interpolated this way.

If that coverage holds and `B <= inf_I G <= U` with `U-B <= eps_scalar`, then

    0 <= inf_P f - B
      <= max_j(Delta_t_j*(dplus-dminus)/4)
         + eps_conditional + eps_scalar.

Here eps_conditional already includes tilt AND support loss; do not add the
same support error again. These are bounds for the declared P, not a precision
guarantee for a different network/source object.

Even for continuous P, convexity of psi in mu at fixed t does not make G or f
convex. For example, m=z in [0,1], b=0,d=1-z gives affine
`psi(mu,t)=t*(1-mu)`, but `f''(0)=-1/2`. No ordinary convex Kelley convergence
or fixed number of refinement rounds follows.

## A finite checkable scalar algorithm

This is a design, not an implemented or executed checker. It avoids trusting a
local optimizer, grid minimum or numerical inverse sigmoid.

Freeze finite limits on cells, rational bit lengths, sigmoid-enclosure work and
the common deadline/Work ledger. Closed rational cells C=[u,v] cover ALL of I.
For each cell:

1. Check an outward rational enclosure T(C) of sigma(C).
2. Compute
   `A_j(C)=max_nu{L_jnu+min(-nu*u,-nu*v)}` for every anchor.
   This bounds Schecked_j from below throughout C, including for negative values.
3. For every nonempty exact intersection
   `T_j=T(C) intersect [t_j,t_{j+1}]`, minimize
   `theta(t)*A_j(C)+(1-theta(t))*A_{j+1}(C)` at the two endpoints of T_j.
4. Take the minimum over those bands to obtain B_C, then B=min_C B_C.

Nonnegative interpolation weights prove `B <= inf_I G`. No uncertain band,
boundary or cell may be dropped; an out-of-coverage gate enclosure is rejected
or soundly intersected with a separately checked global enclosure, not clipped
without evidence. Sigmoid enclosures need independently checked analytic or
rational evidence, not `float sigmoid + nextafter` relabeled as exact.

For scalar error estimation, a checked upper enclosure for G at any point of I
gives U >= inf_I G. If its gate enclosure crosses anchors, include every possible
band when forming that upper enclosure. `U-B <= eps_scalar` only encloses the
surrogate's infimum: U is NOT a feasible upper bound on inf_P f or a counterexample.
An infeasible margin point can still be used for this surrogate-only calculation.

For fixed finite cuts and positive anchor widths, G is continuous. If the maximum
cell diameter tends to zero, all sigmoid-enclosure errors uniformly tend to zero,
and U is the minimum checked point upper bound over a dense sampling such as
one point per cell,
the scalar enclosures converge in the unbounded-refinement mathematical limit.
Refining only a selected subset does not establish that premise. No claim of
completion within frozen bit, cell or 300-second limits follows. Exhaustion
preserves partial evidence and returns incomplete; it never deletes the worst cell.

## Analytic controls and limits of the hypothesis

These are paper derivations, not solver observations, trained-model certificates
or a frozen execution roster. They test different claims.

### Upper endpoint screening can discard a useful candidate

Take P=[0,1], m=z,d=1,b=2/5-z, l=1/2,h=sigma(1). The global minimizer of
`b+h*d` is z=1, where the gate equals h. Nevertheless H2 is -1/10.
With nu=1, Phi(t,1)=2/5+t and the tilted line equals psi(mu,t).
Interpolation recovers f; its derivative is at most -3/4, so its minimum is
`sigma(1)-3/5>0`. The proposed high-endpoint-only stop test would reject this
case. Existing slicing can also prove it; this is not a superiority example.

The no-improvement conclusion requires more: a proven global endpoint optimizer
whose actual gate equals that endpoint AND that endpoint attains the H2 minimum.
A gradient search point or an unquantified "near" endpoint does not suffice.

### Tilts separate a fixed two slice endpoint roster

Take P=[-1,1], m=z,d=1,b=-z/4-3/8, with common gate envelope [1/4,3/4].
Use its two anchors and V={0,1/4} at each: four supports.
Phi(t,1/4)=t-3/8, so G equals f. Its minimum is
`sigma(1)-5/8>1/24`, using e>2 and sigma'<=1/4.

For ANY two closed slices split at c, even granting tight sigmoid ranges:
if c>=0, the first slice's low endpoint has support at most
`sigma(-1)-3/8<0`; if c<0, the second has support
`sigma(c)-5/8<0`. These are also four support queries.
This is a separation from that specific two-slice rule, not from M3 or GenBaB:
ordinary sigmoid envelopes may already prove it. Obtaining and checking gate
ranges and scalar minimization are not free costs in a future experiment.
Also d is constant here: endpoint objectives differ only by a constant, so a
strong native comparator may reuse the same dual and spend its saved query
budget on more slices. The example does not establish four-query superiority
against that comparator. Constant-objective reuse must be available fairly on
both arms, and support counts must not be inflated with dominated duplicates.

### Slices can beat arbitrarily many tilts with fixed anchors

Take `P={(m,v): |m|+|v|<=1}`, b=3/16-v/2,d=v, and gate anchors 1/4,3/4.
Here `psi(mu,t)=3/16-|t-1/2|*(1-|mu|)`. Even if infinitely many tilts recover
both anchor value functions exactly, interpolation gives

    G(mu)=3/16-(1-|mu|)/4,    inf G=-1/16.

Four closed margin slices with endpoints -1,-1/2,0,1/2,1 and valid gate
enclosures `[1/4,1/2]`, `[3/8,1/2]`, `[1/2,5/8]`, `[1/2,3/4]`
give eight endpoint supports with minimum 1/16. The ranges follow from e<3,
symmetry and sigma'<=1/4. This is a dyadic given-source case, without an
irrational stored coefficient. It disproves fixed-two-anchor dominance even
with a generous tilt budget, not usefulness of adding interior anchors.

Constant gate, equal experts, reversed pair orientation, nonzero property
constant, empty/uncertain fiber and retained binary factors are additional
semantic cases for a later control roster. They are not executed here.

## Integration and fair comparison contract

Use the original shared factors and retained constraints. If a bound output view
is `[E_a,E_b,r_a,r_b]`, the query is `(t*q,(1-t)*q,nu,-nu)` with offset c.
Alternatively construct those factor coefficients directly by rational arithmetic
on original stored coefficients. Router constants and property c must be counted
exactly once. Bind and independently reconstruct the router-to-joint factor map;
do not concatenate duplicate constraints or project in floating arithmetic and
call the result the unchanged exact source. No forced shared dual across targets.

The existing batch API admits at most 128 factors, 128 outputs, 256 constraints
and eight queries, and relaxes binary factors. Its checker proves bounds on a
given HZ continuous relaxation, not network conversion, router mapping or complete
MoE coverage; its helper is not a whole-request hard supervisor. This contract
does not grant a new capacity or automatically compose separate Work ledgers.
In particular this API takes a deadline, not the request store's Work ledger
or its integer-bit contract; returned row/entry counts are not quota debits.
The output-view construction also needs `2*classes+2 <= 128` in this version.
Current capture and per-objective validation copy/scan sources; preserving P
does not mean zero-copy execution or free repeated checking.

| Comparison | Controlled difference | Required accounting |
| --- | --- | --- |
| Zero tilt versus finite tilt | Same anchors, P, ranges and scalar checker | All support attempts and certificates |
| Anchor refinement | Same tilt-selection rule; separate factor | New anchors and scalar work |
| Closed-slice H2 and N1 | Same source and common numerical guarantees | Range queries, sliced constructions, all supports |
| Existing M3 | Same source; its joint row/product relaxation | Row construction, solve and checking |
| GenBaB later | Whole-box/guarded/route aggregation labeled separately | Actual full time and hardware, not ACT Work |

Support-call count is a mechanism metric, not the complete cost measure.
Any conditional LP used to choose a tilt counts, including failed calls.
If comparisons allocate the same query cap, the allocation must be fixed before
results and includes those auxiliary queries. Shared setup is either charged
independently or explicitly separated on both arms. Full cost includes source
capture, propagation, rational objectives, candidate generation, sigmoid proof,
scalar coverage, serialization, independent acceptance and cleanup.

The finite-budget hypothesis is that a preselected strategy yields a practically
better checked-bound/cost tradeoff than reasonable native comparators on its
declared population, not universal dominance. A future complete-request test
must also report all pair/property coverage and SAFE/UNSAFE/solved gain and loss.
Retaining a valid baseline bound can prevent arithmetic regression on that
obligation, but does not guarantee zero complete-request loss after paying costs.

## Decision and next boundary

Continue this candidate at the contract level: a nonduplicate finite-support
organization and a fixed-roster separation exist, and its soundness does not
depend on the failed RSS observer. The stronger constant-reuse and sigmoid-row
comparators still prevent interpreting that example as an established advantage.
Do not use the proposed gradient probe's negative result to infer no coupling.
Do not claim novelty or real benefit from the analytic example: stronger sigmoid
rows and external branching can realize part or all of the same gain.

Before implementation, specify one bounded anchor/tilt strategy, the scalar
checker certificate and its limits, and a finite comparison/fault roster.
Native execution and any resource-contract change remain separate decisions;
the user's earlier refusal of G4 adapter execution is not reversed here.
Stop or narrow this mechanism if it merely reproduces strong native results at
higher full cost. No further observer repairs, real-input screening, new model,
GPU run or consumed-attempt retry is implied.

Two separate AI read-only reviews checked the slope proof, scalar coverage,
source-interface boundary and analytic comparisons. They are mathematical/source
reviews, not executed controls, human technical review or independent network
proofs. The new evidence in this stage is the error decomposition and explicit
positive/negative mechanism examples, not documentation volume.
