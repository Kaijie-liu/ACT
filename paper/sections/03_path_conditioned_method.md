# Path-conditioned verification of routed models

Routed models are piecewise programs. Their output is determined not only by
the numerical values computed inside a neural network, but also by the path
chosen by a data-dependent dispatch operation. A conventional reduction first
proves that the router is invariant throughout the perturbation set and then
verifies the resulting static network. This reduction is sound, but it treats
route stability as a prerequisite even though the desired property concerns
the model output. A region can cross a routing boundary while every reachable
path remains safe.

Our verifier therefore conditions on routes instead of requiring one route to
remain fixed. Let (X) be an input perturbation set and let
(S(x)) denote a legal unordered top-(k) route at (x). The verifier computes
the feasible route family

\[
  \mathcal{S}(X)=\{S:\exists x\in X,\;S\text{ is a legal top-}k
  \text{ set at }x\}.
\]

For each (S\in\mathcal{S}(X)), it retains the weak route guard in the same
abstract factor frame as the input, propagates the corresponding expert path,
and proves the output property on that guarded branch. The region is certified
only if every feasible branch is certified. Router stability becomes a cheap
special case in which (|\mathcal{S}(X)|=1), not a soundness precondition.

## Tie-inclusive route semantics

Ties are semantic choices, not numerical nuisances. We use an
any-legal-top-(k) policy: if several sets can be selected at equal scores, all
of them must be covered. For a proposed set (S), legality is expressed by

\[
  r_i(x)\ge r_j(x)\qquad
  \text{for every }i\in S,\;j\notin S.
\]

The inequalities are deliberately weak. Replacing them by strict inequalities
would silently discard executions at a boundary; breaking ties according to a
single concrete library call would verify an implementation accident rather
than the registered semantics. Exact route labels are used only while the
reachable router hybrid zonotope has not undergone relaxation and every
feasibility query has been decided. Otherwise, the route family is a sound
upper set and unresolved branches remain `UNKNOWN`.

The implementation supports two complementary ways to construct this family.
Small routers can enumerate every candidate set and check its guard directly.
For larger (E), a single selector MILP uses binary variables (z_i),
(\sum_i z_i=k), and the implications

\[
  r_j-r_i\le M_{ji}(1-z_i+z_j).
\]

After each feasible set is replayed, a no-good cut excludes exactly that set
and the same loaded HiGHS model is solved again. Enumeration is complete only
when the cut-augmented model is proved infeasible. Time limits and numerical
failures do not become empty remainders. On the frozen E=8 top-2 all-tie
control, this procedure returns the same 28 sets as exhaustive enumeration,
using one model build, 28 cuts, and 29 solves including the final
infeasibility proof.

The implication constants can be derived from an unconditioned generator
bound, a constraint-aware LP support, or an integral HZ support. Every mode is
sound; the latter two retain path constraints and can tighten the encoding.
In the guarded correctness control, exact support reduces selector binaries
from two to zero. A failed support side falls back to the generator bound, and
the result is not labelled exact.

## Retaining a route condition

Router and expert abstractions share input generators and router constraints.
Expert-local ReLU binaries remain distinct, because two experts do not share
their activation choices merely because they receive the same input. This
identity discipline matters when subtracting expert outputs or transferring a
guard: duplicating shared input factors destroys correlation, whereas merging
independent expert binaries introduces executions that do not exist.

Adding a route guard intersects the current abstract domain with a subset.
Consequently, every exact support lower bound can only increase and every upper
bound can only decrease. The implementation exploits this monotonicity before
allocating ReLU binaries. It first propagates cheap bounds, then applies
constraint-aware support only to promising unstable preactivations, and
allocates a binary only if the guarded support still crosses zero. A solver
limit returns the unconditioned sound bound. It can lose an optimization, but
it cannot create a certificate.

This representation choice is empirically consequential. Retained-constraint
support eliminates binaries and improves paired solved coverage in the
confirmatory cohort. In contrast, replacing a high-dimensional box intersected
with a few route halfspaces by its coordinate hull recovers essentially no
guard information, and compiling the guard into a relaxed output property does
not recover certificates on the frozen adapter cohort. The comparison does not
show that guards are intrinsically powerful; it shows that their value is
coupled to a representation capable of retaining them.

### Compiling a hard-top1 guard into a static backend

Some static-network verifiers cannot accept input-side route halfspaces. For a
hard-top1 branch \(i\), write its tie-inclusive guard as
\(m_{ij}(x)=r_i(x)-r_j(x)\ge 0\) for all \(j\ne i\). For safety row
\(s_\ell(x)\ge0\), choose fixed nonnegative multipliers
\(\mu_{\ell j}\) and compile the shared-input router and expert graph to

\[
  \phi_\ell(x)=s_\ell(x)-\sum_{j\ne i}\mu_{\ell j}m_{ij}(x).
\]

On a legal branch, \(\phi_\ell(x)\le s_\ell(x)\). A sound lower bound
\(\phi_\ell\ge0\) over the original box is therefore a sufficient proof of
the expert property on the guarded cell. A tied competitor contributes zero,
so it cannot discharge its own branch obligation (although margins to other
competitors can still make the sufficient condition conservative). It also
avoids the unsound shortcut
\(\max(g_i,s_i)\ge0\), which can pass vacuously when \(g_i=0\).

The multipliers may be selected per property row from a finite, preregistered
grid: taking the largest of independently sound lower bounds is sound. A failed
compiled bound remains `UNKNOWN`. This is a standard Lagrangian sufficient
reduction specialized to routed programs, not an exact representation of the
guard and not a claim of a new generic Lagrangian method. Its possible benefit
comes from preserving router--expert input dependence in one graph; separately
intervalizing the two sides would discard that dependence. Our current CROWN
implementation records only a numerical filter because that backend lacks an
outward-rounding contract.

## A staged verifier for normalized weighted gates

For normalized non-negative gates, the first tier avoids gate modelling. If
every feasible selected expert satisfies a convex linear property on its
guarded domain, then every convex combination of those expert outputs satisfies
the same property. This gate-elimination rule is cheap and exact as a
sufficient condition. Its failure is not evidence of an unsafe weighted model:
one expert can violate a row while its reachable weighted mixture remains
safe. Tier 1 therefore returns only `SAFE` or `UNKNOWN`.

When Tier 1 is inconclusive, Tier 2 models only the part of the gate needed by
one property row. Choose an anchor expert (b\in S). Every normalized top-(k)
output can be written as

\[
  F(x)=E_b(x)+\sum_{i\in S\setminus\{b\}}
  \lambda_i(x)(E_i(x)-E_b(x)).
\]

For a linear safety row (q^TF+c\ge0), define
(u=q^TE_b+c) and (d_i=q^T(E_i-E_b)). The verifier needs only the
(|S|-1) scalar products (\lambda_i d_i), rather than a nonlinear encoding
of every output coordinate. Sound gate intervals, guarded difference supports,
McCormick envelopes, and the intersection of the gate box with the simplex
form an outer relaxation. A strictly positive, outward-corrected lower bound
proves the row. A non-positive relaxation result remains `UNKNOWN` unless a
concrete input is recovered and violates the full routed model on replay.

For selected-softmax top-2, the construction reduces to

\[
  F=E_b+\lambda(E_a-E_b),\qquad
  \lambda\in[\sigma(\underline m),\sigma(\overline m)],
\]

where (m=r_a-r_b) is bounded under the pair guard. The F0 fallback uses this
numeric gate range and one property-directed McCormick product. It encodes no
exponentiation, division, or sigmoid segments. In the independent route-
boundary cohort, F0 resolves 43 of 60 Tier-1 semantic incompleteness cases,
including 31 additional certificates and 12 full-model-replayed violations.
The alternate top-3 normalized-sigmoid execution exercises the general
(|S|-1)-product implementation; it is mechanism evidence rather than a
scalability result.

An optional margin-segmented refinement preserves additional correlation by
conditioning the expert-difference support on affine router-margin intervals.
It resolves the relaxation-limited control but gives only two additional
solutions on a 20-row residual cohort dominated by solver limits. We retain
this null-adjacent result: segmentation addresses relaxation width, not MILP
search throughput.

## Verdict discipline and backend composition

The staged analysis exposes four terminal states. `SAFE` requires every legal
route branch and every property row to have a certified positive lower bound.
`UNSAFE` requires a concrete input that replays through the complete dynamic
model, selects a legal route, and violates the property. `TIMEOUT` records an
explicit budget limit; all other incompleteness is `UNKNOWN`. A per-expert or
relaxation witness is never promoted directly to `UNSAFE` for a weighted model.

Route conditioning is independent of the downstream expert verifier. Exact HZ
and MILP backends provide the correlation-preserving reference path at
verification scale. A routed model that a general-purpose frontend rejects can
also be specialized into static per-route programs and delegated to a CROWN
backend. The contribution is therefore an analysis layer for dynamic dispatch,
not a claim that one expert backend dominates existing neural verifiers.
