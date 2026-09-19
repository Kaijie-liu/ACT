# Exact feasible witnesses: bounded construction research

## Outcome and scope

New optional namespace `exact_primal/` implements a bounded, solver-free
active-face reconstruction **candidate generator**, with analytic controls.
The unchanged `lp_sandwich/check.py` alone checks full rational feasibility and
objective. The generator never reports certified feasibility, network SAFE or
network UNSAFE. No actual network LP is reconstructed or optimized this turn.
The sealed four-query diagnostic remains unchanged and unresolved.

Final controls: [attempt003](exact_primal_controls_attempt003.json), **35/35 PASS**
(13 new controls plus21 LP component and1 archival regression). Attempt00133/33
PASS and attempt00234/35 with one failure are retained. Attempt002's new coupled
system fixture mistakenly used47/35 instead of67/35 for
3*(2/5)+5/7. The constructor correctly solved that different system. Only the
fixture was corrected, not the algorithm, tolerances or acceptance criteria.

## What a certificate must establish

Let the supplied, identity-bound rational LP be

\[
P=\inf\{c^Tx+d:Ax\le b,\ Ex=h,\ l\le x\le u\}.
\]

A witness is a rational vector x*. A separate checker recomputes every box,
inequality, equality and the exact objective U=c^Tx*+d. If all constraints hold,
P<=U follows simply because x* is an element of the feasible set. The method
used to construct x* need not be trusted. A bad active-set guess only hurts
completeness; it cannot make an invalid point pass the full checker.

- U<=0 establishes an obstruction to a strictly positive minimum of THIS LP.
- U>0 alone does not establish LP safety: other feasible points can be negative.
- A checked lower bound L, if separately available for the same objective/LP,
  yields L<=P<=U. Exact optimality requires checked L=U, not numerical closeness.
- An inconsistent guessed active system does not establish LP infeasibility.
- A feasible relaxed point is not necessarily any realizable network execution.

The existing statement binds request/source/export, pair/property and LP hash.
The candidate generator additionally records the original candidate hash and
selected rows. Construction history is provenance, not a premise needed for
checking x*. The isolated checker receives externally supplied statement and
bundle hashes; changing matrices or properties without changing that external
identity is rejected. Network→HZ, guard, route exclusion and F0 lowering are
still upstream assumptions. Nothing here closes that separate chain.

## Research options and chosen prototype

| Route | Benefit | Main unresolved requirement |
| --- | --- | --- |
| Reconstruct from an explicit native basis | Finite exact linear system; avoids guessing every active row | Preserve basis/slacks/scaling/presolve map to the original LP, handle singularity and sparse exact fill-in |
| Reconstruct from hinted active rows | Uses only existing primal candidates; suitable for bounded controls | Degenerate or wrong near-active rows can conflict; free-variable choices can violate unselected constraints |
| Exact LP solver / refinement | Can supply stronger candidate generation and exact optimization | Separate dependency, model export, resource and integration protocol; not installed or run here |

Exact LP solving, rational reconstruction and basis verification are established
methods, not a new theory claimed by this project. SoPlex documents exact
rational solving with iterative refinement/precision boosting and rational
factorization. See its [exact-mode documentation](https://soplex.zib.de/doc-7.0.0/html/EXACT.php)
and the primary paper [Linear Programming using Limited-Precision Oracles](https://optimization-online.org/wp-content/uploads/2019/12/7507.pdf).
These sources motivate possible future implementations; our small generator
is neither SoPlex nor an implementation of its refinement guarantees. No new
dependency or external solver was installed, integrated or benchmarked.

The current prototype takes the second route, with deliberately small bounds:
64 variables,512 supplied linear rows,8192 nonzeros,4096-bit rational values,
200000 counted rational operations, one attempt and an original absolute
deadline no later than300 seconds. These are control limits, not performance
claims. Large saved LPs are outside this implementation's supported size.

## Algorithm and acceptance separation

1. Validate the unchanged LP and external statement. Treat stored finite floats
   as exact binary rationals; never silently replace0.1 by1/10.
2. Include all equalities, all fixed variable bounds, then near-active box and
   inequality rows in stable order. An absolute1e-8 residual radius is only a
   *generation hint*. It is not a feasibility tolerance. This raw criterion is
   scale-sensitive and is not claimed to be invariant to row rescaling.
3. Solve that system by sparse rational forward elimination/back substitution.
   Consistent dependent rows need no extra pivot. Inconsistent rows end this
   attempt; there is no alternative-set search or silent row relaxation.
4. Keep unconstrained free coordinates at their original exact-binary candidate
   values. This is a candidate choice, not a guarantee of feasibility.
5. Emit `CANDIDATE_ONLY` and a standard LP bundle with the new rational point,
   exact objective and `dual=None`. LIMIT/TIMEOUT/ERROR or an inconsistent active
   system emits no witness. Retain the original LP and candidate unchanged.
6. Independently run the existing full checker on the bundle. Only a successful
   all-constraint check establishes an upper bound; the constructor's status
   and rank are not accepted as proof.

The prototype has cooperative deadline/size/bit/operation guards. It is NOT yet
an outer-supervised, fully costed real-request path; production integration
requires that separate work. Generating a candidate, packaging it, and checking
it must eventually share the real request clock without removing prior checks.

## Controls and examples

For3x=1, x in[-1,1], objective x-1, float1/3 is not exactly feasible. The
constructor generates rational1/3 and the relocated `python -I -S` checker
obtains **U=-2/3**, with no solver/model import. With no dual in this bundle,
it does not claim exact optimality. The test also rejects an altered claimed
objective even when the modified file hash is supplied.

Other controls cover a near-active inequality3x<=1, redundant equalities,
zero-width boxes, rank-deficient systems, nontrivial coupled pivots, unchanged
binary-float coefficient semantics, hash binding, expired clocks and resource
limits. Negative controls are substantive:

- The feasible narrow interval[0,1e-9] can make both bounds look active under
  the hint radius; the resulting contradictory system returns unresolved,
  never LP-infeasible.
- In a feasible LP x+y=1, x<=1/5, fixing free y at its candidate0.1 can produce
  a reconstructed x near0.9. The full checker rejects this unselected inequality.
- A checked positive objective1/3 supplies an upper bound but no safety proof.

These show why clipping, increasing tolerance, or checking only the selected
basis would be invalid replacements for full rational constraint checking.

## Next bounded development step

Do not apply this64-variable prototype to the archived LPs with thousands of variables or
raise its limits until a positive example appears. First design/control a
sparse basis/anchor interface: explicit row/column and slack identities,
original-coefficient reconstruction, rank/degeneracy behavior, fill-in/bit
caps, unchanged-LP binding and a full-check rejection path. A native basis is
an untrusted suggestion until those bindings and original constraints pass.

Only after that interface and an owned outer deadline/full-cost supervisor
pass controls should a new, separately frozen real diagnostic be considered.
Any new candidate must live in a new artifact identity, never overwrite the
sealed native point or relabel the previous four runs. No real effectiveness,
new certificates, speedup or complete-network guarantee is claimed here.
