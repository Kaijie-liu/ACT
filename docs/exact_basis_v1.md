# Original-coordinate sparse basis and anchor interface

## Delivery and boundary

`exact_basis/` is a separate analytic research interface. It reconstructs an
explicitly supplied basis/anchor system from the original rational LP, with
no near-active tolerance and no alternative-basis search. It leaves the old
`exact_primal/`, LP checker, execution freezes and all real outcomes unchanged.

[Controls attempt001](exact_basis_controls_attempt001.json): **47/47 PASS**,
12 new tests plus35 prior regressions. Zero real LP reconstructions and zero
real solver calls. Analytic cases use declared bases, not captured native bases.
No solver was installed or integrated. This is not a performance result and
not a scalable exact solver; its limits are intentionally small.

## Original-coordinate contract

For Ax<=b, Ex=h, l<=x<=u, define slack s=b-Ax and z=(x,s). The augmented
equations are

\[
Mz=q,\qquad
M=\begin{bmatrix}E&0\\A&I\end{bmatrix},\quad
q=\begin{bmatrix}h\\b\end{bmatrix}.
\]

The original inequalities are equivalent to the A equations together with
s>=0. Merely satisfying the augmented equations does NOT establish feasibility.
Bounds on x still apply. Original LP checking therefore remains mandatory.

A manifest binds the original LP, statement and candidate hashes, and declares:

- Every original E/A row, exactly once, with kind/index and a hash of its exact
  coefficients/RHS. A permutation is allowed; omission, duplicate or substituted
  rows are rejected. No copied or approximate coefficient payload is accepted.
- Exactly m distinct basic columns, where m=#E+#A. Columns are named x:index or
  slack:A-row-index, not ambiguous positions in a transformed solver matrix.
- Every remaining column, exactly once, with a declared anchor: x at its exact
  lower/upper bound or original candidate value; nonbasic slack at zero.
- A fixed positive-slack sign convention. Presolved/scaled coordinate labels,
  negative-slack conventions and unexplained anchor values are rejected.

Basic and anchored columns form a disjoint cover of all n+#A coordinates. A
candidate-valued x anchor need not lie at a bound: this is a linear-algebra basis
construction and not necessarily a simplex basic feasible solution or vertex.

Given a basis B and anchors z_N, the generator constructs the exact system

\[
M_Bz_B=q-M_Nz_N.
\]

For nonsingular M_B, exact elimination yields the unique basic values for those
anchors. That proves only the selected equation identity, not bounds, slack
nonnegativity, optimality or any network property. The final original-LP
checker recomputes ALL original constraints and objective using the resulting
x. It neither trusts the mapping nor needs basis history to prove x feasible.

## Sparse and resource behavior

The new generator uses sparse dictionaries throughout assembly/elimination;
there is no dense fallback. It records inserted fill-in entries, peak stored
pivot-plus-current-row nonzeros, rational operation count and elapsed time.
The nonzero statistic is an elimination working-set count, **not total process
RSS**; original/system dictionaries and temporary objects are not included.

Current caps:64 original variables,64 augmented equations,8192 input nonzeros,
4096 live elimination nonzeros. It reuses the unchanged earlier4096-bit and
200000-operation arithmetic guards. One attempt and one supplied absolute
deadline, no later than300 seconds. Limits do not certify large-LP scalability.
No earlier module's size cap is raised.

Row pivots are chosen deterministically during exact forward elimination.
Singularity or dependent/inconsistent selected rows returns
`UNRESOLVED_SINGULAR_BASIS`, not infeasibility of the original LP. Redundant
original equalities can prevent this square-basis contract from succeeding;
they are not silently removed. Any row-elimination/presolve support would need
its own original-coordinate reconstruction contract.

The only successful constructor status is `CANDIDATE_ONLY`, with a standard
LP_SANDWICH_V1 bundle, rational primal and `dual=None`. The frozen standalone
checker establishes a legal U only when full feasibility holds. Negative basic
slacks are retained in the proposal: the checker rejects the corresponding
original inequality violation. No clipping or tolerance relaxation is allowed.

## Analytic evidence

For x+y=1 and3x<=1, with x,y in[0,1] and objective x-1, choosing x,y as basic
and slack0 as the anchor reconstructs (1/3,2/3). A relocated standard-library
`python -I -S` check obtains **U=-2/3**, with no solver/model import. Permuting
the manifest's rows and basis columns produces the identical original point.
The constructor itself never certifies that result.

Choosing x and the slack as basic, but anchoring y=0, instead gives x=1 and
slack=-2. The original-LP checker correctly rejects the point. Choosing y=1
gives a feasible different point. These tests separate correct linear algebra
from actual feasibility and avoid treating a basis solve as sufficient proof.

Controls additionally cover fixed variables, exact candidate-valued anchors,
empty bases for box-only LPs, rank deficiency, explicit fill-in cap exhaustion,
deadline/size rejection, coordinate omission/overlap, row omission/hash drift,
unmapped transforms, slack sign mismatch and mutation of the emitted point.
No new real certificate or conclusion about the four sealed LPs follows.

## Relation to public solver interfaces

Exact basis factorization is established numerical verification machinery, not
new theory. [SoPlex](https://soplex.zib.de/) documents exact rational factorization
and refinement. [HiGHS' API](https://ergo-code.github.io/HiGHS/dev/interfaces/c_api/)
exposes basis-related operations, but availability of a basis API does not
validate a particular adapter's row/slack/scaling interpretation. This release
does not call or claim compatibility with those APIs.

## Next bounded step

First build a tiny **native-basis adapter control**, separate from real LPs:
bind solver/version/options and the exact submitted model, map structural and
row/slack statuses into this original-coordinate manifest, and reject missing
or unexplained presolve/scaling mappings. Include row/column permutation,
redundancy/degeneracy, fixed-variable and sign-convention controls. Native status
remains untrusted; the full rational checker decides acceptance.

Then integrate capture→mapping→exact construction→packaging→full checking into
an owned outer supervisor with a single request budget and complete cost
accounting. These are prerequisites, not delivered production capabilities.
Only after those controls should a separately frozen real diagnostic be
considered. Do not raise limits, requery the sealed four jobs, overwrite old
points, or infer network UNSAFE from a feasible point in an LP relaxation.
Network→HZ, guard, route exclusion and F0 lowering trust remain unchanged.
