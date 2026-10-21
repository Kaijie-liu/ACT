# B1 same-source diagnostic design — not an execution freeze

Status: **DRAFT_NOT_EXECUTABLE** at `8534f1dc8`. No new model, clean-forward,
solver or attack ran during this review. B1–B4's real-stage ledger remains0/0/0.
This design closes a specific experimental-design gap: existing scalar records
cannot distinguish an unfinished search from a weak relaxation. It does not
introduce a new verifier, mechanism or claim of algorithmic gain.

## The question and the objects that must not be conflated

For each registered complete request, retain the full route/property denominator
and ask what the evidence actually establishes about its unresolved obligations.
Causes may overlap or remain unseparated; do not force every row into one cause.

| Object | Meaning | Allowed comparison |
| --- | --- | --- |
| Captured guarded expert/router HZ | Original stored coefficients and factor identities from this invocation | Gate-relation diagnostics on this bound source; upstream conversion remains a stated assumption |
| Captured F0 HZ and its native lowered matrix | Actual product relaxation, gate/difference bounds, lowering and objective used online | Native MILP versus its continuous relaxation, with identical stored matrix, bounds and objective |
| Exact rational reconstruction of a stored object | Binary64 coefficients interpreted as exact rationals, with explicitly checked transformations | Independent bound/point checks on that object, not an automatic proof of native floating-point lowering or the original network |

In particular, releasing integrality in the *same native matrix* isolates one
factor. Repropagating a different HZ, changing support ranges, or exporting a
rounded reconstruction does not isolate integrality. Matrix/objective identity
must accompany each comparison, not merely a matching input index or frame ID.
The relation between the HZ and native lowering must be recorded separately.

## What the saved production records lack

`staged_verifier._run_f0_impl` stores scalar support ranges, solver status,
primal/dual objective numbers and full-model replay results. It does not retain
the original factor assignment, dual vector, complete source matrix or values
of margin `m`, expert difference `d`, base `b`, gate `lambda` and product `w` at
one assignment. A scalar `mip_dual_bound` is not a checkable dual certificate.
Timeout returns may censor partial rows; an empty row list is not zero obligations.

The same issue applies to `paired_monolithic.run_monolithic/_run_monolithic`. Adaptive's
single-pair path also uses that implementation. A combined disjunction's
nonpositive bound is not evidence that one particular pair is the sole blocker.

## Minimal optional capture, no new supervisor

Two narrow interfaces are needed before any real diagnostic is frozen:

1. In the existing F0 invocation, associate the actual `guarded_entry`,
   `conditioned_router`, `propagated.joint`, complete property `(q,c)`, gate and
   difference ranges, their generation statuses and source/factor maps.
2. At native solver return, before raw values are discarded, emit a bounded
   record containing the stored matrix/objective/variable bounds/integrality,
   native candidate, status and lowering maps. Monolithic capture also includes
   selector variables, local offsets and the complete residual pair roster.

Candidate hooks would be default-`None` in `hz_minimize_output` (passed through
`solve_weighted_top2_f0`) and `solve_monolithic_weighted_top2_f0`. No default-path
copy or query is added. The sink must not mutate solver arrays, change a decision
or invoke a solver. Its deadline/capacity exceptions must not be caught by the
native solver's broad numerical-error handler. Source copying, serialization,
checking and reception all consume the same request deadline and Work ledger.
Freeze and test this interface's actual failure paths before using real inputs.

Reuse the existing exact source/property contracts, not the stopped strategy:

- `margin_hz_source.bind_source` recombines experts. Any reuse must check the
  recombined coefficients/constraints/maps against this invocation's actual
  `propagated.joint`; equal frame names alone are insufficient.
- Real router equality/inequality row maps must be exported and checked. The
  synthetic control's `(1,)` map is not a real-source mapping.
- That interface explicitly relaxes binary factors. Preserve the distinction
  from production's `beta=2*z-1`, `z` binary representation.
- Do **not** call `native_margin_support.support_property` for B1: it recomputes
  a box margin range and runs the stopped fixed-tilt policy. Reusing its checker
  and exact property view does not authorize rerunning A2.

## Prospective finite roster and accounting

The following is a slot design, not selected indices or permission to launch an
unfrozen worker. First finish the dataset-aware history manifest; use ascending
eligible identities rather than route counts, positive bounds or easier cases.
Disclose prior full-test telemetry and fixed-recipe author test evaluation.

| Proposed allocation in one B1 stage | Maximum charged call slots |
| --- | ---: |
| First12 preregistered candidates, each clean-forwarded on all3 frozen MLPs | 36 |
| First2 common-clean-correct inputs in that fixed order, all3 MLPs, complete diagnostic-enabled request | 6 |
| At most2 separately numbered diagnostic calls for each model/input | 12 |
| Total | 54 |

No checkpoint load, input materialization or selection forward is free. Retain
all attempted, failed and rejected candidates in the ledger. Do not add a13th
candidate or replace inputs using verification outcomes. If fewer than2 qualify,
record the selection shortfall and stop at the preregistered gate rather than
extending the prefix. This design leaves6 slots unused, not an automatic retry
allowance. The stage still has its independent6h and128MiB limits;54×300s does
not entitle execution past either limit. B2 needs its own frozen allocation,
including actual deployment cost; it does not borrow this stage's remaining slots.

The complete-request observer keeps an immutable ordinal roster of all legal
pair/property obligations and records `REUSED`, `ATTEMPTED`, `RETURNED`,
`SELECTED`, `NOT_REACHED` or `CENSORED`. A complete roster requires completed
route enumeration; otherwise record `ROUTE_COVERAGE_INCOMPLETE`, preserving
the known subset without calling it the complete denominator. It may capture at most one source per
request: the first returned unresolved obligation in the preregistered order,
not the most negative row. For a monolithic property, choose its residual pair
by a fixed ordinal rule, not by post-hoc active-branch convenience; preserve the
other pairs as unexamined. No export before cutoff means missing evidence, not
a replacement request. Diagnostic-enabled timing is not uninstrumented verifier
timing, and B2 comparisons must label/pay the instrumentation consistently.

Before execution, the two diagnostic recipes must have fixed internal query
tables (LP/MILP integrality comparison; gate-slice reference), time allocations,
source reception bounds, selector semantics and a terminal coverage audit.
They may reconstruct or consume a saved source only with that cost charged and
identity checked. They are separately numbered diagnostics, never continuations
that convert an old timeout into online success. No query receives its own300s
in addition to its parent request. Do not freeze a recipe that cannot identify
its original matrix or complete cost boundary.

## What would count as an attribution

| Evidence | Supported conclusion | Not supported |
| --- | --- | --- |
| Solver budget ended; no checked bound/point | This execution did not finish its proof | The relaxation is intrinsically too weak |
| Checked nonpositive lower bound | This certificate did not close the obligation | The relaxation optimum is nonpositive or the model unsafe |
| Exact feasible point with nonpositive objective in the continuous F0 relaxation | That particular outer relaxation admits a violation | The native MILP, true gate relation or full model is unsafe; gate is the unique cause |
| Same stored matrix: checked positive MILP bound and exact nonpositive continuous feasible point | Integrality separates these two diagnostic problems | A different source or every unresolved request has the same cause |
| Bound improves under a source-matched gate refinement | The registered refinement helps this obligation | All remaining error comes from the gate, or the refinement is universally better |
| Concrete input passes complete-model replay | UNSAFE under the registered replay policy | A mathematical proof of deployed floating-point semantics |

The table's checked MILP bound is an evidence requirement, not a capability
already provided by the scalar SciPy `mip_dual_bound`. A native numerical MIP
bound alone retains its numerical-policy grade; do not report it as independently
checked integrality evidence.

`check_hz_representation.check_point` can check a supplied LP point exactly.
Most native equality solutions need not pass. Rejection leaves the attribution
unseparated; it does not authorize reopening exact-elimination searches to repair
the point. Numerical `m,d,b,lambda,w`, binary fractionality and residual terms
remain useful leads, clearly distinguished from checked feasible evidence.

B1's output is a full-denominator, possibly multi-label attribution table with
missingness and costs. It is not a new SAFE table or a choice of the next
algorithm in advance of the evidence. Proceed to a B3 mechanism only when this
table identifies a concrete, testable gap; no additional tilt/slice tuning is
authorized by this design document.
