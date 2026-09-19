# Four unchanged-LP diagnostics: executed and sealed

Execution HEAD `9fdb77beac9c82530a2a1114a0c9ff75e7fc5ac1`.
All four frozen jobs ran once, one native call each. Automatic final audit and
fresh-process archival reconstruction PASS, zero issues. No ERROR, TIMEOUT,
missing row, retry, point repair, new range, or source/configuration change.
This supersedes freeze-only status in the earlier delivery; do not launch again.

## Outcome

All four isolated exact checks completed. None supplied an exactly feasible
primal point. Thus **zero checked LP upper bounds, zero exact optimality proofs**,
and all four classifications remain `UNRESOLVED_CANDIDATE_VS_LP_RELAXATION`.
The new exact rational lower bounds equal their corresponding archived lower
bounds, not merely their displayed decimal approximations.

| Input / property | Checked L (display) | Primal status | Max equality violation (display) | Diagnostic seconds |
| --- | ---: | --- | ---: | ---: |
| 220 / 0 | -1.9255226787 | NOT_EXACTLY_FEASIBLE | 2.822e-15 | 5.568 |
| 222 / 1 | -3.5315074681 | NOT_EXACTLY_FEASIBLE | 1.537e-15 | 5.367 |
| 230 / 2 | -1.3654926726 | NOT_EXACTLY_FEASIBLE | 1.642e-15 | 7.591 |
| 232 / 0 | -9.7075542584 | NOT_EXACTLY_FEASIBLE | 7.256e-10 | 7.036 |

All native runs report success/status0 and HiGHS optimal status, with negative
candidate objectives. These are untrusted floating-point proposal metadata,
not checked LP optima. Exact checks find the following violated constraints:

| Input | Box | Inequalities | Equalities |
| --- | ---: | ---: | ---: |
| 220 | 2 | 83 | 1,441 |
| 222 | 0 | 48 | 1,536 |
| 230 | 0 | 75 | 2,136 |
| 232 | 0 | 85 | 2,007 |

Counts include every exact nonzero violation, however small. Raw rational
maxima and first violations are retained in the JSON. No constraint was ignored
because it fell below solver tolerance. In particular, the observed near
agreement of an infeasible point's objective and L is **not a certified gap**;
`upper_bound` and `exact_gap` remain null. No native infeasibility/optimality
status substitutes for a proof. Negative relaxed points are not network witnesses.

## Complete supplied-LP cost

| Exclusive phase / overhead | Total seconds |
| --- | ---: |
| Load / parse / identity binding | 1.707815 |
| Proposal phase, including retained internal check | 17.934386 |
| Portable packaging | 0.555540 |
| Isolated full exact check | 4.712545 |
| Residual driver/admission overhead | 0.651965 |
| **Diagnostic total** | **25.562252** |

Native LP time is **1.269790 seconds**, nested in the proposal phase, not added
to this total. The proposal remainder includes imports, matrix conversion,
candidate construction, serialization, exact dual evaluation and retained
checking; this run does not instrument those subcategories separately, so no
more specific causal cost attribution is claimed.

Separately recorded overhead: preflight2.555738s, resource checks/wait0.000341s,
post-terminal audits0.200582s, automatic final audit0.546652s. Fresh-process
archive reconstruction1.990296s is additional post-run work. All four diagnostic
cost records are complete. These are **supplied-LP** costs: historical network
propagation, range proof and F0 construction were not repeated. They do not
establish a full-request speedup or production-verifier coverage gain.

## What this resolves, and what it does not

The outer watchdog, complete cost records and movable exact checker have now
run on the actual frozen LPs. There was no budget exhaustion in these four
diagnostics. This does not retrospectively explain every timeout elsewhere.

What failed here is admission of the native point as an exact feasible witness.
This is a directly checked observation, not yet a proof that the LP minimum is
negative or that the relaxation cannot certify. The old 30 nonpositive
obligations remain candidate-vs-relaxation unresolved; only four were queried.
No new complete MoE SAFE or UNSAFE follows. Network→HZ, guard, route exclusion
and F0 lowering remain upstream trusted assumptions.

There is no direct motivation from these runs to enlarge the solver budget,
repeat these four queries, or expand to the remaining26. A possible next research
step would be a separately scoped exact-feasibility witness construction/check
interface with analytic controls. It is NOT implemented or authorized as an
automatic repair here; the current no-repair/no-tolerance-change protocol stays
sealed. Do not infer representation changes from the numerical optimum alone.

## Artifacts and checks

- [Frozen protocol](lp_diagnostic_v1.md), [freeze](lp_diagnostic_v1_freeze.json).
- [Execution archive](lp_diagnostic_v1_execution_results.json): exact diagnostic
  fields, native metadata, costs, raw artifact hashes and execution identity.
- `lp_diagnostic_archive/review.py`: fresh-process saved-record reconstruction,
  no solver/model calls or new mathematical bound checks.
- `python -m unittest lp_diagnostic_archive.tests`: one aggregate mutation test
  passes; rejects missing rows, altered cost, infeasible upper-bound promotion,
  false optimality and network-UNSAFE promotion.
- `lp_diagnostic.study.audit_saved()` independently reconstructs the final
  terminal/cost summary; a second post-archive invocation also passed.

The four LPs were actually checked by isolated rational checkers during the
frozen run. The subsequent archival audit checks identities/consistency; it is
not another independent proof of upstream lowering. Raw exports, native vectors
and proof bundles remain local under
`data/moe/results/lp_diagnostic_20260919_v1`; they are not committed.
