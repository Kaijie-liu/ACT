# Guarded routing feasibility: same endpoint, less repeated base search

Two frozen executions completed on old MNIST0. Implementation `32cfe732a`,
freeze `104d22d90`; config SHA256
`573bc68e5a62eff0ee7272d85421c82bdb99bb625e6cfbbfed120c098f2eb5a9`.
See the [protocol](metamoe_checked_routing_protocol_20260923_r1.md),
[config](../configs/recent_moe/metamoe_checked_routing_control_r1.json), and
[independent saved-only audit](metamoe_checked_routing_archive_20260923_r1.json).
Audit PASS, zero issues, zero new audit solves. Raw records remain under
`/data1/Kane/MOE/baseline_runs/metamoe_checked_routing_control_20260923_r1`;
only the result/identity inventory is committed, not weights or raw arrays.

## What was actually compared

Both arms use the repaired BN graph and opt-in checked **expert** base from
the preceding stage. Only **routing feasibility** differs. Same checkpoint,
physical input, `2/255`, 19 global properties, numerical gates, 300-second hard
request deadline, 30-second local query allocation, 8 GiB sampled group RSS
cap and 2 GiB representation policy. Support optimization is unchanged.

The checked arm builds one fresh assignment on each complete guarded routing
matrix. An accepted assignment establishes feasibility only; it cannot exclude
a route or prove a global property bound. Rejection falls back to the original
native query with its remaining original deadline. No cache, deduplication,
additional samples, larger limits or relaxed acceptance rules were introduced.

| Quantity | Native routing | Checked routing |
|---|---:|---:|
| Complete request terminal | POSITIVE / HZ_POLICY_ACCEPTED | POSITIVE / HZ_POLICY_ACCEPTED |
| Charged request seconds | 132.264205 | 39.452816 |
| All candidate analysis, seconds | 92.820852 | 0.108426 |
| Branch 1 feasibility, through pre-publication record | 92.758583 s | 0.027941 s |
| Branch 0 feasibility, same clock | 0.050688 s | 0.069373 s |
| Expert evaluation, inclusive trace seconds | 1.697270 | 1.689633 |
| All required output violation queries | 19/19 infeasible | 19/19 infeasible |
| Score-support optimization, inclusive seconds | 30.058158 | 30.043572 |
| Peak sampled group RSS, GiB | 2.261234 | 2.273430 |

Full charged cost decreased by **92.811389 s (70.17%)** in this single ordered
old-input control. This is not a population speedup estimate, randomized timing
trial, new coverage gain, or comparison against the author verifier. Both arms
reach the same endpoint. Native-first order and shared-machine variability are
explicit limitations; the observed eliminated native feasibility search is
directly identified in the trace.

## What the evidence establishes

- Both complete guarded routing matrices match across arms. Branch 0 is
  excluded only by the unchanged native infeasibility policy. Branch 1's
  checked point covers all 6,852 variables and 3,781 original rows, including
  the membership guard and 1,260 binary variables. Independent scalar-CSR
  evaluation has maximum equality residual `1.3877787807814457e-17`, under
  the unchanged `1e-7` policy. The checked point does not come from the other
  arm or any historical result.
- Both expert matrices match; fresh expert base checks and all 19 native
  property queries were independently accounted for. No property was skipped.
- Candidate `[1]`, excluded `[0]`, unresolved `[]` in both arms: this is a
  **route-stable** request, not a new route-changing certificate.
- The native branch-1 search returned after its 30-second local soft limit.
  Its overrun is explicit in the archive; the full request stayed inside the
  hard 300-second watchdog. Neither arm claims a hard local native limit.
- Score bounds remain the same `fast_fallback` enclosure
  `[3.345338179462453, 4.30685851511294]`. A feasible point did not replace this
  global enclosure, and the failed optimization did not become a proof.

The audit rechecks saved accepted points using scalar CSR rows, verifies
native-return/trace bindings, all output obligations, matrix equality, terminal
precedence, evidence inventory and costs. It does **not** independently prove
native infeasibility, source-to-HZ enclosure, guard lowering or deployment
floating-point execution. `source_complete=false` remains unchanged. The earlier
BN control is same-object **point conformance**, not all-domain equivalence.
All pre-repair runs and earlier failures remain sealed and unmodified.

## Complete cost and controls

Charged requests total **171.717021 s**. Batch wall through final summary,
including postflight inventory, is **173.294488 s**; the final cost-file write
is explicitly outside that clock. Inventory time is 0.014756/0.015334 s.
Independent audit is separate, **1.092572 s**. All 291 raw evidence files total
35,036,650 bytes. No error, omitted request or right-censored trace span occurs.

100 focused tests pass in `act-py312`, and 58 overlapping tests pass in the
pinned execution environment. Controls cover all tie-legal routes, wrong
guard/scope/model/point, unsupported proposals, unchanged support, native
differentials, partial/late publication, exhausted deadlines, exception stop,
immutable output directories and complete outer cost. A broader historical
suite rejects old source hashes in five places (including a class setup): those
manifests bind the pre-repair BN converter. They were not rebound to make old
protocols run on a different object.

## Continue / stop decision

This control supports retaining checked routing feasibility as an **opt-in**
engineering path. It does not justify changing the default, announcing new
formal coverage, or tuning the already-closed expert relaxation.

The unchanged score-support call now consumes about 30 of 39.45 seconds. Its
fallback enclosure already excludes zero. A separately scoped next question is
whether a checked, policy-equivalent sign-only enclosure can discharge this
definedness obligation before attempting optimization. That would need its own
controls and frozen comparison; it is **not implemented or credited here**.
Source-enclosure closure and a fresh author-tool comparison remain separate
research tasks, not consequences of this timing result. No new cohort or
experiment is running or automatically queued.
