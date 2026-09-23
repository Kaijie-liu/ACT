# Repaired MetaMoE same-cohort comparison: all 20 calls executed

Execution HEAD `442e08d5a2e361a9059e3d3f4b124c9dc0d625d1`, initially clean,
on `feat/moe-route-verification`, synchronized with origin. User explicitly
authorized the frozen execution, replay, independent audit, archival and
saved-log diagnosis. No retry, extra sample, new bound query or tuning followed.

- [Frozen execution](metamoe_las_followup_freeze_20260923_r1.md)
- [Config](../configs/backend_controls/metamoe_las_followup_r1.json), SHA256
  `af662082da17e953960ba1127d0f0fcf8783f0dd5810ce9b0bd8f43ab6aa031c`
- [Independent terminal/evidence archive](metamoe_las_followup_archive_20260923_r1.json)
- [Saved-only cost/query analysis](metamoe_las_followup_analysis_20260923_r1.json)
- [Analysis implementation](../scripts/summarize_metamoe_las_followup.py)

## Scope and complete outcome table

Same 10 previously selected, now observed inputs; same checkpoint, materialized
boxes and 19 global output margins plus route/nonzero obligations. This is a
repaired-version followup, **not a new holdout**. Original R1 remains sealed as
14 normal returns + 1 ERROR + 5 NOT_STARTED, and is not spliced into this table.

CPU/float64, two threads, normalized-space epsilon 2/255, clipping [-10,10],
300 seconds per complete request and sampled process-group RSS limit 8 GiB.
Not a pixel-space radius claim. ACT's 2 GiB sparse-representation policy,
30-second expert allocation, all acceptance gates and all obligations remain
unchanged. Both arms independently load and pay for their own computation.
The author arm is **author backend + disclosed lAs compatibility wrapper +
strict route-invariance sufficient adapter**, not literal unchanged author
execution or direct verification of an unrestricted dynamic-dispatch graph.

| Input | ACT terminal / charged seconds | Author terminal / charged seconds |
|---|---|---|
| CIFAR10/1 | UNKNOWN / 12.758418 | BACKEND_POSITIVE / 7.168308 |
| CIFAR10/2 | UNKNOWN / 37.529791 | BACKEND_POSITIVE / 7.141823 |
| CIFAR10/4 | UNKNOWN / 37.924940 | BACKEND_POSITIVE / 7.667664 |
| CIFAR10/5 | UNKNOWN / 40.402685 | BACKEND_POSITIVE / 7.139228 |
| CIFAR10/7 | UNKNOWN / 38.747782 | BACKEND_POSITIVE / 7.539665 |
| MNIST/1 | POSITIVE / 11.002260 | BACKEND_POSITIVE / 7.451619 |
| MNIST/3 | POSITIVE / 9.131068 | BACKEND_POSITIVE / 6.872268 |
| MNIST/7 | UNKNOWN / 76.004460 | TIMEOUT / 300.033312 |
| MNIST/9 | POSITIVE / 10.897942 | BACKEND_POSITIVE / 7.269995 |
| MNIST/10 | POSITIVE / 11.177982 | BACKEND_POSITIVE / 7.089864 |

| Registered denominator = 10 per arm | ACT | Author sufficient adapter |
|---|---:|---:|
| HZ policy acceptance | 4 | not this grade |
| Author numerical sufficient filter | not this grade | 9 |
| Original-full-model replayed UNSAFE | 0 | 0 |
| Completed UNKNOWN | 6 | 0 |
| Outer TIMEOUT | 0 | 1 |
| ERROR / RESOURCE_LIMIT / NOT_STARTED | 0 / 0 / 0 | 0 / 0 / 0 |
| Executed | 10 | 10 |

The four ACT positives have `HZ_POLICY_ACCEPTED`; all nine author positives
have `AUTHOR_BACKEND_NUMERICAL_SUFFICIENT_FILTER` (raw `safe-incomplete`).
These are distinct numerical contracts, not interchangeable formal SAFE.
The common positive set is MNIST {1,3,9,10}; ACT-only is empty; author-only
is CIFAR10 {1,2,4,5,7}. Neither accepts MNIST/7. Decided sets equal positive
sets because no UNSAFE was accepted. All 10 pairs remain in the full outcome
and charged-cost comparison; nine pairs also qualify for the separately
labelled normal-return timing statistic. A timeout is not model unsafety.

Every ACT positive has one feasible route with the other excluded. MNIST/7
retains two candidates but one has unresolved reachability: this is not proof
of two feasible routes. **No new route-changing or source-complete certificate
is obtained.** The finite observed cohort favors the author sufficient path
in positive coverage; it does not establish general author-tool superiority.

## Full cost and repair behavior

| Scope | ACT total / mean seconds | Author total / mean seconds |
|---|---:|---:|
| All 10 requests per arm | 285.577327 / 28.557733 | 365.373746 / 36.537375 |
| Five CIFAR10 per arm | 167.363615 / 33.472723 | 36.656688 / 7.331338 |
| Five MNIST per arm, including timeout | 118.213711 / 23.642742 | 328.717059 / 65.743412 |

All-attempted medians are 25.144105 s ACT and 7.219151 s author. Paired ACT
minus author cost over all 10 inputs has mean -7.979642 s, median +4.839114 s.
The negative mean is driven by the author's unresolved 300-second request;
ACT also does not solve that input. It is **not an equal-success speedup**.
On nine normally returned pairs the difference is +16.025826 s mean,
+5.590111 s median. On four common positives the means are 10.552313 s ACT
and 7.170937 s author. Do not treat an early UNKNOWN as a fast solution.

Charged requests total **650.951073 s**. Batch through final summary is
**651.872033 s**, including parent inventory/postflight; replay 1.142675 s
and primary audit 3.533242 s are separate. Frozen input selection is separate.
Peak sampled group RSS is 2,585,812,992 bytes ACT and 5,185,753,088 bytes author;
these are sampled observations, not instantaneous peaks or representation sizes.
The 0.033 s outer-timeout overrun is termination/cleanup, not a larger grant.

MNIST/7 performs seven checked zero-metadata restorations and seven original
domain insertions, then times out without the former assertion error. No
restoration is invoked by the other nine author requests. This establishes
the exercised repair's compatibility, not stronger bounds or a new certificate.

## Saved-only diagnosis: execution limits, not established precision failure

There are **209 output obligations** across 11 expert evaluations: 116
expanded violation regions excluded under the frozen native policy, 93
UNKNOWN with local TIMEOUT. No output row was omitted. Ten evaluations have
checked feasible bases; the remaining MNIST/7 branch uses native fallback.

| ACT input / expert evaluation | Excluded / required | Local output timeouts | Base seconds / state | Expert seconds |
|---|---:|---:|---|---:|
| CIFAR10/1 | 18/19 | 1, row 1 | 0.063846 / checked | 5.168555 |
| CIFAR10/2 | 0/19 | 19 | 0.083623 / checked | 30.013258 |
| CIFAR10/4 | 0/19 | 19 | 0.108374 / checked | 30.024304 |
| CIFAR10/5 | 0/19 | 19 | 0.113281 / checked | 30.017622 |
| CIFAR10/7 | 0/19 | 19 | 0.137739 / checked | 30.026470 |
| MNIST/1 | 19/19 | 0 | 0.032999 / checked | 1.475217 |
| MNIST/3 | 19/19 | 0 | 0.031583 / checked | 1.361809 |
| MNIST/7, evaluation 0 | 4/19 | 15 | 2.908413 / native unknown | 30.017439 |
| MNIST/7, evaluation 1 | 18/19 | 1, row 10 | 0.036700 / checked | 5.142426 |
| MNIST/9 | 19/19 | 0 | 0.031030 / checked | 1.367619 |
| MNIST/10 | 19/19 | 0 | 0.031675 / checked | 1.381342 |

Five CIFAR10 requests finish routing/base/nonzero checks. Their 77 unresolved
output queries enter native MILP but do not save a native return. Four requests
spend the 30-second expert cap across 19 properties: local allocations roughly
1.486--1.571 s; pre-native costs range 0.255--0.847 s, leaving roughly
0.668--1.314 s at the recorded native-entry point. These timestamps cover
dispatch/setup; they do not separate all import, matrix assembly and presolve
costs. The code terminates a worker on local TIMEOUT; subsequent queries pay
startup again. This is a concrete overhead/fragmentation mechanism, not proof
that eliminating it would close an output property. The remaining native
search versus representation difficulty is unresolved.

CIFAR10/1 is different: only row 1 times out (allocation 1.638 s, pre-native
0.006 s). Later properties finish rapidly, so the whole expert returns in
5.169 s without revisiting the unresolved row. There is unused expert and
outer time, but the frozen single-pass policy intentionally has no retry.
No counterfactual outcome or larger allocation is inferred from that fact.

MNIST/7 has **additional route and nonvacuity gaps**. Route 0's native query
returns late/unknown after about 30.051 s; route 1 has a checked feasible
assignment. Evaluation 0's proposed base assignment is rejected (maximum
row violation about 0.684), then native base feasibility remains unknown.
Both expert outputs remain incomplete. No score-nonzero obligation is reached;
an empty nonzero list is not success. The final reason is
`incomplete_route_coverage`, not just slow output search. This request must
not be grouped with the five CIFAR-only output bottlenecks.

One of the 93 output TIMEOUTs has a saved native status-2 record:
MNIST/7, evaluation 0, property row 6. Its native finish timestamp precedes
the local deadline by about 6.4 ms, but that timestamp is taken **before**
JSON/fsync/atomic publication. The parent did not receive and accept the
result within budget. Its final status stays UNKNOWN. The other 92 have no
saved native-return record; their native durations are censored, not zero.
No feasible violating output assignment was saved. This single unaccepted
record cannot close the request, diagnose all other queries, or justify
relaxing receipt/acceptance deadlines.

**Decision:** seal this batch and stop real retries, sample expansion and
precision/backend tuning. Saved evidence supports investigating local budget
fragmentation and setup/receipt costs as a separate future execution-design
question; it identifies no specific representation defect warranting a new
precision control now. No new control is frozen or launched by this report.
Changing only output allocation would also leave MNIST/7's independent route
and base gaps unresolved. Do not claim the relaxation is inadequate, that
more time must succeed, or that the models are unsafe.

## Audit, reproducibility and preserved boundary

Frozen auditor PASS, 0 issues. A second saved-only audit with native MILP,
support optimization, production sign and production repair/reachability
helpers disabled matches the complete archive except audit clock. All ten
author VNNLIBs are additionally rebuilt from stored boxes/properties,
including the timed-out request. The separate original-model replay ledger
contains zero witnesses; it does not independently prove positive bounds.
`repair_control_gate=false` is intentional for this non-control protocol.

The new analysis has ten accounting controls; existing lAs/observation/
supervision controls add 17, and historical summary controls add seven.
Analysis drafting first exposed the native-fallback audit schema's absence
of `base_seconds`; it now obtains that time from the recorded base query,
never zero. A control covers this case. Finish timestamps are explicitly
distinguished from publication/receipt. No experiment or frozen audit code
was modified to address these reporting cases.

Raw root: `/data1/Kane/MOE/baseline_runs/metamoe_las_followup_20260923_r1`.
The archive binds 1,883 files / 289,953,111 bytes after replay. Raw matrices,
checkpoints, datasets and external repositories are not committed. The
condensed archive and analysis contain identities, outcomes and costs only.
Network-to-HZ/guard conversion, native numerical acceptance and author bounds
remain trusted components. Structural audit PASS is not source-complete proof,
not independent re-certification of all bounds, and not human technical review.
