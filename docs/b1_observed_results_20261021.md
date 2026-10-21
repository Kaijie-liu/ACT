# B1 observed-request attribution results

Status: the second execution stopped after18/41 requests: UNKNOWN3, TIMEOUT14,
ERROR1. No scientific outcome is inferred from the tooling error. The
[frozen queue](b1_observed_freeze_20261021.json) retains all41 identities and
the unchanged adaptive configuration.

## Retained prefix and suffix execution

The second controller has exited. Its final request, seed0/4275, wrote an
UNKNOWN package at293.436246s but exited with SIGALRM at294.185604s. Reception
rejected it; its original ERROR remains unchanged. The worker had left its
294s timer armed after completing business work. The timing supports a teardown
race, but does not identify the native handler transition. No generic signal
exit is reclassified as TIMEOUT, and the saved package is not accepted offline.

The minimal repair cancels this timer after business work and stream flushing;
the existing296/298/300s outer execution, cleanup and reception limits remain.
Twenty-five ordinary tests pass in0.484s, including standard-library child
controls for delayed teardown, authorized in-work cutoff, rejected abnormal
exits, prefix identity, capture allowance and cumulative cost. They invoke no
trained model or solver. Independent read-only review found no suffix blocker.

Execution is fixed to the remaining23 requests, ordinals18–40, beginning with
seed1/4275, in a new `b1_observed_20261021_r3` directory and immutable code copy.
The18 original request directories are hash-bound references, not copied or
rerun. All10 full-source ordinals are already consumed. R1 and R2 contribute
4823.979257s of full execution time and151830125 stored bytes before this
continuation; R3 and later diagnostics carry both forward against the same
six-hour and2GiB caps. Full execution time, rather than only the sum of request
times, is deducted. The complete catalog will still contain41 results and42
attempts including the earlier log-path failure. No extra capture or query is
authorized. The existing145-entry diagnostic catalog and B2 order are unchanged.

R3 is launched from immutable commit`19cf25583416163bfd6be44450c22b8323e24903`,
controller PID1857594 (detached, PPID1). Its first launch is ordinal18,
seed1/4275, with full-capture allowance0. The active checkout is not its code.
The future offline and B2 workers now reuse the same tested timer finalizer;
two additional ordinary wiring tests pass in0.004s. They do not launch a native
solver or change query deadlines, reception, sources or scientific settings.

## Running prefix

The following observations are historical prefixes; the stopped result and
suffix execution above supersede the former live-controller description.

The immutable execution copy is`/data1/Kane/MOE/executions/b1_observed_20261021_88a705275`;
the detached controller is PID1234759 and writes
`/data1/Kane/MOE/results/b1_observed_20261021_r2`. Live ledgers there supersede
this deliberately dated prefix; the active checkout can develop offline queries
without changing the running code.

At the first two completed requests, seed1/4113 is UNKNOWN in55.466712s and
seed2/4113 is TIMEOUT in296.095948s, including terminal/ledger publication to the
stated final clock boundary. Together they retain45 registered pair/property
obligations and28 returned native F0 calls. These counts are not28 solved
obligations or45 independently proved lower bounds. The first10 full-source
ordinals are reserved. Saved artifacts are approximately138MB; cumulative
stage cap remains2GiB. There are zero additional attribution queries so far.
The third request is running; this is not a final comparison table.

## Offline query implementation

[Saved-source helpers](../scripts/b1_saved_diagnostics.py) now implement the
registered same-native continuous LP, integer difference extrema, replacement
of only the difference range, and16 closed margin slices with two endpoints.
Four ordinary decode/identity/residual/deadline controls pass without solving.
Static review checks the shared-factor padding, margin constant, gate direction,
and absence of hidden support queries. The online copied worker is unchanged.

Raw primal/dual telemetry is separate from the original status/finite/outward
acceptance policy; candidate residuals are numerical checks, not exact feasible
certificates. A reconstructed F0 with new integer-domain difference bounds is
explicitly not the original continuous LP. There are no extra routine MILP slots:
the captured online MILP is the baseline, and only the separately registered
eight eligible extended calls can rerun it.

The later saved-query review retains each existing HZ solver return's candidate,
offset-inclusive primal and raw dual, gap, reason and policy minimum for range,
replacement and slice queries. It adds no solver call. One ordinary no-solver
test passes, checking that a positive raw dual at status1 remains recorded with
`minimum=None`; it is not promoted to an accepted bound. Upper difference
queries explicitly label their minimized objective as`-d`.

The slice diagnostic compares margin slices plus endpoint support on the shared
integer HZ, not merely a smaller gate interval in the original F0 matrix.
An improvement over F0 therefore cannot uniquely identify slicing resolution:
the endpoint encoding and query allocation also differ. The first two capture
ordinals remain fixed, even if missing or from the same request. A complete
slice conclusion requires every necessary endpoint, not just the best completed
ones. Same-matrix LP changes only integrality; comparison to the earlier MILP
still has different solve allocations. Internal elapsed fields cover different
scopes, so full costs use query `receipt.seconds`. These diagnostics remain
numerical; a negative lower bound alone does not establish a nonpositive optimum.

The [offline queue](../scripts/b1_diagnostic_queue.py) now uses the existing
owned-session supervisor and a separate frozen code copy. Its unchanged catalog
has145 entries:41 candidate-replay/PGD,10 same-matrix LP,20 integer difference
extrema,10 range-only replacements, and64 closed-slice endpoint queries for the
first two source ordinals. Missing ordinals are not replaced. The cumulative
online clock, including the retained tooling prefix, is deducted from six hours;
the separately authorized eligible extended calls retain their own four-hour cap.

PGD is fixed before execution:20 steps,3 restarts,step size epsilon/4,cross-entropy,
saved-box projection,CPU/float64,seed20261021 plus request ordinal. Each replay
query has60s including loading and reception. Only a finite, in-domain, dynamic
full-model misclassification is a replayed counterexample. Source/network point
differences, native LP residuals and static-pair guard violations are diagnostic
fields, not exact feasible certificates or unique-cause proofs.

Seven queue/mapping controls pass without native solving, covering the missing
second source, unchanged145-entry ordering, prior costs, wrong request binding,
offset-aware extension eligibility, binary mapping and authorized cutoff handling.
The saved-source decoder regressions also pass. Before real diagnostics, one
fixed tiny native control executes two calls (continuous and integral versions
of the same two-variable matrix, objective offset1/4, expected optimum−3/4),
each solver limit2s, within one300s control budget. No trained source is loaded.
This is an API control, not a comparison or repeated algorithm experiment.

The native API unit control at`cfd1ed3ec` passed: both calls returned status0,
primal−0.75 and the policy's outward-adjusted lower bound. Test-suite time was
1.172s; this is not end-to-end supervisor timing. The seven queue tests passed in
0.038s. No real diagnostic slot was used. These unit tests do not replace the
actual queue's per-process address-space limit, hard deadline and owned cleanup.

The later ordinary queue controls distinguish the separately authorized1800s
extended diagnostic from ordinary≤300s queries. Worker and receiver use the same
accounting factory; extension eligibility, source binding and exact arithmetic
bit limits are unchanged. This repairs an accounting-constructor mismatch,
not the online deadline or numerical acceptance policy. No extended solve has
yet run.

At six completed online requests,99 pair/property records are retained:
29 reused,66 returned and4 in flight at cutoff. There are70 registered F0 calls
and66 returned native calls. Of the returned observations,47 have native status1;
43 of those contain a positive raw dual telemetry value that the unchanged
status0 policy does not accept. Two status0 returns have negative candidate
objectives. These are descriptive observations, not checked optimality or a
unique explanation of failure. Three additional observations meet the registered
extended-solve sign condition but may lack retained matrices; missingness will
not cause substitute capture or propagation.

The two status0 negative candidate observations provide a specific gate-coupling
hypothesis. Seed1/4113,pair{2,5},property0 has candidate−0.069105366,
product gap0 and gate-output gap−0.710506384; replacing only the saved point's
gate by sigmoid(saved margin) gives0.641401018. Seed0/4140,pair{3,6},property3
has candidate−0.028358019,product gap about−4.44e−16 and gate-output gap
−0.035915644; the same pointwise substitution gives0.007557625. Both saved
dynamic-model replays are nonviolating. This is not a feasible global upper
bound, a proof of safety or a unique-cause attribution. The registered slice
diagnostic tests the first retained source; the second request has no retained
full matrix and will not be substituted into the fixed capture catalog.

At ten completed requests the saved-only report counts UNKNOWN2/TIMEOUT8,
110 registered F0 calls and102 returned native calls. There are144 property
records:34 reused,93 returned,8 in flight and9 selected. Of the saved return
observations,78 have solver status1 and62 of those have positive raw dual
telemetry rejected by the unchanged status0 policy. These are overlapping
diagnostic categories, not62 certificates. Fourteen observations meet the
extended-solve sign condition; the fixed first-eight rule and retained-matrix
availability still apply. Attribution queries remain0. The live ledger
supersedes this prefix.

## Saved invocation costs and attribution limits

The [fixed11-request record](b1_observed_records_20261021.json) binds its ledger,
inventory hashes and summary implementation. It has171 model–input–pair–property
obligations,130 registered calls and121 saved native returns. `SELECTED` means a
returned obligation was selected for full capture, not another solve. General
multi-pair monolithic calls can cover several obligations, so the summary now
separately deduplicates invocation statistics by request and ordinal. Two new
ordinary controls check duplicate-call and missing-value handling; all11 queue
tests pass without native solves. The live frozen worker is unchanged.

Of121 returns,35 have status0 and86 have status1. Seventy status1 calls, across
all nine TIMEOUT requests in this prefix, retain positive raw dual telemetry.
Their recorded elapsed sum is1467.399s, median20.891s, range9.805–32.551s.
The raw dual values with objective offset range0.359715–7.313035; raw solver
`mip_gap` ranges0.007241–3.275441. These values do not satisfy the original
status0 acceptance policy and are not independently checked certificates.

| Saved evidence | Attribution supported | Still unresolved |
| --- | --- | --- |
| Repeated status1 returns with positive raw dual telemetry | Investigate cost of per-property allocation and optimal-status requirement | No dual multipliers or independent residual certificate |
| Five single-pair requests return8 properties; the9th is in flight at cutoff | The complete obligation set remains unfinished | Exact granted time and native termination messages are absent from timeout packages |
| Two status0 negative candidates with nonviolating replay | Investigate gate/product/lowering contributions using the registered diagnostics | No exact candidate-feasibility or global network conclusion |

`decision.elapsed` includes lowering/target construction, native solve and
candidate validation/recovery; subsequent full capture/serialization is outside
it. It is not pure HiGHS time or total request cost. The adaptive code divides
remaining time across remaining unreused properties rather than applying a
uniform10s limit. Exact per-call grants survive in complete package budget
events, not in these nine timeout packages. The compact observations retain
status/gap/node count but not the native message. PyTorch and BLAS environment
settings are one-thread; actual HiGHS thread count was not measured. The raw
`mip_gap` belongs to the native objective scale and is not recomputed after
adding the HZ objective offset.

These are reasons to perform the already registered LP/integer-range/slice
diagnostics, not to relax a gate, rerun the online queue, or declare one failure
cause for all requests. This saved-only analysis adds zero attribution queries.

## A complete obligation set with status limited returns

The separately bound seed2/4223 case in the existing records has all27 obligations
accounted for across pairs{0,2},{0,3},{0,6}:10 reused facts,11 accepted status0
F0 rows and6 status1 UNKNOWN rows. All17 native calls returned; reception completed
in286.187s. This is neither missing coverage nor an outer timeout.

The six unresolved rows are pair{0,2}/properties0,2,3 and pair{0,6}/properties2,3,6.
Their raw duals **after adding the objective offset** are1.299,2.985,3.775,
3.400,4.271,6.726; their native grants range12.385–16.947s. None satisfies the
unchanged status0 acceptance requirement. All17 candidates are positive, with
minimum0.170906. In particular, the last row's native `fun` is−1.576630, but its
offset is1.747537: the complete property objective is positive, not a new negative
relaxation candidate. The six grant/elapsed pairs are retained in the record;
elapsed includes construction and candidate handling, not just native solving.

All six unresolved points have positive expert difference and a gate at its lower
endpoint. Their saved product discrepancies are near floating-point roundoff;
substituting sigmoid(saved margin) increases already-positive candidate values.
This does not supply a third negative-to-positive example like4113/4140.
Consequently, the observed blockers for this request are the limit/status-policy
combination, not evidence of a nonpositive relaxation optimum. The raw duals lack
independent residual certificates; neither SAFE nor success with extra time
follows. The request remains UNKNOWN and no extra solve is added. This distinction
must survive the final multi-label attribution before selecting a B3 mechanism.

## Retained tooling prefix

The first invocation, seed1/input4113 at`49fea3658`, stopped after3.640219s:
the independent git archive lacks an untracked default ACT debug-log directory.
No F0 obligation or diagnostic query was reached. The original ERROR and empty
diagnostic Work counters remain in`/data1/Kane/MOE/results/b1_observed_20261021_r1`.
The ordinary supported `PerformanceOptions.set_debug_output_file` redirects
this log into each request directory. No solver, propagation, acceptance or
sample change accompanies the repair. The continuation uses a fresh immutable
code copy and new result directory, retains the original call and costs, and
charges its bytes and seconds against the same stage. Expected total attempts
are42 for41 registered mathematical requests; the tooling failure is not erased.

The first shell-based background launch produced only an empty controller log,
no request directory or worker. The subsequent detached Python launch produced
the retained invocation above. No extra model query was made by the shell attempt.

Final analysis will distinguish execution limits, nonpositive numerical evidence,
gate/product discrepancies, LP/integer-domain sensitivity, route coverage and
replayed counterexamples. Missing or unfinished diagnostics remain in the
denominator; offline outcomes cannot upgrade an online terminal.
