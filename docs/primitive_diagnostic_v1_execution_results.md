# Primitive arithmetic: four real diagnostics, completed

Execution HEAD: `17a7976ad90daa96960ec372472cb32938f99a9d`, clean and remotely
synchronized before launch. This was the separately frozen primitive V1 run,
not a retry or alteration of native-fidelity V2. Each of the four original LPs
ran once, in order, with the same 300/298/218-second clock, native<=10-second
cap, one basis attempt and4096-bit arithmetic limit. No new inputs, changed
properties, extra time, fallback, training or production-gate change.

## Result: 4/4 LIMIT, zero checked feasible points

All four native calls and mappings completed. Every exact constructor then
returned `LIMIT` at an integer cross-product in elimination. None reached
back-substitution, candidate publication, packaging or original-LP feasibility
checking. There are **zero checked LP upper bounds and no new network verdicts**.

| Job | Integer maximum recorded on entering elimination | First rejected product bits | Recorded pivots: old V2 / new | New constructor (s) | Whole supplied-LP clock (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| input220_p0 | 144 | 4,112 | 3,760 / 3,760 | 1.328533 | 13.489612 |
| input222_p1 | 139 | 4,097 | 3,051 / 3,050 | 1.354204 | 12.660985 |
| input230_p2 | 143 | 4,113 | 4,939 / 4,939 | 1.779918 | 18.821466 |
| input232_p0 | 154 | 4,116 | 4,081 / 4,081 | 1.879114 | 18.017880 |

Pivot counts are diagnostic counters: the selected pivot is counted before
its affected-row update may fail. They are not a claim that that many complete
pivot updates or a full basis solve were certified.

The batch/archival status `PASS` means the registered execution and evidence
accounting completed consistently. It is **not** a positive feasibility or
effectiveness result. Likewise `arithmetic_progress.complete=true` means the
LIMIT result and its journal closed normally; it does not mean elimination
completed successfully. Package/check durations remain null for every request.

## Comparison identity and what the logs actually distinguish

All four newly obtained basis structures match their corresponding V2 basis;
all four assembled exact-system hashes also match. This rules out an observed
basis/system change as the reason for these recorded comparisons. It does not
turn non-interleaved historical timings into a controlled speed experiment.

The first limits are consistently:

```
phase=elimination, operation=row_product,
kind=integer bits, cap=4096
```

In the frozen engine, each integer product is checked **before** the row
subtraction and subsequent whole-row gcd/content reduction. Thus the precise
finding is that the current raw-product contract is exceeded during recursive
elimination. This is not the earlier analytic adverse example in which an
input row's denominator LCM already exceeds the cap: all four real row-clearing
phases finish and the entry maxima above remain at most154 bits.

Fill insertions are zero for every row. Peak live entries are respectively
35,429 /34,594 /59,773 /58,056, below the two-million cap. Counted operations
are1,004,551 /1,117,581 /1,450,086 /1,577,562, below twenty million. Every
request completes its unresolved terminal in under19 seconds, not near218 or300.
The observed stop is therefore neither time exhaustion nor a recorded fill,
operation or live-entry cap. Structural counters are not physical RSS measures.

This still does **not** separate:

- large temporary products which could cancel in later operations;
- large reduced intermediate rows;
- an intrinsically large final rational solution;
- exact infeasibility of the native basis for the original rational LP.

The failed operands and a complete elimination proof are not stored. A bit
maximum and value hash locate an implementation stop; they are not independent
proof of a lower bound on required solution size. Being only1–20 bits over the
first gate does not imply that adding20 bits would finish the solve. No later
growth was observed because the contract deliberately stops at that first gate.

The earlier analytic success of primitive rows remains valid, but it did not
transfer to an endpoint gain on these four real systems. Conversely, these four
failures do not establish that primitive rows or all exact methods fail generally.

## Costs and retained evidence

Four complete supplied-LP clocks sum to **62.989942 seconds**. Native calls took
0.325706 /0.072445 /0.128283 /0.153317 seconds respectively. Native and exact
constructor times are nested inside those request clocks, not additive extras.

The new journals retain16 /17 /20 /21 snapshots. Assembly, row clearing and
elimination are separately recorded; interrupted stages are not invented. In
this run all four LIMIT records and their small result serialization completed
normally. No primal bundle was generated. Diagnostic entry/limit metadata is
bound to the plan, original LP, statement and returned hint.

Accounting outside or above the supplied-LP clocks:

| Recorded cost | Seconds | Relationship |
| --- | ---: | --- |
| Preflight | 3.349173 | Before batch |
| Resource checks | 0.000406 | Within attempts, outside request clocks |
| Post-terminal audits | 42.185560 | Within attempts, outside request clocks |
| All attempts | 105.225716 | Includes requests and the preceding two rows |
| Batch | 105.226850 | Includes attempts and batch overhead |
| Final summary audit/publication | 42.377780 | After batch clock |
| First independent archival pass | 43.860706 | Additional read-only work |

The second fresh-process review reports its own duration in
`primitive_diagnostic_v1_execution_review.json`. Do not add requests, attempts
and batch as if they were disjoint. The upstream network→HZ, guard/range and
F0 export costs were not rerun; none of these numbers is full MoE verification
latency or a speedup claim. The audit costs are disclosed, not subtracted to
make the pipeline appear faster.

`primitive_diagnostic_v1_execution_results.json` binds all211 raw execution
artifacts. The fresh reviewer re-runs the read-only archive reconstruction,
checks all identities/costs and compares the complete saved archive, then derives
the phase/bit/pivot table from stored records. Its source is
`scripts/review_primitive_diagnostic_results.py`. It does not invoke HiGHS,
reconstruct a basis, create a new bound or turn JSON agreement into a proof of
native arithmetic. Old sources, freeze, failures and V2 results remain intact.

## Decision: close this run; change the research question before another run

The scheduled execution and archival task is finished. **No unchanged rerun,
sample expansion or additional time is justified by these results.** The
current integer-row variant has zero endpoint gain in this bounded four-LP
study and should remain optional rather than becoming a default replacement.

The next useful research question is narrower than “try harder exact solving”:

> Can a separately specified candidate-construction method avoid materializing
> the observed large cross-products, while still producing a point that the
> unchanged original-LP checker can validate under an explicit resource budget?

This is a new arithmetic development task, not a continuation allowed to change
this freeze. Select one candidate approach and first test it on controls for
avoidable intermediate growth, intrinsically oversized solutions and inexact
bases. Keep candidate generation separate from acceptance. Do not silently move
the bit check after cancellation, raise the cap, change pivot order or add an
algorithm portfolio in the archived run.

There is still no checked feasible nonpositive upper bound for these four LPs.
Consequently this stage does not yet justify attributing the original output
obligations to an intrinsically nonpositive LP relaxation, changing the MoE
representation, or asserting model unsafety. A new arithmetic approach needs
its own scope and controls before another real diagnostic is frozen.
