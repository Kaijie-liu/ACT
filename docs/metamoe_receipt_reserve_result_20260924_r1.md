# Receipt reserve: frozen eight-call study completed; no endpoint gain

User authorized execution on 2026-09-24. Started clean and synchronized on
`feat/moe-route-verification`, HEAD `f9898eb8e55916b999638d04dae6d0609fbc0173`.
The [freeze](metamoe_receipt_reserve_freeze_20260923_r1.md), implementation
`cff54e000`, and config SHA256
`b0c71b5df6ae9190770b03869e99cd5d02a6b013e1cf483498b3336ba581ff9f`
were unchanged during execution. No retry, extra request, author-arm timing
comparison, relaxed acceptance, or historical-result overwrite occurred.

Only the output-query native soft cap differs: all of its assigned query time
versus 80%, reserving the remainder for receipt. Both arms pay the same receipt
instrumentation. The parent 300s budget, expert 30s budget, property allocation,
all 19 output obligations, checked-base fallback and numerical gates remain.

## All attempted requests

| Input | Full-native status / charged seconds | Receipt-reserve status / charged seconds |
|---|---|---|
| CIFAR10/1 | UNKNOWN / 14.591734 | UNKNOWN / 13.475758 |
| CIFAR10/2 | UNKNOWN / 37.978010 | UNKNOWN / 37.517267 |
| MNIST/1 | HZ-policy POSITIVE / 8.590110 | HZ-policy POSITIVE / 8.384968 |
| MNIST/3 | HZ-policy POSITIVE / 8.505548 | HZ-policy POSITIVE / 8.169206 |

Both arms: 2 positives, 2 completed UNKNOWN, 0 UNSAFE, 0 outer timeout,
0 error and no omitted call. Same positive and decided sets; gained/lost = 0/0.
This is four previously observed inputs, not four new holdout inputs, and the
two positives are not newly source-closed or route-changing certificates.

| Accounting / mechanism observation | Full-native | Receipt-reserve |
|---|---:|---:|
| Sum charged request seconds | 69.665402 | 67.547199 |
| Native worker starts | 23 | 23 |
| Missing native runtime records | 20 | 20 |
| CIFAR1 excluded / unresolved output rows | 18 / 1 | 18 / 1 |
| CIFAR2 excluded / unresolved output rows | 0 / 19 | 0 / 19 |
| Each MNIST excluded / unresolved output rows | 19 / 0 | 19 / 0 |

The observed total difference is -2.118203 seconds (about -3.04%). One run per
arm does not quantify timing variability; **no stable speedup is established**.
CIFAR2 spends 30.0145 / 30.0142 seconds in expert work. The intended reduction
in worker restarts or missing returns was not observed. Neither completed
relaxation precision nor model unsafety follows from the unresolved rows.

## Audit and complete costs

Batch through final summary: 138.054155s; charged requests: 137.212601s.
Replay/audits/archive are separate administrative work, with their actual
recorded durations retained in the archive. There are no UNSAFE witnesses in
this batch; the explicit zero-witness replay inventory still binds both arms.

[Compact archive](metamoe_receipt_reserve_archive_20260924_r1.json) contains
every raw file hash/size, launch identity, per-request counters, statuses,
expert costs and paired differences. Raw records remain in the frozen new
directory `/data1/Kane/MOE/baseline_runs/metamoe_receipt_reserve_20260923_r1`.
Original independent audit: PASS, zero issues. A second saved-only reread with
`NativeSession.query` disabled reproduces every audit field except its own
elapsed audit cost. `python scripts/archive_receipt_reserve.py --check`
reconstructs the committed archive without new queries.

The audit checks recorded identity, obligations, numerical-policy acceptance,
receipt/terminal provenance and cost; it is not an exact solver proof or source
containment repair. The historical 23 main-table source gaps remain unchanged.

## Decision

**STOP_THIS_FACTOR.** Keep the reserve option experimental, not a new default.
Do not tune the fraction, expand the cohort, increase time, or infer a relaxation
defect. This finite mechanism hypothesis produced no endpoint or restart signal.
The separate source/output conversion line can continue through controls; it
does not use these timings or positive statuses as mathematical evidence.
