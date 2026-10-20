# Native margin source integration result

The frozen A1 batch passed on execution commit
`7a26ffe3d1b63355053ec0f62d1763a2d6d66974`. All seven source-interface groups,
six observation cases, both online receptions and two expected process faults
completed. The outer command exited0. The sole attempt is consumed; it is not
to be rerun as a performance comparison.

The [protocol](margin_native_control_20261021.md),
[freeze](margin_native_freeze_20261021.json) and
[lossless saved records](margin_native_records_20261021.json) bind the result.
The archive reconstructs prefixes using the single shared source and the
appropriate prefix of candidate records; the auditor verifies original byte
counts and hashes after reconstruction. Original run files remain untouched.

## What the control establishes

The native source has3,072 shared continuous factors, unequal A/B private widths,
four binary factors, all ten classes and the two router outputs in one frame.
Exact source composition, nonidentity router row maps and independent direct
coordinate projection agree with the sparse property views. Two four-column
CPU candidate calls reuse the existing128-update generator.

All eight original-coefficient support bounds were checked in the worker,
recomputed by the controller, and recomputed from a relocated copy with
`python -I -S`. Scope/property/source mutations, missing objectives or dual
rows, invalid signs, malformed CSR, live aliases, bit overflow, partial evidence
and expired reception were rejected. The four binaries are explicitly relaxed.

**All eight checked support bounds are negative.** That is neither a model
counterexample nor a failure of this pre-frozen integration gate, which did not
require positive bounds. No sigmoid scalar aggregation, complete output
obligation set, trained model or route-changing witness was evaluated.
The grade is `CHECKED_SUPPORT_ON_GIVEN_HZ_CONTINUOUS_RELAXATION`;
`scalar_aggregate_checked` and `complete_request` remain false.

## Execution and costs

| Measurement | Observed value | Scope |
| --- | ---: | --- |
| Entry through terminal file export | 8.558092s | Includes native and portable execution, faults and cleanup; not full cold start |
| Native session execution plus cleanup | 7.206989s +0.025479s | Includes imports, source/views, candidates, exact checks and prefix serialization |
| Native imports | 2.990140s | Nested inside native execution |
| Source capture/composition | 0.293215s | Nested inside native execution |
| Two proposal-plus-check calls | 0.461860s,0.426031s | Not isolated solver/kernel times |
| Controller exact reception | 0.243669s | Same absolute deadline and continued Work |
| Relocation copy/check | 0.002673s | Same request budget |
| Portable session execution plus cleanup | 0.527276s +0.026042s | Includes stdlib startup and exact reception |
| Complete observed native session/caller RSS sample peak | 762,007,552 bytes | Telemetry, not aggregate hard limit |
| Native direct-child wait4 ru_maxrss | 739,348,480 bytes | Not session simultaneous peak |
| Retained run artifacts | 1,551,417 bytes | All22 files, including copied checker and prefixes |
| Repository records archive | 189,856 bytes | Together1,741,273 bytes, below128MiB |

These nested times must not be summed as disjoint costs. There is no baseline
arm or acceleration estimate. Interpreter setup before entry and stdout/exit
after the recorded export are not reconstructed; `full_cold_start_seconds=null`.
The separately observed exit0 is retained in the archive, rather than inferred
from a PASS JSON file.

Native execution recorded one incomplete RSS sample out of167. It remained
unknown under the explicitly adopted development policy; it did not establish
process exit. Normal exit was separately observed, all four owned leaders were
reaped, and cleanup found no live session members. The expected nonzero child
remained `ERROR/7`; the deadline child remained `TIMEOUT/−9`. FDs stayed4→4.
Legacy RSS acceptance and old failed attempts are unchanged.

Work continued11,643,669→11,977,711→12,311,753 through worker, controller and
portable receiver; each complete reception added334,042 visits. Of the total,
8,000,001 visits are the declared artificial accounting control. Remaining
recorded visits are4,311,752, including faults and repeated checking, not a
measure of useful algorithmic speed. Stored/intermediate bit gates remained.
The worker recorded the existing NumPy2.3.5, SciPy1.16.3 and Torch2.9.1+cu128;
execution was CPU-only with no dependency installation.

## Independent saved-only review

Two read-only AI reviewers checked mathematics/source binding and execution
accounting. Eight bounds also matched a second exact Fraction implementation
that projects the original22 output rows directly, rather than using b/d/m
projection. These are not human technical review or a clean-environment study.
No native call or controlled child was repeated by the saved-only audit.

Reproduce the retained-record audit without numerical packages:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -I -S -B scripts/audit_margin_native_20261021.py
```

The audit returned PASS in0.354s on this machine; its cost is offline and is
not added to the online acceptance record. It checks freeze identities,
lossless reconstruction, all eight exact supports, complete terminals and
continued accounting. New archival records occupy less than the reserved
additional copy of all run artifacts. No failure/source/checkpoint/environment
was deleted.

## Research decision

Continue A1 to sparse margin planning and checked scalar aggregation on this
source interface, then test the actual operator against zero tilt and the
registered strong slicing alternatives. Do not resume M1b or expand the
observation chain. The useful new fact is native integration and portable
support acceptance, not evidence of tighter complete-model bounds or net gains.
G1–G6 remain OPEN. New real-development ledger remains0 stages/0 calls/0 seconds.
