# Request-store device lifecycle: bounded CPU and simulated controls

The [frozen protocol](hz_store_device_supervision_design_20261013.md) has completed
all ten once-only calls. Final offline audit V2 returns **PASS, 0 issues**, with
12 interface controls and six additional accounting/batch-stop controls passing.
The first offline audit failure is retained. No numerical request was rerun,
no physical CUDA initialized, and no real model was loaded. All G1–G6 remain OPEN.

## Integration, not a new bound algorithm

The [policy](../scripts/hz_store_device_supervised.py) and
[worker](../scripts/hz_store_device_worker.py) connect the existing immutable
common/private block candidate algorithm to the existing five-phase lifecycle:
admit, produce, release, independently check, receive. The earlier CPU policy
only factors its existing payload/check/reception validation into shared helpers;
its default mathematical and numerical contract is unchanged.

The optional CUDA context binds the actual newly generated store, request,
property contract, absolute producer deadline, invocation, GPU UUID and 1 GiB
allocator limit. Import, source construction and planning cannot initialize CUDA.
A fresh device observation must pass immediately before a CUDA-intent record is
published and initialization becomes possible. The physical path requires a new
execution freeze; the ten controls provide no such permit.

Each request retains one 300-second clock, sampled combined parent/worker RSS
limit of 2 GiB, a 64 MiB archive cap, and the cumulative 96M exact-composition
work limit. CPU/float64, eight-column tiles, 128 candidate updates and all tie-legal
pair/property/distinct-endpoint duties are unchanged. Candidates are independently
checked against original bound source rows; partial or late evidence cannot
establish a complete positive request. Numerical candidates and exact checking
remain different trust levels.

Release confirmation means **the owned producer PID is absent in two device
observations after process cleanup**. It is not proof that all driver allocations
have been reclaimed, and it never admits the next request. A later physical call
must obtain fresh admission. Unknown real cleanup/release stops batch continuation;
only the two explicitly registered no-CUDA release simulations may continue after
their expected refusal. Other users' jobs are never terminated.

## Frozen outcomes

Times below include stage processes, parent overhead, final publication and
observed API return. Short fault phase caps are inside, not replacements for,
the common 300-second request clock. These are controls, not speed benchmarks.

| Call | Execution terminal | Mathematical result / preserved evidence | Full observed seconds |
| --- | --- | --- | ---: |
| half | CHECKED_STORE_DEVICE_EXECUTION | 3/3 positive duties, 5 checked endpoints | 2.936460 |
| negative | CHECKED_STORE_DEVICE_EXECUTION | UNKNOWN_NONPOSITIVE, 3/6 positive duties | 2.997920 |
| partial | CHECKED_STORE_DEVICE_EXECUTION | UNKNOWN_MISSING_EVIDENCE, 2/3 duties; one endpoint missing | 2.941060 |
| admission_busy_stub | RESOURCE_UNAVAILABLE | no producer launched | 0.257784 |
| pre_cuda_busy_stub | RESOURCE_UNAVAILABLE | source built; no CUDA intent or candidate execution | 2.735317 |
| candidate_exception | ERROR | actual first matrix-transfer seam raises | 2.410929 |
| readback_delay | TIMEOUT | one published tile, no complete proof manifest | 15.356521 |
| release_owned_stays | DEVICE_RELEASE_UNCONFIRMED | simulated owned PID remains | 2.618665 |
| release_delay | DEVICE_RELEASE_UNCONFIRMED | release worker timed out | 7.220059 |
| wrong_candidate_metadata | ERROR | completed candidates rejected for wrong device identity | 2.560963 |

`CHECKED_STORE_DEVICE_EXECUTION` is an execution/accounting result, not a synonym
for mathematical SAFE. The half result concerns a small synthetic declared-real
source; it proves neither trained-checkpoint correspondence nor native floating-
point execution. Its empty guarded branch can have a vacuous bound. Negative
checked lower bounds do not establish UNSAFE. Missing duties remain missing.

The audit checks complete phase/status prefixes and exactly one correctly placed
fault witness for each fault call. It rechecks all three normal proof records
with the existing independent standard-library mathematical checker; this is
not a second mathematical implementation or independent human review. Every
control retains its terminal and complete cost denominator.

## Retained accounting failure and correction

The first audit stopped with `unexpected nested operation`: the candidate stage
contains a nested pre-CUDA recheck, while the older event parser only accepted
flat operations. The worker, its frozen source snapshots, logs, source objects,
candidates, budgets and mathematical results were not changed.

The [supplemental V2 auditor](../scripts/audit_store_device_supervision_v2.py)
reproduces and preserves that failure, archives its own source, and then audits
the original records using a stack-based event reader. Only depth-zero operation
times are additive; child times are diagnostic. Unfinished durations are `null`
and right-censored. Their observed phase spans include unseparated cleanup or
parent work and are explicitly **not bounds on operation duration**. The release-
delay fault has no operation interval, but its full phase cost remains charged.
Do not interpret unavailable operation timings as zero.

All phase totals and parent/publication overhead remain authoritative; the
named operation rows are not an exhaustive, disjoint decomposition. V2's full
offline audit takes 0.214012 s; including reproduction of the first failure,
the pure supplemental controls and source snapshot takes 0.400983 s before
final report publication. These are additional offline costs, not deductions
from or additions to the original per-request execution result.

A read-only review caught the original nested-parser incompatibility and the
censored-span wording. Both are addressed in the supplemental audit rather
than silently replacing historical execution. The six supplemental tests cover
nested accounting, malformed events, censoring, fault placement/duplication,
batch API exceptions and batch stop on unknown device release.

Separately, `act-py312 -B -S -m unittest` ran the policy, previous supervision,
endpoint, source/store, capacity, layout and proof-closure suites: **88 tests,
0.854 s, OK**. This is tool-observed regression output, not an additional archived
test-outcome ledger. Twelve of those tests overlap the frozen policy controls;
do not add all test counts as distinct coverage.

## Archive and next decision

Protocol commit `a3a521fe4`; execution commit `9ebd9586f`; supplemental auditor
execution version `cbea561a6`. The [compact binding](hz_store_device_supervision_20261013_r1.json)
identifies all original and supplemental receipts. Raw archive:
`/data1/Kane/MOE/baseline_runs/hz_store_device_supervision_20261013_r1`.
After final audit it contains 539 regular files and 1,191,595 bytes, below 64 MiB.
No checkpoint, raw dataset or external repository is committed. `/data1` has
about 87 GiB available; no cache deletion was needed and no evidence was removed.

Stop extending these lifecycle controls. The next device task is a **separately
frozen finite physical comparison of this same store/block algorithm on CPU and
GPU**, after fresh UUID/resource admission, with source capture, transfers,
synchronization, exact acceptance, reception and cleanup all charged. Bind the
V2 audit and its source in that new permit; do not rely on the failed flat parser.
Do not retry the old eight unstarted GPU slots or claim a speedup from this study.
Captured trained-source streaming and convolution support remain separate
capacity/closure blockers; no sealed real input or old holdout is admitted.
