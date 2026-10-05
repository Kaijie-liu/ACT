# C124 audit — exact full-source reductions, but reporting payment FAILED

**C124 v1 is CLOSED, not qualified for advancement.** The native mixed theorem,
all3495 tests and complete ordinary HZ comparisons passed. However, terminal
review found that actual JSON serialization exceeded its fixed reservations.
The raw supervisor's `all_stages_passed=true` and worker's `completed=true` do
not cover that missing check and MUST NOT be treated as complete qualification.
This dated audit supersedes that interpretation without rewriting either file.

## Positive mathematical and actual physical evidence

All four fresh C16/K32/h6 original nonconvex sources were constructed. The
original dense/historical-masked fixtures are unchanged; heterogeneous uses
12dense+4corner-only channels; no-hit uses16 individual central parents covering
all original output positions. The unchanged uniform C122 rule selected16/16/12/0.

Every original source row was bound:2048 output equations and191104 actual
native coefficients across all four sources. The independent mixed oracle
proved6576 emitted rows/107856 coefficients across the three nonempty packets.
No abbreviated transform proof was used: exactly18432
transformed coefficients per nonempty case, **55296 total**, were reconstructed
independently. Heterogeneous residual work covered all18432 scan positions and
all128 nonzero direct occurrences; no direct tail was omitted or refunded.

| Complete source | Selected/direct channels | Before bytes | After bytes | Saved bytes | Before entries | After entries | Saved entries |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Unchanged dense | 16/0 | 1095680 | 820816 | 274864 | 94353 | 76009 | 18344 |
| Historical masked | 16/0 | 863560 | 786648 | 76912 | 74626 | 72778 | 1848 |
| Heterogeneous | 12/4 | 865400 | 719944 | 145456 | 74790 | 66894 | 7896 |

Each after-state includes the owned16-byte/16-entry routing mask. These are
actual complete C91 source-state numeric comparisons, not packet-only savings.
Ac nnz falls76417→40961,57697→38737,57833→34121. Auxiliary counts are1728/1728/
1584, and complete old+new emission799744/768768/683264. Full old positive-entry
reserve131072 is retained; all three mask-inclusive packet entry deltas are
nonpositive. No old budget is assumed zero or refunded.

Original control sources remain unchanged. Candidates preserve binary/EQ/INEQ
semantics, value maps, frame/UID identities, source quotient and original input
inverse. Replaced output EQ rows and exact owner-incidence updates are checked
independently. Complete C91 audits cover all1601/1457/1461 original physical rows
and every newly written row. Original source fingerprints agree before/after
each transformation. Complete
original inverse recovery passed; diagnostic points are NOT network witnesses.
The zero-hit case retained the exact original fields object with no packet,
no kernel preparation and no auxiliary coordinates. All512 zero-hit outputs
were still bound; no smaller output population substituted for the guard.

## Numerical/resource observations — not a complete payment certificate

All**3495tests/151files** passed: collection+execution52.49510169774294s,
pytest41.18s,13 inherited warnings, zero failures/errors/skips. All3470 inherited
tests are unchanged. The25 additions independently eliminate actual mixed rows,
test cross-route shared-ID cancellation, binary/inverse semantics, tampering,
mask custody, no-op and fail-closed native/payment guards.

Recorded work counter222618270 is below the254M category sum and256M cap, but
the omitted reporting validation below means this counter is NOT a complete
prepaid-work certificate. No retroactive debit repairs the original execution.

| Category | Recorded | Frozen ceiling |
| --- | ---: | ---: |
| Original source reservations | 100000000 | 100000000 |
| Setup/route/source geometry | 1102144 | 4000000 |
| Every original row binding | 20085472 | 24000000 |
| Complete construction | 6895616 | 10000000 |
| Independent complete native proof | 56569728 | 60000000 |
| Complete source/owner/inverse | 16167744 | 24000000 |
| Fingerprints/physical/metadata/report reserve | 16569182 | 24000000 |
| Packet/case evidence | 5228384 | 8000000 |

The constructor and independent-proof charges equal their preregistered scalar
bounds. Combined retained numeric storage is6845956B/770827entries/408storages.
Known nonoverlapping metadata is26453156B with8 explicitly opaque inherited
objects; this is not a complete Python allocator-occupancy measurement.
BOTH1GiB measured limits passed: RSS growth347582464B; tracepeak158932938 plus
tracer metadata95109536B. Measured build28.079728s, worker34.194014s, complete
supervisor91.031080s including tests/identity checks. No speedup claim.
Hash traffic4642 source calls/20083320020B and3 evidence calls/1670952B is
separate from generation counters; no all-CPU accounting claim.

## The failure: compact guard, indented writer, unguarded aggregate

`c124_mixed_source_worker_v1.py` checks case length using default compact-ish
`json.dumps(...,sort_keys=True)`, but `_atomic_exclusive_json` writes sorted
**indent=2 plus newline**. Large full reference-point lists are serialized in
each case and duplicated in the aggregate result with deeper indentation.
The final524288-byte ledger/result reservation also lacks an actual size guard.

| Artifact group | Actual bytes | With1024/header per file | Fixed reservation | Excess incl headers |
| --- | ---: | ---: | ---: | ---: |
| dense_complete_source.json | 218136 | 219160 | 131072 | 88088 |
| masked_complete_source.json | 207065 | 208089 | 131072 | 77017 |
| heterogeneous_complete_source.json | 207414 | 208438 | 131072 | 77366 |
| noop_complete_source.json | 6235 | 7259 | 131072 | 0 |
| complete_held_ledger.json + result.json | 1276241 | 1278289 | 524288 | 754001 |

The aggregate consists of117520B ledger and1158721B result. It exceeds its
reservation even WITHOUT headers. Partial constructor reports are2544/2546/
2559B and fit their individual65536 reservations. Unused category/global
headroom does not retroactively finance the overflowing fixed serialization
reservations. The failure is in experiment accounting/serialization, not a
counterexample to the mixed HZ theorem; both facts must remain visible.

## Terminal disposition

Session10724 exited0 and is fully consumed; no owned numerical job remains.
All raw logs/XML/inventory, three complete native NPZs, construction reports,
four full source reports, ledger, phases/result/exit are retained unchanged at
`results/c124_mixed_source_20260927_v1`. Source/input/production drift is false.

- exit SHA256 `01c44e0f5aaa08025a241df83692f8cbfd0c98c52735ac6f35b69e5ce9c42aab`
- result SHA256 `a4fa77b2d99a0fc166886b0d51b2c1142c80c2bd813763ad4872a3ab03d401ac`
- prereg SHA256 `50c1330ff294e6da0b31a850362bb45e62df6100fdc7ef0280f607cef29f50fe`

The terminal integrity record must set complete qualification false and record
all reporting overflows, while preserving the positive mathematics/physical
evidence and the contradictory raw green flags. No edited old files or rerun.
Next: [complete kernel proof and correctly paid evidence](C124_KERNEL_PROOF_HANDOFF_20260927.md).
No original target-network source, joint real plan, LIVE/solver/shadow/replay
or score admission. Formal1870/2413 and separate CIFAR25/Tiny36 unchanged.
Full goal ACTIVE; all13-family/every-old-solve protections remain in force.
