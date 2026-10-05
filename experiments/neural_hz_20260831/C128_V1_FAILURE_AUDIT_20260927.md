# C128 v1 failure audit — full tests pass, first complete case rejected

Disposition: **C128 v1 is CLOSED and FAILED.** All3632 tests pass, but the
complete worker rejects the dense case at semantic metadata comparison. Only
the original input and complete old/new dense source stages are saved. No
complete four-source qualification, independent full-source owner/inverse
qualification, actual held-root ledger or new target admission was completed.

This independent audit uses saved JSON/XML/logs and one complete streaming-hash
check only: all15 run artifacts listed by the exit record,2515 frozen dependency
hashes and3 frozen input hashes agree. No experimental module was imported,
proof/test rerun, numerical NPZ payload analysed, model/HZ restored or solver
executed. All nine frozen C128 v1 files and historical artifacts remain intact.

## Terminal and test facts

Run: `results/c128_channel_support_20260927_v1`, relative to
`experiments/neural_hz_20260831`. Session57940 is terminal, exited1 and consumed.
The supervisor reports `all_stages_passed=false`, `tests_exit=0`,
`worker_exit=1`, and `formal_gain=0`. Worker failure is exactly:

```
ValueError: full source graph/frame/predicate/UID metadata differs
```

The complete XML/inventory contains3632 unique passing test cases in156 files:
3584 inherited qualified tests plus48 new cases, zero failures/errors/skips.
Collection plus execution57.23885356262326s fits the unchanged60s gate; pytest
reports44.62s and13 inherited warnings. Those unit tests are qualified evidence
of their own bounded incidence/graph cases, not completion of the failed
four-source worker. C127 v2/C126/C125 remain ordinary-only qualified; C127 v1
and C124 remain failed. Nothing in this audit rewrites those outcomes.

## Exact cause from saved metadata, without a numerical retry

Recursive comparison of `dense_old.json.metadata` and `dense_new.json.metadata`
finds nine scalar differences. Eight are the preregistered support-work changes
already normalized by the comparison function: one graph node's support work,
the matching count record, and six top-level work totals. The remaining field
is a nested resource diagnostic that the v1 comparator did NOT normalize:

| Field | Old | New |
| --- | ---: | ---: |
| `report.alias_quotient.coupled_extra_capacity` | 30897272 | 30983608 |

This is the capacity of the original coupled-work pool, copied by
`c64_birth_quotient_v1` from `pool.capacity`, not a coefficient, row, ownership
event or cap setting. Both builders retain source whole/branch limits32000000.
The saved values satisfy their exact defining equation independently:

```
old capacity = min(32000000-790648, 32000000-1102728) = 30897272
new capacity = min(32000000-704312, 32000000-1016392) = 30983608
```

Their difference86336 is precisely the saved dense support-work saving:
159552→73216. The total generated-source upper decreases1038508→952172;
largest-branch upper decreases1350588→1264252. Actual coupled work remains
247860 in BOTH reports, and the complete nested actual work-parts inventory is
otherwise identical. These are retained partial-stage logical counters, not a
successful complete C128 qualification or a wall-clock improvement.

Because the v1 predicate compares this nested remaining-capacity field as if it
were semantic, it rejects BEFORE the explicit old/new numeric comparison loop,
independent `actual_words` owner oracles, complete inverse-point population,
remaining three source fixtures and aggregate ledger. The failure is real under
the frozen comparator. Earlier static review missed the nested diagnostic and
does not override the failed gate.

The full old/new dense numeric NPZ files have the same authenticated byte hash.
That independently establishes equality of the saved archive bytes without
loading their arrays; it does NOT pretend that the later in-worker proofs ran.

## Complete retained stages and exact evidence payment

The stage design preserves all completed source/control/candidate evidence
before the rejected comparison. Each record states
`proof_completed_at_save=false` and `formal_gain=0`.

| Saved stage | Arrays | Numeric entries | NPZ bytes | JSON bytes |
| --- | ---: | ---: | ---: | ---: |
| Dense original source | 29 | 12246 | 88136 | 3144 |
| Dense old complete source | 48 | 178267 | 1069810 | 9323 |
| Dense new complete source | 48 | 178267 | 1069810 | 9319 |

The125 saved arrays contain368780 logical entries. NPZ payment is
`16*368780+3*1024=5903552`; three stage JSON allowances add196608. All three JSON
files are exactly canonical compact bytes and actual bytes+1024 fit65536 each.
Their complete manifests agree with the fixed inventory, including CSR data,
indices and indptr rather than confusing NPZ entries with C62's convention.

The result is1640 bytes;1640+1024=2664<524288. Its exact-byte success allowance
was prepaid even though its CONTENT correctly reports failure. This is not a
successful qualification or a reduced-success fallback. Failure65536 and
ledger131072 allowances were also prepaid and unused; no refund occurs.

There are no owner-stage archives, full-point JSON files, completed case report,
masked/heterogeneous/no-hit files or complete held ledger. The planned25 JSON
receipts,16 NPZ files,127 arrays per full case and eight complete source builds
were NOT achieved: only two returned dense builds and their three saved stages
exist. Failure retained those artifacts; it did not invent absent later stages.

## Paid work and partial observations

| Completed/reserved category | Paid work |
| --- | ---: |
| Full old/new dense source reservations | 64000000 |
| Fixture creation and three original-source bindings | 281096 |
| Packet/source headers and complete comparison prepayment | 2868656 |
| Three complete numeric stage archives | 5903552 |
| Three stage JSON reservations | 196608 |
| Upfront terminal success/failure/ledger reservations | 720896 |
| Total | 73970808 |

No independent owner/inverse proof charge was reached. Comparison was prepaid
before rejecting metadata; unused work is not refunded. The failed total is
below the unchanged limits but is an INCOMPLETE run counter, not a complete
254M/256M payment certificate. No actual C62 retained-root/64M-entry ledger was
produced, so no complete-state occupancy claim follows.

C41 explicitly reports `build_returned=false`. Its partial observed RSS growth
is9641984 bytes; trace peak5349434 plus tracer metadata592736 equals5942170,
both below1GiB for the aborted prefix only. Partial build elapsed1.786044426s,
worker8.330806429s and supervisor70.459445890s are not full-source performance
measurements. Source authentication5030 calls/20141465234 bytes and evidence
hashing3 calls/2227756 bytes remain outside the generation-work counter.
Source/input/production drift is false; no numerical job remains running.

## Artifact authority

| Artifact | SHA256 |
| --- | --- |
| `exit.json` | `243e8a544c877553227c418ee84e32850eb5a2777a34445a28dc6b682934720d` |
| `result.json` | `ba666689328184398facfd4164399c58b39a323b65580fc1022b3c52d8320584` |
| `preregistered.json` | `8969ccce79d5b5f00996868a40908917ecbcd8daedbf92fcb16e7e30cfe3161a` |
| `tests.xml` | `5145adf740e77b4b90fa343521b0327f1f6d2d0731d33fd277320e01d17ce61a` |
| `dense_source.json` | `8c439e794f45e778d309f9d85baf0f11804e790d19de1cd7eb64a4531cdfdb25` |
| `dense_source_arrays.npz` | `0c3f6bc29b15075d3e4c0ad8e13223e99b5b832eae710b846c174b6d9f142fa5` |
| `dense_old.json` | `9d100f29e2658f02d8f3a716ef6804c8701142d88f38c0101c8e2cd5385441c3` |
| `dense_new.json` | `b7a0c8924a1c7b624f1b40e58ee75040054353768d07d86c5827bfc3492820d3` |
| `dense_old_arrays.npz` and `dense_new_arrays.npz` | `71a357583a941cdb4e0a85dc7eebd0105fa42e824537024d1e5bbaaa055da71f` |

The exit hash binds all15 individually hashed artifacts, including collection,
inventory, phase and worker logs. All16 files including exit remain exclusive
v1 history. The nine frozen source/prereg/preflight hashes remain equal to their
authenticated preregistered values; no v1 code or test is repaired in place.

## Next-version boundary

A separately frozen v2 can validate the exact capacity equation against each
builder's unchanged source reservation and reported whole/branch bases, THEN
normalize only `report.alias_quotient.coupled_extra_capacity` for semantic
comparison. It must retain strict comparison of actual coupled work, work-parts,
all other nested fields and every source array. This is a corrected executable
accounting check, not permission to ignore arbitrary resource/semantic fields,
raise a cap, relax equality or call v1 successful.

All3632 passing tests remain inherited; the exact new check needs its own bounded
tests and a complete new qualification. Full eight builds/four modes, all owner
and inverse populations, stage custody, exact evidence and both memory gates
must run to completion before any v2 ordinary qualification. No such success
is asserted by this failure audit.

No new F4 packet, original target network, archived HZ, solver, LIVE, BASE,
same-structure shadow, capability concurrency test or2413/400 replay occurred.
Formal1870/2413 and independent E0 CIFAR25/Tiny36=61/400 are unchanged. Every old
solve/all13-family safeguards, production/default state, historical archives,
whole256M/branch200M and all numeric/ownership/inverse guards remain intact.
