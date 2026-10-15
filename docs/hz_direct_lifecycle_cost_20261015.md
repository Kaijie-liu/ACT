# Direct lifecycle cost reconciliation

The V2 connection at `9d30c76f8fcf866302a304b3712e2f69aeeee0b0` releases
production objects before online replay and supports both registered consumers.
**It still does not admit the registered eight-expert topology.** Its known
membership and complete-path Work requirements exceed unchanged limits. The
fixed-stub supervisor also has a deliberately narrower input/file contract.

This [source-bound ledger](hz_direct_lifecycle_cost_20261015.json) and
[calculator](../scripts/assess_direct_lifecycle_cost.py) reconcile the implemented
V2, including parent reception. They use code and saved controls only: no source
construction, candidate generation, worker, model, optimizer or GPU execution.
The [133-test result](hz_direct_lifecycle_controls_20261015.md) remains an interface
control result, not a capacity or algorithm-performance result.

## What changed and what did not

| Boundary | V2 result |
| --- | --- |
| Source mathematics and binary format | Unchanged hashes from V1; one construction and two independent source checks |
| Candidate consumption | Two complete exact checks, with the same selected consumer in production and replay |
| Candidate lifecycle | One fixed proposer; roots are leased, read-only, charged and revoked |
| Production owners | Source bytes, first graph, Driver, plan, receipts and candidates released before replay |
| Parent reception | After worker exit and owned-group cleanup; checks identity, coverage, aggregation and cost, not another LP proof |
| Whole request storage | Includes original source and packaged source, logs and terminal metadata |
| Real/native interface | Absent: public spec admits only `half` and `guarded` fixed stubs |

The source-core upper of **140,116,400 Work** from the
[V1 derivation](hz_direct_join_cost_20261015.md) can be inherited only because its
source code and schedule are unchanged. It is not a complete-request upper.
V1's object-overlap and total I/O statements do not describe V2.

## Work and the roots access comparison factor

The registered shape has 3,072 inputs, eight experts and 252 properties over
28 conservative tie-legal pairs. Complete candidates require at least 252
endpoints; the upper is 504. Covering all these pairs is not a route-flip witness.

Three passes over first-layer weight positions alone cost at least 20,054,016
visits, including positions holding zero weights. Adding only the already-derived
input-factor merge and two norm checks gives:

| Complete execution path | Source floor | Two consumer floors | Combined floor |
| --- | ---: | ---: | ---: |
| Dense-owned | 20,054,016 | 17,031,168 | **37,085,184** |
| Pair-scoped check-two | 20,054,016 | 34,062,336 | **54,116,352** |

Both exceed 8,000,000, before planning, roots access, proposal and encoding.
These are conditional requirements of this implementation if all evidence were
completed, not observed counts, algorithm-independent lower bounds or suggested
quotas. Earlier gates prevent those completed executions.

The new fixed proposer has its own countable cost. Let `E` be the endpoint count,
`N` the node count, `k_t ≤ 4` the objective root count, `K = sum(k_t)`,
`Q = sum(eq_t + ine_t)` and `U = sum(k_t*(k_t+1)/2)`. Excluding
`Driver.prepare`, encoding and consumption:

| Fixed proposal path | Row visits | Coefficient visits | Arithmetic |
| --- | ---: | ---: | ---: |
| Dense | `2E+K` | `2NE+4K+2Q` | `E+K` |
| Pair-scoped | `2E+K` | `3K+U+2Q` | `E+K` |

On CPython 3.12, constructing the keys tuple asks `ExactMap` for its length and
iterates it: two full node-slot scans. The pair-scoped descriptor has at most
four entries. With the registered dimension uppers, roots-only coefficient
access is bounded by 19,660,032 versus 11,088. Including fixed dual generation,
the local totals are bounded by 22,387,176 versus 2,738,232. Neither is the cost
of the unconnected native optimizer or of a complete request.

Saved V1/V2 counters independently confirm the new row/arithmetic contributions
and preserve identical exact results in all 16 configurations. For example,
`guarded` adds 96 rows and 72 arithmetic visits to either consumer, but adds
1,488 versus 312 coefficient visits. No execution was repeated for this analysis.
Consequently, future two-arm timing would include **roots representation/access**
as well as check-block differences. It cannot all be attributed to the residual
algorithm. Equalizing roots access would be a separate disclosed change, not a
silent improvement to the frozen comparison.

## Logical reads and retained bytes

Let `T` be all tensor bytes including center bytes `X`, `S` forest bytes and `C`
all candidate-frame bytes. The successful online schedule reads at least:

`18T + 3X + 9S + 9C`.

The ledger lists all 15 contributing stages. The four whole-request censuses
hash contents; the last three see both source copies. Parent reception and
finalization alone contribute `7T+4S+4C`. For the registered dimensions, the
tensor terms alone are 1,002,953,088 bytes. These are logical file reads, not
measured disk traffic; metadata, code, imports, in-memory hashes and copies are
additional. Offline audit/relocation has separate costs.

With source manifest `H`, retained storage includes `2(T+H)`, plus the forest,
all frames, checker code, contract, package manifest/bundle, result, inner
terminal, invocation, progress, log, candidate report and outer receipt.
The complete formula is in the machine ledger. The worker reserves 3,276,800
bytes outside its artifact allowance and 8,192 bytes inside for its terminal;
file slots also reserve progress, candidate and receipt. These are reservations
within the original 128 MiB, not additional allowances.

For a **first-layer-all-nonzero subcase**, just the two tensor copies and explicit
first-layer forest coefficients need **191,647,104 bytes**, exceeding 134,217,728
before all other members. This is a conditional capacity counterexample, not a
measurement of any checkpoint. Actual counts remain unknown.

Two other independent gates remain decisive:

- Required ordered memberships are at least **126,632 > 65,536**. The encoder
  checks after construction; this known shape should be refused before allocation.
- The fixed-control worker has `RLIMIT_FSIZE = 1 MiB`. A format-level 4 MiB tile
  limit or 128 MiB aggregate does not authorize that worker to write a larger file.

Local/reference limits and exact binary64 source representability also remain.
Changing case names alone cannot turn this worker into a full-model interface.

## Object lifetime and the three resource ledgers

V2 fixes V1's first-graph/replay overlap. Production still temporarily holds
the builder graph with mutable and immutable encoding buffers, then source bytes
with the first checked graph. Replay has a new parsed graph; its forest byte
buffer lasts through construction/checking, not throughout every later tile.
Current tile bytes, owned arena and arithmetic temporaries still coexist.
Parent reception holds bounded JSON and hash inventories after the worker exits.

These distinctions close an ownership-accounting error, not a full RSS bound.
Python allocator arenas and fragmentation need not disappear when references
are cleared. The inherited 114,390,448-byte pair-arena bound excludes the source,
Python objects and parent; sampled 2 GiB RSS is a refusal observation, not an
instantaneous peak guarantee.

Keep three ledgers separate:

1. **Work** counts explicit rows, coefficients and arithmetic. Capture, hashing,
   JSON/struct, I/O, GC and most parent aggregation do not increment these counts.
2. **Bytes and ownership** count retained copies, reads, buffers and lifetimes.
   Archive size is neither read volume nor process RSS.
3. **Wall time** includes all online stages from `supervise()` entry, including
   worker imports, cleanup and reception. Parent pre-import and offline audit are
   outside that call. Late reception fails, but OS cleanup and final accounting
   can overrun; there is no unconditional return-within-300-seconds guarantee.

The parent restores cumulative child counters rather than resetting them. A
killed worker's unavailable final counter remains unknown. Neither the 8M cap,
300-second refusal nor small observed controls proves successful capacity.

## Decision and next implementation boundary

The lifecycle task is complete in its approved fixed-stub scope. **Stop full-size
or native execution at this gate.** No amount of candidate batching or buffer
release removes the three source passes alone exceeding 8M. The next decision
must jointly address source coefficient/membership representation, the checked
traversal algorithm and its resource policy, whole-request copies/file layout,
and comparable roots access. Merely increasing a quota from a partial bound or
compressing files while keeping every full scan is not a capacity proof.

This analysis changes no runtime or limit. Native/GPU execution and any resource
policy adjustment remain separately authorized. G1–G6, historical source gaps
and publication claims are unchanged; no new scientific certificate was produced.

## Reproduce the static audit

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S scripts/assess_direct_lifecycle_cost.py --check-report docs/hz_direct_lifecycle_cost_20261015.json
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_direct_lifecycle_cost
```

The 13 static tests cover arithmetic, code bindings, complete archived rosters,
mutation refusal, byte accounting and no-runtime scope. Together with navigation
and handoff regressions, 26 tests pass. Three read-only reviews separately
reconciled lifecycle, supervision and admission boundaries.

The [validation record](hz_direct_lifecycle_cost_validation_20261015.json) retains
an earlier combined invocation that refused a historical capacity-test hash.
That test predates direct-source integration and is not silently rebased. The
calculator instead checks its frozen Git blobs: only direct-source type admission
and the extra terminal source reseal differ. Consumer arithmetic and arena code
are unchanged; the extra reseals are charged in the V2 schedule. This explicit
migration check supports using those local bounds, not the entire old execution
contract. Old files and results remain untouched.
