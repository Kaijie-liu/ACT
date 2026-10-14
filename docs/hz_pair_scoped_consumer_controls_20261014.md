# Pair-scoped consumer: implementation and finite controls

The optional candidate-8/check-2 consumer is implemented, default off. Its
[protocol](hz_pair_scoped_consumer_protocol_20261014.md) was committed and pushed
as `6ee055f630fff745a5cc4a98d34b21d9fff2b955` before implementation. The
[machine record](hz_pair_scoped_consumer_controls_20261014.json) binds the code,
tests, limits and environment. **This does not admit a full model: complete
cost is still NOT CLOSED. No real model, optimizer or GPU was run.**

## Interface and mathematical contract

Use `direct_candidates.Driver(..., pair_scoped=True)` explicitly. Candidate
width must be 8; the checker uses consecutive blocks of at most 2, including a
singleton tail. Default `pair_scoped=False` retains the dense workspace.
No production/native entry is automatically switched to this optional path.

`pair_workspace.Scope` selects common ancestry (including the entire router),
the current pair guard and both experts' ancestry from a fully checked V3
forest. It excludes other pair guards and the third or later expert's private
coordinates. All global IDs survive in owned bidirectional index buffers;
equal coefficients or scales never identify factors. Scope membership and
inverse lookup are checked against source ownership. Scope outside zeros are
implicit, not dropped obligations; nonzero writes outside the closure fail.

The root descriptors contain at most four exact roots per endpoint. They are
sealed before exposure; objectives are expanded only after the complete typed
candidate frame has been published. Private node accumulators are consumed
in place, with the old handle invalidated and its slot reference cleared.
Objective and dual propagation remain separate; only their final factor
vectors are added. This preserves the old intermediate bit checks.

Legacy JSON, dense typed and scoped typed paths share `_consume_block` in
`scaled_endpoints.Session`. Each retains both proposed and explicit-zero
candidates, every ordered row (including duplicates), dual sign checks, global
factor traversal, absolute value before scale grouping, the original strict
positive threshold, and all route/property/endpoint aggregation. Partial roles
are errors; missing whole candidate pairs remain missing evidence. A later
block failure invalidates all receipts, including earlier successful blocks.

The execution order is intentionally different across columns. Each column's
arithmetic order is preserved, but first failure, deadline prefix and Work
totals can change. Exact-result equivalence on completed controls does not
imply identical admission sets under finite resources. Wire root metadata now
uses `NodeTable.descriptor()` rather than decoding a coefficient row; this also
changes the dense wire path's access/Work trajectory, not its mathematical checks.

## Controls

All **88 tests pass**: 33 scoped tests (including inherited lifecycle controls),
21 old owned-runtime tests, 21 wire/source/portability regressions and 13 review
and navigation checks. Under `act-py312 -B -I -S`, the four frozen synthetic
fixtures and two exact candidate encodings give **8 configurations and 164
proposed/zero bound comparisons**. All agree exactly with the dense typed
consumer, old Session and independent expanded-matrix oracle; complete results
also agree. These are not trained-model or performance results.

| Fixture | Endpoints | Unchanged complete result |
| --- | ---: | --- |
| mixed | 6 | checked positive captured real source |
| deep | 6 | UNKNOWN nonpositive |
| half | 5 | checked positive captured real source |
| guarded | 24 | UNKNOWN nonpositive |

Controls cover third-expert/private-factor and other-guard exclusion, source
mutation, foreign owner/Work/pair, index pollution, scope close before allocation,
immutable roots, ownership transfer, intermediate overflow before cancellation,
missing evidence, partial roles, later-block bad signs/false zero/exception/
deadline, release cutoff, full-byte publication and cumulative non-refund.
The odd-block partition test is a consumer-only control on the same complete
plan, not a change to the registered experimental tile configuration.

The new path replays copied source and candidate frames from a new location in
an isolated no-site interpreter, without checkpoints, datasets, numerical
libraries or solvers. Wire resealing still rejects a modified third expert
source even though that expert was excluded from a particular pair's workspace.
These are cooperative in-process ownership APIs, not a sandbox against arbitrary
Python private-memory modification.

Two read-only AI code reviews found lifecycle issues before finalization:
transferred handles retained an exposed slot reference; scope maps were mutable
and closed scopes were not refused before allocation. Those interfaces were
hardened and tested. The first old portability run also exposed a missing code
module in its explicit copy inventory. A subsequently added scope-close test
contained a test-local NameError; it was corrected. These development failures
are recorded, not model outcomes; no acceptance threshold was changed. AI review
is not independent human technical review or a scientific reproduction.

## Owned payload and accounting, not full RSS

Native-profile controls measured these owned-buffer peaks and cumulative Work:

| Fixture | Dense peak bytes | Scoped peak bytes | Dense Work | Scoped Work |
| --- | ---: | ---: | ---: | ---: |
| mixed | 976,698 | 341,278 | 8,505 | 12,333 |
| deep | 1,398,582 | 598,290 | 13,761 | 21,309 |
| half | 472,030 | 158,440 | 3,592 | 4,974 |
| guarded | 1,290,172 | 125,732 | 7,308 | 9,172 |

Smaller owned peaks accompany **more Work**, not a demonstrated speedup. The
general-rational profile is separately recorded; all owned bytes are released
at terminal or abort. No full-shape allocation or capacity experiment occurred.

The prior proposed 117,009,527-byte envelope is **not an implementation-bound
total**. Node/factor vectors and index/descriptor buffers are packed here, but
norm groups, offsets, dual RHS values, planning and Python metadata are not the
proposed packed scalar/group bank. Exact temporary arithmetic, the checked
source, native coexistence and external copies remain outside this arena.
Scope is rebuilt per tile in this version, not once per pair: the earlier static
pair-count scan term cannot be reused as its executed Work count.

All old ceilings remain: 4096 stored/8192 working bits, 4 MiB/frame, cumulative
16 MiB minus the 8192-byte reserve, 128 MiB arena, 8M Work and the original
absolute deadline of at most 300 seconds. No new fallback budget is granted.
Local Driver phase times overlap and are not an additive end-to-end cost table;
outer request-start to final-cleanup time remains authoritative. This stage
does not claim a whole-process hard supervisor, bounded Python RSS, complete
archive lifetime or full source/native/checker cost closure.

## Reproduce and next boundary

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c "import sys,runpy; sys.path.insert(0,'.'); runpy.run_module('scripts.run_pair_scoped_controls',run_name='__main__')"
```

The next meaningful step is to reconcile this actual per-tile implementation
with the remaining complete source/native/checker cost contract. In particular,
old source-membership, cumulative Work and full candidate-stream barriers are
not removed by lower vector peaks. Do not extend the synthetic workload or run
the registered model to infer admission. Quota changes and real/GPU execution
still require separate authorization. G1–G6 and all sealed scientific outcomes
remain unchanged; this interface stage is complete, the research goal is not.
