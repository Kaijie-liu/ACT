# Bounded scalar batching and separate source-cost diagnosis

Goal: keep bounded additional serialization memory without the measured
per-scalar Python overhead of R1. This is evidence engineering, NOT a new
relaxation, source guarantee, verification endpoint or theory claim. Start
8aacd64b2cf137f7d7ec43de25ce1d191f45367f, clean/synchronized. No real requests,
sealed input reopening, extra native solves, sample selection or production
default change. All573 prior source/protocol bindings remain unchanged.

## Mechanism and controls

Only short scalar ARRAY runs are grouped for standard-library JSON encoding.
Frozen limits:128 scalars and conservative16384 escaped bytes/batch; outer
write block65536 bytes. Every scalar is visited and validated; every source
entry appears in the same order. Long strings, large scalars and nested
containers use the R1 bounded traversal. R1 depth, key, type, finite-value and
integer bounds remain; same canonical bytes, hashes, receipts, exact checker,
full original obligations and numerical acceptance. No cache or deduplication.
Atomic exclusive publication, fsync and inherited deadline are unchanged.

Controls:21 new/inherited-on-new-implementation tests plus all137 previous
regressions. Include batch boundaries, depth, all scalar inventory, long/escaped
values, missing/altered duties, model/source/guard bindings, partial/late files,
blocked writes, exceptions, same300s full synthetic pipeline and cost closure.
No new implementation may enter a timed study before the controls pass.

## Study A: finite publication comparison

Same two pre-existing SYNTHETIC payload sizes2^18 and2^21. Three writers:
legacy whole-buffer, R1 streaming, new batched streaming. Three repeats each,
rotated writer order per input/repeat. Two separate instrumentation strata:
tracemalloc ON for additional peak workspace, OFF for throughput.36 calls total.
No result-based retuning of128/16384, adding repeats or sizes. Fixed30s total
per probe,2s terminal reserve, sampled8GiB,2threads, no GPU. All imports,
generation, encoding/hash/fsync, report, cleanup, reception and ledger costs
retained. OS RSS and Python traced additional allocation are not interchangeable.

Acceptance is correct byte/hash equality, bounded chunks/batches and<=1MiB
traced additional serialization on these fixtures, all old proof/cutoff checks,
finite full costs. Runtime improvement is measured, never a reason to drop a
failed call. Report per-stratum/per-size paired times, all three arms, repeated
range and median. No inference of real full-request speedup or new certificates.
Traced and untraced times must never be pooled.

## Study B: separate unchanged construction/check diagnosis

Two fixed declared synthetic graphs from the existing generator, seed724:
(E4,C3,width4,depth1), (E8,C10,width8,depth2); each once,300s,2s reserve,
sampled8GiB. No model/dataset/checkpoint or native LP optimization. The existing
shared router proposal, retained construction and independent source-check
algorithms are unchanged. Publication explicitly uses OLD R1 streaming, not
the new writer. Thus this is NOT another factor in Study A.

Time nonoverlapping expert propagation, factor join, guard, property projection,
weighted LP construction and their checking counterparts; report repeated
route checking, publication and residual overhead separately. Include unchanged
whole-object identities. Preserve source/bundle/report when available, no
attempt to reconstruct unsaved4099 matrices. Instrumentation remains charged.
Fresh saved-only archive invokes independent source checks without producers
or solver imports; it does not produce positive output lower bounds.
These component times motivate future tests, not a causal claim about real4099.

## Execution and decision

Commit tested implementation, then freeze identities and finite call roster in a
new directory; commit freeze before executing once. Keep timeout/error/missing
artifacts and all denominators. Saved-only audit verifies source binding, exact
byte comparisons, source checks, finite cost accounting and complete roster.
Later administrative auditing is separately costed, not added to request time.

If batching helps, keep optional until a separately scoped full-request test;
if not, retain the negative result. Source-cost profiling may identify a next
single-factor engineering intervention, but no automatic optimization/query is
authorized. Historical source-gap, external competitiveness and high-accuracy
certification claims stay unchanged. ISSTA readiness requires honest guarantee
scope, meaningful comparison and independent human review, not this microbench
or a promised venue outcome.
