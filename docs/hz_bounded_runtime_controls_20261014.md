# Bounded exact workspace and direct candidate controls

The optional interfaces authorized on 2026-10-14 are implemented and control-
checked. The [protocol](hz_bounded_runtime_protocol_20261014.md) was committed
and pushed as `36c0e8b91f59f495252f449aaa75b97240454ff3` before implementation.
The [execution record](hz_bounded_runtime_controls_20261014.json) binds code,
protocol, environment, all test names and the 16 differential configurations.
**Full cost remains NOT CLOSED; no new quota or real-model/GPU run is admitted.**

## What changed

`scoped_source/bounded_runtime.py` adds request-owned fixed-size working scalar
slots. Integer magnitudes are encoded in private bytearrays, with dyadic and
general rational profiles and the unchanged 8192-bit working limit. Stored
candidates retain the 4096-bit limit. A general `1/3` is accepted only by the
general profile; the binary64 profile refuses it rather than rounding it.
The endpoint workspace uses general working slots, so intermediate dyadic
optimization is not assumed. Fraction arithmetic temporaries are unchanged.

The unique-root table indexes an **already checked** V3 forest by node identity
and owner. Every ordered view membership remains checked. Different node IDs
are not merged merely because coefficients match; repeated dual rows remain
separate obligations. A same-node/different-center alias is refused by this
optional profile. The table is an additional consumer-side index: the upstream
forest, source checks and memberships still exist, and their costs still count.
It is not a direct-source format replacement or a claim of source compression.

`scoped_source/direct_candidates.py` provides typed immutable candidate frames
with request, forest, contract, ordinal, slot, query, role and dimension bindings.
Frames are consumed directly, without reconstructing candidate JSON or charging
hypothetical JSON bytes. Native binary64 and general rational profiles have
different encodings and evidence hashes. The old JSON path remains available.

`scaled_endpoints.Session` now routes both closed accessors through one
mathematical consumer. The optional workspace replaces objective, alpha and
residual scalar banks; it does not introduce a second proof rule. Sign checks
occur per row before root accumulation. Proposed and zero candidates are each
recomputed; residuals still cover every factor, with absolute values before
scale grouping. Candidate choice remains a maximum, duty coverage a minimum,
and acceptance remains strictly greater than `1/10000000` for every obligation.
The advisory candidate `lower` is not trusted. Legacy failed-slot bytes remain
charged, and wire-source terminal resealing remains mandatory.

`Driver` is the fail-closed invocation boundary, not the standalone helpers.
It allows one live tile with an owner-specific token. Immutable bytes must be
published before mathematical consumption; exceptions, stale tokens, late
completion or partial roles poison that invocation. Source/Work identity and
receipt identity remain bound. Cleanup releases owned references after cutoff
without resetting Work or cumulative bytes. Intake accepts immutable bytes,
copies them into the owned arena, and does not expose mutable buffer views.
This API is not a sandbox against malicious mutation of Python private state.

## Controls and observed results

All **55 tests** pass under the existing `act-py312`, `python -B -I -S`:

- 21 new representation, ownership, publication, cutoff and differential tests;
- 21 existing wire, source-mutation, archive and portable-checking regressions;
- 13 existing review-handoff and project-navigation checks.

The four existing synthetic fixtures, widths 1/8 and two encoding profiles make
16 configurations. **328 candidate bounds** agree exactly with both the old
Session and an independently expanded scalar oracle. Both proposed and zero
are compared, not only their selected maximum; the full aggregate also agrees.

| Fixture | Required endpoints | Unchanged complete result |
| --- | ---: | --- |
| mixed | 6 | checked positive captured real source |
| deep | 6 | UNKNOWN nonpositive |
| half | 5 | checked positive captured real source |
| guarded | 24 | UNKNOWN nonpositive |

These are synthetic interface controls, not trained-model certificates or a
new performance experiment. Native floating-point execution and checkpoint
correspondence remain unchecked. General-profile candidates include non-dyadic
values. Positive fixtures were not selected or modified to repair failures.

Faults cover magnitude/byte/allocation ceilings, root constants/owners/order,
mutable aliases, foreign/unissued/stale tokens, wrong identities/dimensions,
truncation/trailing bytes, nonfinite/negative-zero encodings, dual signs, false
zero roles, missing evidence, invalid publication acknowledgments, actual
durable-write-then-error, deadline before/during/after checking and release,
late/early/repeated terminal calls, and Work/cumulative-byte non-refund.
Duplicated-row cancellation tests are explicitly consumer-only controls, not
new source-correspondence evidence. Source mutation after the last direct
candidate is rejected by the existing full wire reseal.

The typed evidence is also replayed after copying checker code, synthetic
source and frames to a new directory, using `-I -S` with no repository imports,
checkpoint, dataset, optimizer, Torch, NumPy or SciPy. The existing wire package
regression separately retains its stricter file-access-denial portability test.
Temporary synthetic control trees are cleaned after tests; compact identities
and outcomes, not raw captured data or binary packages, are committed.

Two read-only code reviews checked mathematical equivalence and ownership.
They identified and helped repair legacy failure-byte timing, allocation-free
cleanup, publication accounting and post-release cutoff boundaries. Initial
test suites passed; later suites add these regressions. These AI code reviews
are not independent human technical review or a scientific reproduction.

## Cost contract and limits

The unchanged ceilings are 4 MiB per candidate tile, 16 MiB cumulative candidate
bytes (with the existing 8192-byte metadata reserve), 128 MiB owned arena,
8M cumulative Work and one absolute deadline of at most 300 seconds. Admission
precedes arena allocation and the immutable sealing copy; the copy's overlap
counts. No quota was raised, no mathematical check removed or moved inside a
previously unchecked cancelling arithmetic expression.

The machine record includes peak/live/cumulative owned payload, copy/release
bytes and Work for every differential. Largest observed owned payload is
1,493,526 bytes in a tiny control. This is **not RSS or a full-scale bound**.
Zero live bytes at termination means owned references were released, not that
all Python allocators returned memory to the OS or external byte aliases died.
Fixed headers/slots can cost more on tiny objects; no compression or speedup
claim is made.

Publication accounting separates attempted and acknowledged bytes. A publisher
can write and then fail: its actual partial/durable-write ledger must be read
alongside the Driver's counters. The fault test demonstrates nonzero archived
bytes with zero acknowledged bytes. A digest acknowledgment is an interface
contract, not by itself proof of disk durability.

Driver phase times are local diagnostics, **not an additive complete request
cost decomposition**. Failure cleanup can be outside a phase, while other
cleanup fields overlap. The same Work deadline includes these operations;
outer request-start to final-cleanup wall time is authoritative. The standalone
control suite wall time is only test runtime. This stage does not replace or
claim to have tested a whole-process hard supervisor for the new path.

Still open: upstream source/checkpoint capture and checking storage; planning
temporaries and Python metadata; Fraction/PyLong scratch and allocator lifetime;
coexisting native candidate arrays; external copies, process-tree lifetime,
complete durable request manifests and total archive copies. The earlier
~96.8 MB design envelope does not describe this implementation. Historical
static source-anchor reports retain their original hashes/commits; their
assumptions are not silently rebound to this refactored consumer.

## Reproduce and next boundary

From the repository, print the control report without installing anything:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c "import sys,runpy; sys.path.insert(0,'.'); runpy.run_module('scripts.run_bounded_runtime_controls',run_name='__main__')"
```

The next bounded task is to reconcile the implemented layouts and lifetimes
with the **complete** source/native/checker cost contract, explicitly retaining
unclosed terms. Do not use these finite controls to admit the registered model,
repeat sealed searches or announce G1–G6 completion. Any quota change or actual
model/GPU execution still requires separate approval. The default/native ACT
candidate path is not switched to this optional Driver.
