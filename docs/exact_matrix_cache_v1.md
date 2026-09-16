# Exact matrix parsing reuse V1: contract and frozen offline timing

User authorized implementation, poisoning/binding/uncached controls FIRST,
then timing. This is a separate optional checker adapter, not an update to a
frozen verifier experiment. No source under act/scripts/moe_evidence/
portable_proof/evidence_cohort/evidence_handoff changes. No real-model request,
solver proposal, training, gate refinement or old TIMEOUT promotion is planned.

## What is reused

`exact_matrix_cache/cache.py` retains only immutable exact CSR parses: shape,
tuple-of-tuples rows, and a read-only coordinate→Fraction map. It reserializes
current supplied matrix content on **every lookup**. The key binds request ID,
parser policy and SHA256 of the full canonical JSON matrix (including shape,
indices, pointers and numeric representations). On a hash hit, canonical bytes
must also match; a forced hash collision rejects rather than adopting an old
parse. Mutable object IDs, filenames, a claimed matrix hash or an old proof
verdict are never cache keys/authority. Floats keep exact binary rational
interpretation; there is no approximate arithmetic or conversion tolerance.

Each miss parses a fresh snapshot of the canonical bytes with the original
strict CSR row validator. Cached and uncached adapters both reject malformed
noninteger dimensions/indices; on some malformed source matrices this is
stricter than the old `_entries` helper, never a weaker acceptance rule.
Supported canonical serialized HZ/LP inputs and all mathematical obligations
remain unchanged. All caller-visible parsed structures are immutable.

Cache lifetime is one request check, cleared also on exceptions; there is no
persistent cache load API or cross-request reuse. LRU retention caps are fixed
at64 entries,64MiB canonical payload and2,000,000 stored nonzeros-plus-rows.
Oversized parses are validated but not retained. These bound retained cache
representations, not transient parsing allocation or total Python RSS. Eviction
changes only work, not arithmetic. Stats expose hits/parses/evictions/peaks and
confirm zero live entries after the request. A cache is trusted checker state,
not a defense against arbitrary Python code execution/memory modification.

## What is still checked every time

The adapter uses original mathematical function code objects with **fresh,
function-local global dictionaries** to bind parsing operations. Original
modules and global functions are not monkey-patched. The substitutions are:

| Original operation | Optional binding |
|---|---|
| HZ export and McCormick `_entries` | request-local immutable exact map |
| Sparse LP dual evaluator `rows` | same request-local exact row tuples |
| Their `check` dependencies | original dual-check code with the above parser |
| Request aggregator's checker dependencies | the above local functions |

Original identity, source/property/range/guard binding, projection and
constraint comparisons, signed-dual evaluation, residual correction and
complete-obligation aggregation still execute. Dense interval-fact checks
remain the original implementation. No source validation, support bound,
projection, dual result or SAFE verdict is memoized. A matrix reused across
two sources is only the same numerical array; each source's whole identity,
frame, expert order, property and proof domain must independently pass.
The uniform upstream network→HZ/source/guard/route-exclusion trust assumptions
and positive threshold are unchanged. This optimization is not a new theorem.

## Control gate before timing

Controls cover immutable output/alias mutation; shape/index/value changes;
deliberate digest collision; independent request caches; refused persistent
cache injection; eviction/oversized fallback; noncanonical pointers/columns,
bool and nonfinite values;80 seeded random canonical CSR cases; original,
cache-disabled and cache-enabled complete exact results across E/C/tie/partial
reuse/negative/missing-evidence controls; semantic changes after transport
rehashing; per-property dual/source/guard/frame binding even after earlier
properties warm the cache; invalid file hashes; original globals unchanged;
and timeout on cache hit. A stdlib-only `python -I -S` control checks the same
analytic evidence with neither torch, numpy nor scipy imported in the child.

The source-bound controls receipt includes original checker dependencies and
all new code. Timing refuses to start before its PASS, with any changed source
or a dirty/unpublished branch. Old experiment source freezes must still pass.
This is not full integration into the production or portable bundle schema:
the isolated control builds a temporary checker tree; no old published bundle
is rewritten or silently assigned a new trusted checker identity.

## Frozen offline measurement (not a new verification experiment)

Subject: the first already archived saved precheck, rank0/input114, bound to
parent archive SHA256
`67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe`.
Keep its stored HZ, all9 properties, all ranges and certificates unchanged.
The expected exact result is UNKNOWN_NONPOSITIVE with3 positive/6 nonpositive
obligations. Do not choose another record based on timing or positivity.

Fixed sequence, each in a fresh process with no parsed data shared:

1. Original reference checker.
2. New adapter with cache disabled.
3. New adapter with cache enabled.
4. New adapter with cache enabled.
5. New adapter with cache disabled.
6. Original reference checker.

The reversed second block exposes order effects without claiming adequate
replication for statistical inference. Both original and optional paths are
imported in each child before timing, but no proof matrices are preloaded.
Checker time includes referenced-file loading, full checker work and cache
cleanup. Report separately full-process time (startup, identity checks, inputs,
imports, result submission included), Linux peak-process RSS and cache counters.
All six exact result objects must equal the saved precheck. There is no
cProfile. Never compare these timings to the earlier102.85s profiled run.

One process/one numerical thread, CPU only, nice10. Check existing resources
before each job; minimum16GiB RAM/5GiB disk and load/core<=0.5. Each offline
checker has a300s owned-process watchdog/280s cooperative limit. This is not
extra budget for the original request. ERROR/TIMEOUT stops the study, retained
without retry, sample substitution, limit tuning or automatic extension.
Raw destination: `data/moe/results/exact_matrix_cache_benchmark_20260916_v1`.
All observed costs and incomplete jobs stay visible. Archive the result even
if caching is slower or uses more memory. No end-to-end speedup or extra SAFE
can be inferred from this single stored-case measurement.

After the implementation/control/protocol commit is pushed, run once:

```
python -m exact_matrix_cache.controls
# clean published feature HEAD and PASS controls required:
nice -n 10 python -m exact_matrix_cache.benchmark
```

Any production/portable integration or real-request evaluation is a subsequent
separately bound stage, not implicitly authorized by this timing protocol.
