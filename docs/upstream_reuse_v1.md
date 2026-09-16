# Upstream decoded-source and exact-CSR reuse V1

2026-09-16. Implementation/control stage only, based on the saved-log cost
analysis in `upstream_generation_cost_v1.md`. No sealed real requests, new
samples, extra solver time or support-order optimization in this stage.

## Contract

`upstream_reuse.proposal.run(root, request, budget, source_enabled=False,
matrix_enabled=False)` is a **new, opt-in research adapter**. Both defaults
are off. It is not wired into existing frozen workers or batch experiments.
Its input is the already captured generic weighted-top-2 request, not a
checkpoint or a hardcoded input. It retains the 300-second shared clock,
60-second proposal cap and 80-second tail reserve. It cannot reset upstream
capture time. The caller still needs the original outer hard-deadline
supervisor: cooperative Python clock checks do not interrupt a native solve
or filesystem operation immediately. Full-flow integration is not claimed.

Only reserve exhaustion yields an ordinary incomplete handoff. Other errors
propagate; no diagnostic or partially committed manifest establishes SAFE.
The original complete checker/portable checking stage remains required.
The adapter rejects a prior query journal, handoff or its own one-shot marker.
No cache or earlier proof result survives request cleanup, including errors.

### Source reuse

Pin source/export references to the validated request manifest. Every read
still validates root/path scope and **reads and hashes current file bytes**.
Only JSON decoding is reused; this is not an I/O-elision cache. A hit also
compares complete bytes and request scope, beyond the hash. Internally stored
trees are immutable and never handed to mathematical functions. Hits return
fresh dictionaries/lists, so caller mutation cannot corrupt future reads.
Mutable manifests are not cached. No caller-supplied persisted cache is loaded.

Limits: 8 entries, 64 MiB serialized payload and 2,000,000 tree nodes, LRU
eviction and oversized bypass. These are bounds on retained logical content,
not a promise of 64 MiB process RSS; decoded, copied and in-flight objects have
additional overhead. Freezing and copying can cost more than avoided decoding.

### Exact matrix reuse

Use the already controlled `exact_matrix_cache.MatrixCache` in the **upstream
proposal, construction and checking** path. Its previous use covered the tail
checker only. Retain its 64-entry / 64 MiB / 2,000,000-cell caps, canonical
current-content serialization, collision-byte checks, immutable exact
Fractions, width validation and request-local scope. Rows handed to the
original builder are fresh lists, preserving its list-concatenation semantics.
All canonical serialization and source-identity computations still execute.

### Checks and order deliberately unchanged

Every support export and weighted construction is independently checked both
before and after a proposal. Every successful sparse proposal retains all
three exact dual evaluations: the candidate bound, its internal check, and the
caller's postcheck. Source/objective/property/range/dual checks are not cached.
No numerical gate, proof obligation, trusted-base assumption, multiplier or
McCormick construction is relaxed. No route or property is skipped.

The original deterministic support barrier remains: finish **all** support
queries before **any** weighted query. Sorted pairs, property order, budget
grants and failure branches are unchanged. `schedule.py` and `native.py` are
small derived copies for local dependency binding; AST controls require their
function bodies/signatures to equal the originals after removing local import
statements. `_bind` uses separate function globals, never module monkeypatches.
The sparse proposal's SciPy import is lazy and still charged; its native
solver is merely timed. Dense proposals retain the original fallback.

## Accounting

Nested timers distinguish current-source read/hash/decode/copy/freeze,
canonical identity, CSR parsing/retrieval, exact construction, full export and
construction checks, exact dual evaluation, proposal imports/native linprog,
and save/reference serialization. CSR timing consumes the iterator rather than
timing generator creation. Record inclusive and exclusive costs separately;
never sum inclusive parents and children as independent costs.

`elapsed_seconds` in the adapter diagnostic is measured through cache cleanup,
before writing that diagnostic itself. The diagnostic write is checked against
the same request deadline, but complete end-to-end cost must come from the
outer supervisor, including capture, packing, final checking and all writes.
The timer's proposal-exclusive residual includes matrix preparation and other
Python work; it is not pure solver time. No real-stage timings were measured
this turn. Synthetic differential tests use a nonbinding clock for determinism
and are not performance observations.

## Controls and result

`docs/upstream_reuse_controls_attempt001.json`: **86/86 PASS**, no skipped tests,
43.279 seconds for the control suite. This runtime is not an algorithm speedup.
It binds new and old source hashes and rechecks the old frozen-source inventory.

Eight new controls cover AST identity, four-way cache on/off differentials,
alias mutation, current-byte and scope rejection, poisoned scope, LRU/oversize,
deadline checks, unbound paths, request mutation, warm-cache source/dual checks,
matrix mutation/width, reserve handoff, error cleanup, one-shot refusal, and
nested timing with exceptions. Differential cases include E=2/3/4, C=3/4,
multi-pair/tie fixtures, partial reuse and nonpositive properties. Each of the
four option combinations has identical complete manifest and unchanged
request-checker output to the original proposal loop, not just equal status.
All successful-query counts retain three exact dual evaluations per query.
The other 78 existing controls include portable relocation, pollution,
semantic tampering, missing obligations, no-solver checking and hard deadlines.
Those older full-flow tests do not constitute new-adapter full-flow integration.

The first direct 8-test invocation had two fixture filename errors; preserved
in `upstream_reuse_initial_test_failure.md`. No research result was overwritten.

## Interpretation / next gate

This establishes a candidate reuse implementation with analytic equivalence,
not a real-request speed or closure improvement. Default production behavior,
sealed eight-request comparison, query order and all acceptance checks remain
unchanged. Trusted network→HZ, guard lowering and route exclusion assumptions
remain unchanged; caching does not turn a conditional proof into a deployed
floating-point execution proof.

Next, if continued: separately integrate the opt-in adapter with the complete
upstream worker, portable tail and terminal audit, testing original-start
deadline accounting and failure/relocation first. Only then freeze a small
paired performance protocol with identical queries/checks and no ordering
change. Source-only and matrix-only switches are available to isolate costs.
Query-order optimization is a different ablation and must use a separate
execution identity; do not combine it to rescue a reuse result.
