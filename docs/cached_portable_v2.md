# Optional cache in a portable weighted top-2 checker (V2)

Scope: integrate the already controlled exact CSR cache into a **new** portable
package and an optional evidence-tail execution path. No old bundle, source,
numerical gate, cohort terminal, model, sample or solver configuration changes.
No new real-request experiment is authorized by this module. The cache is OFF
by default and explicitly enabled when building the new package/execution.

## Portable identity and conditional guarantee

`cached_portable.pack.pack` builds only into a fresh authorized destination.
It uses the unchanged V1 content-addressed array transport and stripped
solver-free mathematical modules, then publishes a distinct schema,
`PORTABLE_WEIGHTED_TOP2_CACHE_V2`. The bundle hash binds the statement, source
references, all executable bytes, cache policy and boolean enabled flag.
Changed cache mode means a different bundle identity; no runtime CLI switch
can override it. Legacy V1 bundles/checkers are not upgraded in place.

The launcher requires `python -I -S`, independently pins executable contents
before importing bundle modules, and forbids external reads/writes, model or
solver imports, subprocesses and network. Only the bundle and interpreter
stdlib are readable. As with the old checker, the caller must trust the
checker entry implementation and obtain the expected bundle/statement hashes
from a trusted channel; executing attacker-supplied Python is not a sandbox
security guarantee. Tests rehash selected analytic mutations deliberately to
exercise semantic checks beyond transport integrity.

The exact parsing cache remains request-local, immutable and bounded by the
previously frozen64-entry/64MiB-payload/2,000,000-cell policy. Every logical
file and source/property/guard/range/dual/coverage check still executes. There
is no memoized proof verdict. Parsed and transport caches are cleared/closed
after the check. The V2 result equals the original mathematical result, not a
new or stronger safety grade. Complete positives remain conditional on the
unchanged network→HZ/input-source, guard lowering and route-exclusion trust
assumptions. Missing/nonpositive proofs remain UNKNOWN, not UNSAFE or SAFE.

Offline relocated check (same bundle hashes, new local finite checker clock):

```
python -I -S /new/location/verify.py \
  --bundle-hash HASH --statement-hash HASH --timeout-seconds 300
```

This command's clock is an offline checking allowance, not an extension of a
historical verification request. In-budget execution instead supplies the
original request's absolute monotonic deadline on the same host/boot.

## One request clock and terminal admission

`cached_portable.execution.supervise(source, request, destination,
started=original_request_start, enabled=True)` is the optional admission entry.
It reads saved evidence only; it does not capture networks, generate proposals
or select a cohort. There is deliberately no default fresh `started` value.
Future upstream integration must pass its own original request start, rather
than invoking this as an extra300s after a failed run.

The whole owned tail-driver process shares an absolute `started+298` work
cutoff; so do its precheck, packaging and isolated-check subprocesses. The
checker's cooperative tick covers file reads, identities, exact matrix cache
operations, full mathematical checks and response serialization. A native
call that does not cooperate is stopped by the owned-process watchdog. Only
owned processes/descendants are stopped, using the existing identity-aware
cleanup. No new resource setting, proposal cap or80s reserve change is made.

All startup/import, source loading/hashing, optional cached precheck, packing,
isolated checker startup and work, result reads, inventories, candidate writes
and child cleanup are included in the parent-observed request clock. Logs give
stage start/end/allowance, prior cost at entry, tail overhead, candidate
publication observation, whole-driver exit and outer review observation.
The latter includes inner terminal serialization, which a checker-only timer
does not. This is a total budget, not300s for each stage.

`run()` and inner candidate/admission files alone are NOT authoritative for
the whole request. `supervise()` admits only a successful, timely driver and
read-only terminal review. Missing stages, unsuccessful exits, killed drivers,
or any observed completion past the total300s cannot promote positive-looking
files. Late terminal publication emits an overriding TIMEOUT marker. Outer
metadata writes are deadline-checked too; markers invalidate a late ledger.
Filesystem cleanup/timeout logging can finish after300s: this does not promise
a real-time OS latency bound, but no late positive is accepted. Last observer
record I/O is not recursively self-timed; whole-driver costs include all inner
record I/O, and outer publication has explicit post-write deadline checks.

`audit_outer()` recomputes admission from driver/phase states and timings,
request/start/cache identity, source hashes, precheck-vs-isolated results and
package hashes. It is a structural/accounting audit, not another independent
implementation of the rational proof mathematics. It never rescues timeout
requests using later files.

## Control gate

Analytic controls include:

- Copy a package to a new directory (including spaces); destroy the original
  source tree before `python -I -S` checks every required obligation.
- Cache on/off/original exact differential, multiple tie-legal pairs, changing
  expert/class dimensions, partial reuse, missing and nonpositive evidence.
- Removed obligations, changed properties/source/request, code bytes, option,
  missing executable inventory, extra unbound code and wrong statement hash.
- Explicit probes for forbidden solver import, external read and file write.
- Expired/oversized checker clocks, expiry after math but before response,
  packing expiry, native process hang, no-overwrite and discarded late output.
- Actual supervised analytic tail with a **simulated270s already spent**:
  both modes must use only the remaining28s work window, then pass final audit.
- Candidate/admission serialization crossing300s, expired outer clock,
  stage-cost/allowance/upstream mutations and no promotion from late files.

Run in act-py312 with one numerical thread/no GPU. The controls runner also
runs existing exact-cache, general evidence, reserve-handoff, cohort ownership
and cost-accounting suites. It verifies previous frozen identities before and
after, and writes numbered immutable attempt receipts (including failures).
Recorded analytic package/check times are observability controls, not a
performance experiment. Do not infer real-model speedup from them.

## Bounded saved-evidence integration check

After a passing source-bound control receipt and a clean published commit,
one optional offline tail check is fixed to **the same archived rank0/input114**
used in the previous parsing comparison. New directory, cache enabled, one
CPU thread/nice10/no GPU, same300s inclusive budget, no proposals or checkpoint
loads. It must reproduce the saved3-positive/6-nonpositive UNKNOWN result;
the historical terminal remains TIMEOUT regardless of this offline outcome.
Its purpose is to measure packaging plus isolated checking, not to find SAFE.
No retry, substitute case or budget/cache tuning after observation.

This is an offline stored-proof integration check, NOT a real-request study:
upstream HZ/proposal costs are absent and explicitly excluded, not claimed as
zero-cost production work. Separate analytic tests verify inherited prior
cost. Real-request deployment or a comparative cohort requires another
execution/selection freeze and terminal-audit gate, after reviewing this stage.
