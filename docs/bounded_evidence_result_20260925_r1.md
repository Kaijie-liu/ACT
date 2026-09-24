# Bounded evidence publication: memory control passes, throughput tradeoff

Completed synthetic controls only. **Zero new real requests, zero new real
output certificates.** No sealed input was opened for construction or solving;
the frozen4099 timeout/resource stop remains unchanged. Existing source and
acceptance implementations are untouched; the new worker is an opt-in control
adapter, not a production default or a new real experiment.

## Outcome

137/137 tests pass:16 new serialization/integration controls and121 unchanged
source, route, output, receipt and watchdog regressions. Test runner63.717s
(unittest reports63.415s); complete control operation64.883s before its summary
write. First local16-test development check also passed. The archived attempt
is `data/moe/results/bounded_evidence_controls_20260925_r1_attempt001`.

Saved-only audit PASS,0 issues (0.0535s separately charged). Fresh `python -S`
replay PASS,0 (0.0556s separately charged);36 mutated cost records rejected.
All573 source/protocol bindings rechecked, including all565 sealed bindings.
310 raw files /15,384,915 bytes retained outside Git, including synthetic
checkpoint, intentional corruption fixtures, timeout partials and error logs.
No downloaded dataset, real checkpoint or external repository is committed.

## Exact identity and acceptance

`bounded_evidence.stream.save` produces identical bytes, SHA256 and byte count
to the unchanged canonical writer on supported evidence types. Large values
are fragmented; arrays are not copied; all repeated matrices remain present.
Nothing about the property, guard, factor identity or exact checking changes.

Checked byte differentials include escaped/non-BMP/unpaired-surrogate text,
large rational strings, extreme finite floats, signed zero, nested/aliased
structures, CSR-shaped arrays and an actual SYNTHETIC construction bundle.
Unsupported nonfinite/cyclic/oversize/type inputs fail closed, with no hidden
fallback to whole-buffer encoding. Admission limits are in the protocol.

The full synthetic pipeline uses the original300s ceiling and all seven
phases. It checks all6 original duties:4 discharged by2 checked route exclusions
and2 by fresh exact positive output bounds. Existing independent construction,
source and complete-output checks run unchanged; the complete synthetic request
is rechecked after relocation in a fresh model/solver-free process. This is not
a real learned-model certificate or a deployed-float proof.

Removed obligations, changed coefficients/factors/pair identities, wrong run or
hash and missing completion receipts are rejected. Exclusive partial creation,
short/zero writes, disk/fsync/link failures, symlinks, publish races, deadline
before/during/after publication are controlled. A final file without a timely
bound receipt is never acceptance. The actual owned-worker blocked-write test
ends TIMEOUT; injected disk failure ends ERROR. Both preserve partial bytes,
keep full costs and fail to certify. Short8s fault controls do not alter real
request budgets. No check is replaced by a hash or a diagnostic timing field.

## Fixed synthetic memory/cost probes

Two sizes and two methods, once each, order old/new then new/old;30s and sampled
8GiB per probe, no retry. Sources combine CSR-shaped rational data with a large
escaped scalar. Each old/new pair has byte-identical output.

| Serialized bytes | Legacy traced extra peak | Streaming traced extra peak | Legacy serialization | Streaming serialization |
|---:|---:|---:|---:|---:|
|811,622|1,624,474 bytes (1.55MiB)|207,847 bytes (0.20MiB)|0.00911s|0.08128s|
|6,553,322|13,107,874 bytes (12.50MiB)|207,862 bytes (0.20MiB)|0.06394s|0.64461s|

These are `tracemalloc` peaks measured AFTER object generation, not total RSS,
not a real4099 allocation profile and not whole-verifier memory usage. Largest
emitted block65536 bytes; largest observed escape fragment9907 bytes. The
bounded stack/sorting workspace is additional;64KiB is NOT a total-process
memory promise. The admitted resident object itself remains in memory.

The fixed1MiB additional-memory control gate passes at both sizes. Runtime is
not a success gate. **Streaming is slower on both probes with memory tracing
enabled**; there is no end-to-end speedup claim. This stage did not run an
untraced throughput experiment or tune block sizes to obtain a better ratio.
Directory fsync is also charged in the new publication path.

| Probe | Whole time before final cost-ledger write | Charged worker/cleanup | Parent reception/hash overhead |
|---|---:|---:|---:|
|Small legacy|0.09346s|0.09278s|0.00068s|
|Small streaming|0.16467s|0.16417s|0.00050s|
|Large streaming|0.71945s|0.71574s|0.00371s|
|Large legacy|0.15990s|0.15637s|0.00354s|

Whole costs include imports, synthetic object generation, serialization/hash,
flush/fsync/publication, worker report, owned cleanup and parent hash reception.
The enclosing recorded return also includes the final ledger write; a ledger
overrun invalidates success. Later saved-only audits are separately timed.
In the full synthetic proof, serialization diagnostics and their publication
are inside the charged construction phase, with no renewed deadline.

## Decision: keep optional, do not launch another real request yet

This closes the intended control gap: bounded additional serialization memory,
canonical identity, interrupted/partial/atomic publication, unchanged checks
and complete costs. It does NOT close the complete proof gap.

At the earlier real4099 in-memory return (~283s), only~15s useful work time
remained. New slower serialization, subsequent whole-buffer identity()/load(),
source checking and all output LPs still need time and memory. The resident
construction bundle and all its repeated objects remain unchanged. Solving a
publication peak alone therefore gives no basis to promise a300s certificate.

Next bounded work should separate construction, downstream source-check and
serialization costs, using controls/saved logs rather than recovering sealed
4099 matrices or automatically running a new real request. If improving encoder
throughput, keep exact bytes, all checks, cutoff and accounting; freeze that
change separately from mathematical representation or solver changes. No new
query, extra time/memory, support tuning or acceptance relaxation is authorized
by these results. Historical source-gap claims and external comparison tables
remain unchanged.

Evidence: [protocol](bounded_evidence_protocol_20260925_r1.md),
[controls / independent audit / raw manifest](bounded_evidence_controls_20260925_r1.json),
[fresh replay and cost-negative controls](bounded_evidence_replay_20260925_r1.json).

Read-only reproduction, archived raw files present:

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S scripts/review_bounded_evidence_costs.py
```

Software rechecking is not independent human technical review or a proof of the
native solver/floating-point execution stack. No ISSTA acceptance, real-model
gain or superiority over external tools is inferred.
