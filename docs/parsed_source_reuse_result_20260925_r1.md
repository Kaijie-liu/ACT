# Exact parsed-source reuse: correct but slower; keep disabled

**Stop this version's performance rollout.** All12 frozen saved-evidence
checks completed and reproduced the original checker result, but all6 paired
checker times were worse with reuse. Do not promote the cache to production,
tune its limits, expand samples, or remove identity/alias checks to claim a win.

This closes a concrete hypothesis from the previous diagnosis: repeated exact
source parsing exists and can be safely avoided, but safe materialization cost
in this implementation consumes the savings. It is not a new output proof or
an improvement in certification power. No original real request was reopened.

## Implementation, freeze and evidence

Implementation `f98e3f5d0a3d9ed81334374585e28bc6f441c5bb`; clean freeze launch
`4baf88eb9`. Config SHA256
`f499fb93eef7fb01a35ec43d40c455f9b45f6763846c8da8bbdd0f1e84536d13`.
207controlsPASS (27new+180unchanged),86.111s;601bindings intact. All original
source/guard/output checker predicates retained through private globals and
identical function bytecode, with no global module mutation. Only unpacking
is reused; no route decision, bound, property acceptance or prior proof cached.

Complete current source bytes + current request/source identity + fresh
invocation bind each lookup. Frozen storage never shares mutable containers
or Fraction objects with callers. Controls explicitly mutate a returned
Fraction's private slots and verify that later parsing remains correct.
Hash collision, altered source/CSR/factor, cross-scope transplant, wrong
property and missing duty with a warm cache all fail closed.

The optional worker is integrated into the original300s hard supervisor:
profile and receiver share the298s work deadline,2s terminal reserve,sampled
8GiB,2CPU threads. Read/hash/check/copy/freeze/reception/publication and owned
cleanup charged. Actual cutoff, partial file, exception, late receipt/terminal,
resource and cost controls passed. No production default or math change.

## Frozen finite comparison

Two prior saved synthetic constructions,each off/on x3 rotated pairs. Inputs
were fixed before timings, not selected for a favorable cache result. No source
regeneration, model loading, dataset, native solver or output bound generation.
The off arm uses the original exact parser directly with the same lightweight
instrumentation; it does not pay fictitious snapshot or copy costs.

All12 results equal the original independent source checker. Small source:
12original duties,12retained constructions; medium:252original duties,
225checked exclusions and27retained constructions. These are source/LP
construction checks, NOT positive output lower bounds or complete SAFE.

### Timing: medians of three, seconds

| Saved object / metric | No reuse | Reuse | Interpretation |
|---|---:|---:|---|
| Small full checker |0.042491|0.045886|about8.0% slower|
| Medium full checker |0.124504|0.134216|about7.8% slower|
| Small whole supervised segment |0.146132|0.147105|no observed overall advantage|
| Medium whole supervised segment |0.253130|0.257398|no observed overall advantage|

Checker ranges:small off0.042474–0.043789,on0.045786–0.049604;
medium off0.124313–0.129011,on0.133771–0.136889. All6 paired checker differences
are positive (reuse slower),from+0.002097 to+0.012385s. Median ratios above
are NOT medians of paired ratios. No inferential generalization from3 repeats.

Whole segments include interpreter startup,source reading,receiver and terminal
cost. Small reuse range0.144062–0.182627 includes the slower third repetition;
it is not discarded. These subsecond synthetic checks cannot establish
full-MoE speed or explain the sealed real4099 construction bottleneck.

### The mechanism worked; the net-cost hypothesis did not

| Counts per call | Small off / on | Medium off / on |
|---|---:|---:|
| Lookups |60 / 60|30 / 30|
| Original exact parses |60 / 24|30 / 15|
| Hits |0 / 36|0 / 15|
| Peak retained payload bytes |0 / 265,393|0 / 884,176|
| Peak retained cells |0 / 14,640|0 / 34,168|

Counts match the previous saved identity inventory exactly. Retained bytes
are serialized payload only, NOT total Python memory; object storage and
temporary snapshots also cost memory. RSS samples remain in per-call ledgers.

Median parser cost components, seconds:

| Component | Small off | Small on | Medium off | Medium on |
|---|---:|---:|---:|---:|
| Original parsing |0.025813|0.012392|0.065309|0.033598|
| Current-source snapshot/serialization |0|0.005029|0|0.009905|
| Identity lookup |0|0.000221|0|0.000457|
| Immutable freeze |0|0.003277|0|0.009901|
| Defensive fresh copy |0|0.007702|0|0.019799|
| Total parser interface |0.025958|0.028871|0.065410|0.073822|

Reference parsing alone almost halves, but snapshots+freeze+copy cost more
than it saves. Hash lookup itself is not the main added cost here. Rows are
separate component medians and need not sum to the median total; each actual
call's full cost closes in the ledger, including retention/other overhead.

## Decision and bounded next direction

Keep this version opt-in and disabled by default; seal the12-call comparison.
No more repetitions,cache-size tuning,larger inputs,real retries or weaker
checks are justified by this negative result. Fewer parses is not a performance
contribution by itself, and this is not sufficient novelty for an ISSTA claim.

There is now a more precise interface problem to consider separately: the
existing checkers consume mutable parsed structures,so a safe general-purpose
cache has to reconstruct protected containers/rationals on every access.
If further work is authorized,the bounded next design is a **read-only checked
source view**,not another content-cache parameter search. It must preserve
complete source/factor/guard identities and every original predicate,prevent
container AND rational alias mutation,and keep lifetime/cutoff/cost guarantees.
That is an API/representation change requiring its OWN controls and freeze;
it is not implemented or performance-validated here. Do not merely return the
current cache's internal objects to make this table faster.

The core research priority stays full routed-MoE output evidence and fair
verification gains. Do not turn this supporting engineering into a succession
of microbenchmarks claimed as the main innovation. High-accuracy real-scale
strict closure, historical23 source-containment gaps, external competitive
coverage and independent human/clean-environment review remain separate.

## Independent audit

Saved-only original-checker recheck and archive audit PASS,0issues,12/12;
36 corruptions of actual cost records rejected. The archive process forbids
model/solver/producer/new-parser imports and process/network access.158raw
files/349,804bytes retained,plus hash-bound parent source files (not copied
per repetition). Source-check conclusions and all denominators unchanged.
Audit0.188390s separately charged; fresh replay recorded separately too.

Evidence: [protocol](parsed_source_reuse_protocol_20260925_r1.md),
[207 controls](parsed_source_reuse_controls_20260925_r1.json),
[freeze](parsed_source_reuse_freeze_20260925_r1.md),
[full results and raw hashes](parsed_source_reuse_audit_20260925_r1.json),
[fresh replay](parsed_source_reuse_replay_20260925_r1.json).

Recheck without rerunning the experiment:
`python -S scripts/audit_parsed_source_reuse.py`.
