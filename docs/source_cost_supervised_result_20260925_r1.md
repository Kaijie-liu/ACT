# Repaired profiling now has valid data: 2/2 completed, no output certificates

Implemented hard-budget supervision, passed **180 controls**, committed and
pushed the independent freeze, executed exactly the two frozen synthetic
repair-follow-up calls, and audited/replayed their saved evidence. No retry,
extra size, real request, native LP query, encoder change or relaxation change.
The original batched study remains **0/2 source-profile completions**; its
interface errors were not replaced or reclassified.

Implementation `d88e7c9f40906a622c53731b1821d8a85b18282c`; clean execution
HEAD `10451b678` (committed freeze). Config SHA256
`e9f8e267c21c9a36d596ce865a40ef0e3001003ce545b9491131d622c794c3c5`.
593 source/protocol bindings intact, including all582 historical bindings and
the four already-repaired interface files. Each call:300s,2s reserve,sampled
8GiB,2 CPU threads,no GPU,unchanged V2 profiler and OLD R1 streaming.

## What was actually exercised

| Fixed source | Original output duties | Checked exclusions | Retained constructions checked | Expert traces |
|---|---:|---:|---:|---:|
| E4/C3/w4/d1/seed724 |12|0|12 (6 pairs)|4|
| E8/C10/w8/d2/seed724 |252|225|27 (3 pairs)|4|

These are checked **LP constructions**, not positive lower bounds. No output
solver ran and no output certificates were produced. Retained pairs are not
claimed concretely reachable. All tie-legal original duties remain covered
by the registered excluded/retained accounting; nothing was dropped on time.

## Whole cost and components, seconds

| Measurement | Small | Medium |
|---|---:|---:|
| Entire supervised call, including terminal write/hash |0.226111|0.438149|
| Profile worker, including import/generation/publication/cleanup |0.167990|0.367276|
| Separate, budgeted candidate reception/cleanup |0.053593|0.066396|
| Router prefix construction |0.001346|0.006546|
| Retained expert/output construction |0.026537|0.078350|
| Source+construction serialization/publication |0.028654|0.064969|
| Full source/construction checking |0.040708|0.126003|

Lower rows are contained within the profile worker; **do not sum them with
the whole call or worker row**. Identity, proposal, report publication and
remaining overhead are retained in the JSON ledger. The enclosing batch
caller also records0.227827/0.438160s respectively; no unreported phase budget.
All hashes/large reads in reception were watchdog-supervised, not an unbounded
parent operation. Later saved-only audit0.189418s and fresh replay0.191286s are
separate from online cost.34raw files,2,081,612bytes,retained outside Git.

The source-check stage is larger than construction in BOTH measured calls,
and larger than publication. In the medium case its breakdown is:

| Checked work | Calls | Seconds |
|---|---:|---:|
| Router/route recheck |1|0.012238|
| Expert traces |4|0.027772|
| Shared-factor joins |3|0.020087|
| Guard attachment |3|0.019852|
| Property projection |3|0.023121|
| Weighted LP construction |3|0.022818|

No single one of these is overwhelmingly dominant. This is one execution per
small deterministic object, with charged instrumentation and process startup;
not a timing distribution, production speedup, or explanation of the sealed
real4099 run's~283s construction. No efficacy/strict-certificate claim changes.

## Saved-record attribution: a concrete reuse opportunity, not a proved bottleneck

An additional read-only inventory used only the hash-bound saved construction
and the frozen checker call sites; it did not call a producer/checker/solver or
repeat the profiles. It covers `check_join -> check_guards -> check_projection
-> check_outputs`, excluding router and per-network internals.

| Call-site inventory | Small | Medium |
|---|---:|---:|
| Expected exact `unpack` sites |60|30|
| Distinct full source identities |24|15|
| Repeated identity sites |36|15|

The same joint source is unpacked at join+guard, guarded source at guard+
projection, and projected source at projection+output checks. Base/router and
expert sources also repeat across pairs. This identifies *what could be
reused* without removing any mathematical check. It does **not** show that
parsing occupies60%/50% of time: those are site-count fractions, not timing
shares or achievable speedups. Identity hashing, immutable representation,
copying, cache memory and lookup costs are still unmeasured.

## Continue/stop decision

Stop encoder/batch-size and relaxation tuning. Do not expand these profiles
or reopen a real request. There is now enough evidence for a **separate small
control proposal** targeting exact parsed-source reuse in the checker, not a
default cache change or a new performance claim:

1. Start with the three adjacent shared-source transitions above; immutable,
   invocation-local parsed representations, complete source/factor/domain
   identity binding, bounded memory. No pointer-only or partial-matrix keys.
2. Retain every guard, mapping, projection, output-inventory and McCormick
   check. Reuse parsing only, never skip acceptance predicates or import a
   previous request's positive result. This is distinct from older LP-parser
   caching: the target is the upstream source-state checking chain.
3. Before any timing comparison, test mutated source/CSR/factor identities,
   wrong guard/pair/request, mutable alias contamination, cutoff and partial
   records, and differential acceptance/rejection against the unchanged path.
4. Only after those controls, separately freeze complete-cost measurement of
   parsing, identity checks, freeze/copy, remaining exact checks and peak memory.
   Stop if amortization does not exceed its overhead; no mandatory speedup.

This is a bounded next research direction supported by repeated checked-source
identities. **Parser dominance is not established** and construction has not
been eliminated as a real-scale issue. No cache/algorithm default was changed
in this stage. No need for another broad research-direction document before
these focused controls, but any real request experiment needs its own freeze.

## Audit and evidence

Saved-only audit and a fresh `python -S` replay:PASS,0issues,2/2 valid profiles;
both re-ran the unchanged source checker against saved constructions. No model,
data, producer or solver imports/process/network access allowed by the archive
script. Eight corruptions of actual cost records rejected. These audits are
not independent machine proofs of the Python implementation or deployed float
semantics. Explicit source assumptions and zero-output-bound scope remain.

- [Protocol](source_cost_supervised_protocol_20260925_r1.md), [freeze](source_cost_supervised_freeze_20260925_r1.md)
- [180 controls](source_cost_supervised_controls_20260925_r1.json)
- [Full saved audit, component costs and raw file hashes](source_cost_supervised_audit_20260925_r1.json)
- [Fresh replay](source_cost_supervised_replay_20260925_r1.json)
- [Read-only identity inventory](source_cost_reuse_inventory_20260925_r1.json)
- Reproducible entry: `python -S scripts/audit_source_cost_supervised.py`
- Read-only inventory: `python -S scripts/source_cost_reuse_inventory.py`

Historical23 source-containment gaps, sealed98/4088/4096/4098/4099 and the
lack of new source-complete real output certificates remain unchanged.
