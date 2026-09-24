# Bounded batching: local throughput gain; source profiling failed closed

Completed and sealed the exact38-call synthetic roster. **36/36 publication
probes completed,0/2 construction profiles completed (both ERROR).** No retry,
extra size, modified batch limit, native LP solve or real request. Zero new
real complete output certificates. Do not call the full study38 successes.

Implementation `c3b3dd131d6dbd5618873e1f906940076b196732`; clean launch
`839e6cd6d` after committed freeze. Config SHA256
`5288249917407546ad4f76d0e6148bdf3c6fc5d4985e79f5ccbbfd451e41970d`.
158 tests passed before freeze (21 new/on-new-writer +137 unchanged),75.011s.
Independent saved-only audit PASS,0 issues means faithful evidence/accounting,
NOT that the two diagnostic calls succeeded. All582 frozen bindings intact;
190 raw files /132,707,344 bytes preserved outside Git.114 cost corruptions
rejected. Audit0.163s separately timed; later audit is not online proof work.

## Mechanism, not a new verification theorem

Short scalar array runs use bounded standard-library JSON batches,<=128 items
and<=16KiB conservative escaped size. This removes repeated Python fragment
dispatch while preserving visits, finite/type/size checks, complete scalar
inventory, canonical bytes and hashes. Long values/nesting use bounded R1
traversal. Atomic publication, fsync,<=64KiB write blocks, inherited deadline,
all source/guard/output checks and full accounting remain.

This is an engineering improvement to the evidence path. It does not alter HZ
relations, McCormick precision, certificate strength, candidate selection or
solver behavior. No claim of independently novel batching/JSON theory is made.

## Fresh three-arm comparison (do not pool tracing strata)

Each cell is the median of3 pre-registered, rotated-order runs. Same synthetic
payload for all arms. All36 payloads have exact byte equality within size.

Untraced serialization alone:

| Serialized bytes | Whole-buffer legacy | Per-element bounded R1 | Batched bounded | Reduction vs R1 |
|---:|---:|---:|---:|---:|
|811,622|0.003455s|0.011476s|0.005520s|51.9%|
|6,553,322|0.019118s|0.080061s|0.031914s|60.1%|

Larger-case observed range: R1 0.079369–0.080081s; batching
0.031772–0.032035s. Legacy remains fastest for serialization. The new path
improves the bounded implementation, **not all competing implementations**.

With tracemalloc enabled, the corresponding medians are:

| Serialized bytes | Legacy | Bounded R1 | Batched bounded |
|---:|---:|---:|---:|
|811,622|0.009106s|0.082426s|0.051124s|
|6,553,322|0.064512s|0.648975s|0.399599s|

Traced additional peak at the larger size (maximum over3): legacy13,107,870
bytes (~12.50MiB), R1 207,856 bytes, batched209,835 bytes (~0.20MiB). The new
path retains the fixed<=1MiB workspace-control gate. This is NOT total process
memory; the constructed input object and later load()/identity() still exist.

Whole-call untraced medians, including imports, generation, publication and
owned cleanup: small legacy/R1/batch0.09096/0.09114/0.09045s; large
0.09111/0.16376/0.09106s. Startup/watchdog sampling dominate small calls.
Neither these figures nor the serialization reduction are full-MoE latency or
new-certificate gains. There is no random-seed/general-architecture inference
from three repetitions of deterministic synthetic data.

## Why both source profiles failed

Both stop in the route-certificate binding check, around0.093s whole-call time.
The R1 diagnostic script constructed `rows = [shared_certificate]`, although
`residual_proof.build.finish(..., mode='shared')` expects the certificate
dictionary itself. The unchanged checker correctly rejected this as an invalid
request/source/factor/run binding. This is OUR profiling-entry interface bug,
not model difficulty, insufficient relaxation, memory pressure or slow solving.

The pre-freeze158 controls covered the encoder/full proof adapter, but did not
exercise this separate diagnostic entry. They were therefore insufficient to
catch this bug. The two logs/terminals and their original frozen source are
retained. No construction/source-check cost decomposition was obtained. Do not
use0.093s as a construction measurement or manufacture a root-cause ranking.

## Separate repair controls, no rerun of the frozen profiles

`source_cost_controls/profile.py` is a separate V2 interface. It passes the
bound dictionary directly, checks its type/binding before construction, keeps
the source/guard/output checker unchanged and retains nonoverlapping timing
and old R1 publication. There is no real loader, native solver, automatic sample
chooser or retry CLI. R1 files remain unchanged.

4 controls passed on a DIFFERENT tiny analytic object (E2,C2,width1,depth0,
seed91): full diagnostic and fresh solver-free source recheck; list/empty/wrong
run certificate rejection; expired deadline; changed source rejection. These
are correctness tests, not substitutes for the original two measurements.
No new output bounds/real certificates. The V2 repair has not repeated either
frozen profile. See [repair controls](source_cost_interface_controls_20260925_r1.json).

## Decision and relation to the paper

Keep batching optional. Its local positive result is enough to stop further
batch-size/repeat/fixture tuning in this round. It does not repair the earlier
real4099 construction cost (~283s before publication), and does not prove
that the remaining~15s suffice for publication, source checks and all outputs.

The next scoped task is to validate V2's external supervision/partial records
and independently freeze the finite source-cost diagnosis, explicitly labelled
an interface-repair follow-up, NOT a replacement for this0/2 result. Only valid
cost data should motivate a new single-factor construction/source-check change.
Do not automatically rerun real4099, enlarge a budget, or infer LP looseness.

For an ISSTA-targeted paper, the core remains route-conditioned full-output
verification, scoped proof reuse and evidence whose guarantee is explicit.
This encoder gain is supporting engineering, not the main novelty claim.
Historical23 gains still have the recorded source-containment gaps and are not
upgraded to source-complete strict certificates. Same-object strict real output
closure, competitive external results, independent human review and clean
reproduction remain distinct unresolved obligations; no venue outcome is
guaranteed by this study.

Evidence: [protocol](batched_evidence_protocol_20260925_r1.md),
[freeze](batched_evidence_freeze_20260925_r1.md),
[158 controls](batched_evidence_controls_20260925_r1.json),
[audit, all calls, timings, raw hashes](batched_evidence_audit_20260925_r1.json),
[historical source scope](main_table_source_applicability_20260921.md).

Saved-only reproduction (no new proposals/queries):

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -S -m batched_evidence.audit
```
