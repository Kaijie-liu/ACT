# C33: collision-safe logging, identical C32 native algorithm and gates

C32's one frozen worker failed BEFORE structural selection: dict(elapsed_s=...,
**event) duplicated a C5 event field. No fresh C31 generator, native splice,
1GiB/256M/full-LIVE gate or solver was executed. Its complete source/results
remain immutable under CHECKPOINT_C32_LIVE_SPLICE_FAILED_20260911_SHA256SUMS.

New source c33_live_splice_worker_v1.py is mechanically identical to C32 except
the logger import, use of collision-safe event_record, and exclusive directory.
A complete-source regression proves that exact diff. Preserve source event
elapsed_s; add worker_elapsed_s under a distinct reserved key. No timing gate
uses either event field: process monotonic timeout and measured_build remain
unchanged. Eight new regressions include the triggering duplicate-key case.

Reuse the independently completed3553byte transfer proof, SHA
bab5946186e350159087a9a5d512d8e01759591e0d4309d9d2998e2936b87e86,
and NEW C31 source proof cad0401a... as text only. Do not rerun offline numeric
proof oracles. All exact C32 runtime/discovery/writer/binding/lineage code and
original graph/binary/frame/box/reconstruction semantics remain unchanged.
The representation schema remains C32 because this correction changes only
event serialization; run/worker/checkpoint provenance records C33 source.

All C32_LIVE_SPLICE_PREREG_20260911.md restrictions remain in force: one fresh
original Tiny143 run, no saved HZ input, no old complete post assembly, one
native phase, no fallback/terminal/other iid/replay/default/archive change.
CPU1/GPU0, AS16GiB, worker240s, tests60s, restore60s, entries64M, whole256M,
branch200M, radix16M, conservative construction1GiB, SAME full LIVE two-leaf
reference629346312bytes/52428800entries. Native payload and complete source
hash work remain paid/reported SEPARATELY, not free generation credit.

Run once in results/c33_live_splice_20260911_v1; freeze inherited sources plus
exact diagnosed C32 ERROR/empty-island/proof hashes. Auto-save tests/exit/drift,
original prefix result, qualification and any successful reconstructable native
checkpoint; fresh-process complete restore guard after success. No retries or
cap relaxation in this version. Formal1870/2413 and E061/400 stay unchanged.
