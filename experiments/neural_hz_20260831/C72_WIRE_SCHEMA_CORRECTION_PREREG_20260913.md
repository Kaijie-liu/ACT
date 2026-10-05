# C72v2: correct the exact C31 archive protocol spelling

C72v1 is terminal exit1; all1807 tests passed. Complete C31/C30 loading fits
both1GiB gates (growth482844672B, traced472045528+1063264B), but the source-chain
guard rejects before recovery. Read-only inspection identifies a concrete
adapter error: C31's original writer emits `c31_new_checked_prepared_Closed_v1`
(c31_prepared_generator_worker_v1.py:127), whereas C72v1 required the nonexistent
`c31_checked_prepared_Closed_v1`. The source/proof/hash identities are unchanged.

New worker v2 accepts ONLY that exact original schema; no alternate schema,
wildcard, weakened proof, old-receipt substitution or fallback. All full source,
native/lineage/transfer hashes remain mandatory. Correct the OFFLINE C31 hash
ledger conservatively to pay three full traversals: restore's initial check,
issued receipt's validation and the final preservation fingerprint (previous
v1 budget prepaid only two). This increases work; no tariff or cap is reduced.

All1807 inherited tests rerun with unchanged full source/test inventory and
caps; no unrelated test framework added. Fresh exclusive
results/c72_inverse_phase_20260913_v2, new worker/supervisor, complete v1
source/failure references frozen. Phase60s/component240s/restore60s/tests60s,
CPU1/GPU0,AS16GiB,both1GiB,64M,whole256M/branch200M and paid native payload stay.
Same complete source/phase scope and no full LIVE or score promotion. All old
versions remain immutable. Formal1870/all13/E0 CIFAR25/Tiny36; goal ACTIVE.
