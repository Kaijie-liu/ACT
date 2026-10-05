# C56 v2: inverse-live exact scalar pool after unchanged gauged carrier lowering

v1 is CLOSED/NOT ACCEPTED: all25 tests and exact/restore proofs passed; all three
fixed cohorts reduced predicate nnz, numeric bytes and numeric entries. Conv's
fresh numeric+reported shallow accounting was128148->128167B, a19B failure.
The fact that restored Conv passes does not substitute for its failed fresh gate.
Keep the recorded workload, strings/metrics, accounting and threshold unchanged.

The new rule is ordinary scalar liveness, uniformly applied after the SAME C56v1
mathematical lowering. Native consumer coefficients and RHS now live explicitly
as exact binary64 values. The retained exact pool is needed only for the original
coordinate inverse; entries referenced solely by superseded exact CSR/RHS are
dead. Build a canonical pool from ALL inverse references, remap ONLY inverse
scalar IDs, preserve every inverse root and exact coefficient value, then retire
the old pool and inverse arrays. No source scalar value required for inverse,
original predicate, concrete witness or phase semantics may be lost.

Keep the same state field/schema contract and complete report fields. Do not
shorten strings, omit diagnostics or change Python accounting to cross19B.
The independent audit compares inverse roots and canonical scalar values
coordinate-by-coordinate, rejects unused/missing pool entries, and still checks
every original source predicate/output against the ORIGINAL C54 scalar table.
An inverse-local scalar ID is never interpreted as an original source-table ID.

All original C56v1 tests remain as v2 tests, plus live-pool coverage guards.
All three fixed width128 cohorts, C52 SAME-source reference, original-source
retention, native matrices and four strict physical gates are unchanged. All
scalar liveness scans, repacking, remapping, proof, ownership and restore cost
remain inside the measurement and lift16M/whole256M/nested200M ceilings.
Retire TEN original unshared C54 arrays: five CSR/RHS/old-EQ-UID, four scalar-pool
buffers and the inverse array. The original global/removed/INEQ UID roots remain
shared and fully charged. No retained record silently keeps the old coefficient
pool alive. Preserve 60s focused/240s worker, CPU1/GPU0,AS16GiB,64M entries,
both1GiB transient and16384-auxiliary/131072-added-entry limits.

Use one exclusive frozen run in results/c56_inverse_live_20260912_v2, with normal
tests, complete source/row/inverse proofs and48 original feasible vectors,
protocol5 export/C41 authenticated decode and fresh source-based replay. Save
all outcomes and exits automatically. It is mathematical binary64 realization
only: no native solver admission, real source/target run, production/history
edit, forbidden rescue, full-suite qualification or score promotion. Formal
1870/2413 and separate E0 CIFAR25/Tiny36 are unchanged.
