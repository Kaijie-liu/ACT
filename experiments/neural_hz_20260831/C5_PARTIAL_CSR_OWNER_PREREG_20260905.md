# Partial CSR owner accounting, unchanged metric and immutable compiler

The full native probe + 1097-row follow-up passes all operator/HZ comparisons.
Cumulative channel products are 80639104, within frozen 200M/branch and 256M
sequence caps. The new schema/bytearray adapter measures the entry boundary at
64805800 candidate vs 410290612 expanded bytes, but correctly rejects the
completed probe's Gc.data as a short view retaining a larger NumPy allocation.
The completed transaction V1 record and its failure remain immutable.

This is an actual SciPy allocation pattern, not an extreme numeric case.
Introduce a separate V3 ledger with no source/result copies or compaction.
Keep the frozen metric: FULL retained owner bytes; CSR entries are data.size
only; index/indptr arrays contribute bytes but zero entries. For several CSR
data aliases of one owner, count the union of their active byte intervals
divided by element size. Exact duplicate spans count once, disjoint spans add;
partial overlaps, dtype conflicts and mixed dense/index/data semantics remain
rejected. Owner identity and physical byte count remain unchanged. The old
ledger and V2 remain unchanged and must still reject their unsupported case.
The new adapter does not reinterpret hidden capacity as a physical saving.

Test native/bytearray short CSR data, identical and disjoint aliases, full
owner bytes, small active data counts, overlap/gapped/dtype/semantic rejection
and native full-owner compatibility. Re-run all previous ledger/ordered tests.

One NEW accounting-only repetition of the same complete native sequence is
allowed after these tests. Bind the same snapshot/compiler/provenance and
the new adapter. Keep the original probe admission, ALL rows/branches,
cumulative budgets and both complete scalar-oracle comparisons unchanged.
This is changed measurement instrumentation, never a solver/verdict retry.
240 seconds, 16 GiB, CPU/BLAS one thread, exclusive source/log/result/exit.
Output evidence/c5_full_native_owner_ledger_20260905_v2.json.

Apply the new visitor only to the registered loaded-snapshot boundaries.
No construction/allocator, original live-worker ownership, atomic publication,
ReLU advancement, four-concurrent, E0 or formal-score gate is inferred.
All old failures remain saved; formal gain zero and goal active.
