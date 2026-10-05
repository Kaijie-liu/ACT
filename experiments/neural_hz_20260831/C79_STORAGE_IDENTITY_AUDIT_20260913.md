# C79: exact complete storage/source/native gates pass; process crashes in inverse

All85files/1895 tests pass, including all1875 old checks and20 ordinary storage
identity/view/stride/offset, separate equal-storage, array owner and complete
source/native/inverse fixtures. Full suite27.568446361s; pytest21.36s. One
TypedStorage deprecation warning is retained, not a test failure or suppression.
The pre-freeze20-test run passed0.98s with the same warning.

Actual complete C77 archive decoded in1.384763056s. The scope-local decoder
observes663 storage calls,522 original explicit identity keys and141 reuses.
Full headers, extent and every original byte agree for each reused key; all
Tensor objects/views and distinct equal-storage identities remain separate.
All394 array reducers and the sole200-entry/1600B readonly copy remain.

The complete full-root ledger is now466368992B/42949161entries/916 storages,
exactly the original LIVE union plus that already-authorized readonly copy.
This crosses the original entry-envelope gate that rejected C78. Full traversal
finishes26.023s and physical ledger26.367s, all662 numeric roots retained.
Full C74 source binding then matches9e8c3ddcbe09ebf2bdc97a87379beecebb659e3549ede4d10f802b53eba94349;
native binding matches HZba3a84a1d508db6330df50e81c688747755134a55a1f097a698a789b927723d1,
the complete journal/phase/event proofs and all original array/cache identities.
Before inverse at26.874s, HWM1566195712B against entry578572288B; traced peak
712530397+metadata77233952B. These are intermediate observations, not a final
whole-restore resource pass.

At the30s periodic stack dump during Fraction inverse work, the child exits
with signal11. The stack file stops in fractions._from_coprime_ints; there is
no completed inverse or result.json. Supervisor31316 exits1 after58.592694691s,
restore_exit=-11; source/provenance drift false. Exit SHA
3fdc5d6495849b1d3e56686da9ecfcacc95a3b8db889fddb6b17bc24c88f11ff.
C79v1 therefore remains unqualified as a whole; no witness or solver admission.

A subsequent standalone stdlib-only probe reproduces signal11 during periodic
Fraction stack sampling without importing ACT, C78, Torch or NumPy. Fatal-only
control passes2s. This establishes an independent diagnostic/runtime failure
and motivates C80 removing only the optional periodic sampling thread, with
fatal reporting and every source/proof/resource/test gate retained. It is not
a general proof that native code can never have defects. Preserve this failed
run and its partially positive boundary evidence unchanged.

Formal1870/all13 and E0 CIFAR25/Tiny36 unchanged. No production/default/history,
original network, terminal solver, commit or push modification occurred.
