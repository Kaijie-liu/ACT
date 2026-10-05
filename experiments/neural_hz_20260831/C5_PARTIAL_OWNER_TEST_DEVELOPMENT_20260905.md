# Development-test failure retained

Before the new partial-owner visitor's real accounting run, its initial test
suite reported **7 failed, 66 passed in 1.13s**. The failed cases were both
short-owner variants, both disjoint-owner variants, and overlap/dtype/dense
alias guards. The initial helper directly returned
`sp.csr_matrix((values, indices, indptr), shape=..., copy=False)`.
In this SciPy environment the resulting data did not retain the intended
owner, so those tests were not exercising the registered state.

The fixture now explicitly assigns `out.data = values` and asserts
`out.data is values`. Expected owner bytes, active entry counts and all reject
conditions are unchanged. No visitor implementation was changed in response
to these seven failures. The corrected suite passed **73/73 in 1.18s** before
the source-frozen supervisor repeated the suite and preserved its raw log at
`results/c5_full_native_owner_ledger_20260905_v2/tests.log`.

This note preserves the development attempt; it is not an immutable raw log of
that first terminal run and does not relabel it as a passed test.
