# C40 post-exit synthetic diagnosis: readonly protocol5 owner rejection

This is NOT part of C40's frozen run and does NOT reopen its failed version.
No actual C34/C40 HZ was reloaded, no archived data changed, no collector was
relaxed, and no source-bound restore was implemented by this diagnostic.

The actual failure was WholeStateReject/unknown_numpy_external_buffer at
runtime_numeric_roots/phase_events during the first closed numeric ledger.
C23 creates owned uint64 events and marks them readonly. In this environment,
a synthetic owned readonly uint64 array of200 entries survives protocol5 with
identical shape/dtype/bytes, but its decoded owner chain ends in bytes. The
unchanged closed owner ledger rejects that representation with the SAME error.
Its writable control instead uses the existing accepted buffer adapter.

Synthetic payload SHA256 on ALL variants:
de663cfb3b82787ec7c32624aba8cd8de6555fc906b507971a2c2f3207b5fe52.

Original owned readonly array: accepted1600 numeric bytes/200 entries.
Decoded readonly array: rejected unknown_numpy_external_buffer.
Explicit owned copy, readonly flag restored: accepted1600 bytes/200 entries,
identical hash. Original decoded array remains non-owning and readonly.
Writable original and decoded control: both accepted1600 bytes/200 entries.

Separate test file test_c40_pickle_owner_postmortem_v1.py:
3 passed in1.12s, exit0. This is additional POST-EXIT evidence, not a retroactive
change from C40's1714 frozen tests or an actual target replay.

The synthetic copy is not a production fix. A future actual restoration must
independently bind every source image, preserve ALL reachable alias identities
and readonly semantics, pay for copying and authentication, expose every owner
to the same ledger, and prove old decoded buffers retire. Do not merely copy
the measurement view, hide phase_events, accept arbitrary external owners,
mutate archived pickle bytes, or issue an old receipt for changed predicates.

The actual bytes-vs-memoryview terminal owner class was NOT inspected through
a second target load. The exact error is real; protocol5 readonly ownership is
a reproduced explanation, not a claim that every archived owner is diagnosed.
