# C5 V2 first complete terminal attempt: closed at the unchanged wall cap

This additive entry closes `results/c5_first_terminal_20260905_v1/`; it does
not rewrite the successful prefix audits or relax their gates. 129 tests
passed. Source and provenance drift are false. The worker hit its registered
240-second wall cap; supervisor wall was 242.8899236479774 seconds. There is
no terminal result, final HZ checkpoint, solver witness or completed ReLU78.
Formal gain is zero, not an inferred UNKNOWN solve. No retry of this version.

Dense77 completed at 13.3364 seconds; ReLU78 then entered native lazy
materialization. The last start event at 13.791534 seconds is a left matrix
of shape (200, 6272), 1,254,400 stored entries, composing a (6272, 6272)
implicit convolution. There is no corresponding completion event. This
mixed-chain suffix is outside the frozen two-convolution C5 compiler.

The read-only postmortem is
`evidence/c5_first_terminal_postmortem_20260905_v1.json`, SHA-256
`5d3646babd994e22fa84e0102e8a8729eefe89f0eef9ddf60aba9c15c54f360f`.
It verifies the source freeze and actual ADD75 snapshot (SHA-256
`d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed`).
All 14 terms and 9 distinct source identities are inventoried, not merely
the first failing term. Terms 0, 1, 2, 4 and 10 have identically zero value
maps. The first still carries nc=11708, nb=1150, neq=1150, nineq=2300 and
frame 1. Its predicates MUST survive any skipped value computation.
The pending convolution has 5,914,624 structural expanded entries; its full
200-row left pattern implies a 1,182,924,800 structural multiplication bound.
That is a bound, not a measured count of executed operations.

The next same-structure hypothesis is source-bound support propagation
through the exact affine program: retain every source, predicate and shared
latent identity, but avoid composing coefficients at certified zero value
coordinates. Zero-source paths are one case of this rule; nonzero paths
also encounter the exact P44 diagonal with 124 supported coordinates.
Removing zero-source work alone is not yet evidence that the entire suffix
will fit the existing work/storage/wall caps. No such implementation or
complete-state reduction is claimed by this postmortem.

All raw events, tests, logs, complete snapshots and exit hashes are retained
in the exclusive result directory. Process inspection after exit found no
live terminal worker. Formal 1870/2413, independent E0 61/400, the 13-family
retention requirement, concurrency gate and default-off status are unchanged.
