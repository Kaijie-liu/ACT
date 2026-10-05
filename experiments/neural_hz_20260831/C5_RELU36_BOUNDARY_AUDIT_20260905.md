# ReLU36 complete live-boundary audit

The fresh V1 run is closed, not erased: all two native requests/four source
branches compared byte-for-byte, but constructing all reference operators
would exceed the original aggregate 64M diagnostic preflight. Evidence SHA
`c00a8daed2725fdacc935d522304bc0e8182f70fac7ad4b419c0afb1e9fd10c8`.

The separately preregistered V2 uses exactly the same candidate/runtime,
native requests and scalar oracle. It changes only the proof method for the
SAME complete owner-aware physical inequality; see
`C5_RELU36_OWNER_LOWER_BOUND_PREREG_20260905.md` for the monotonicity argument.
No full reference, higher budget, smaller candidate root set, GC advantage,
different phase selection or different comparator is used.

V2 evidence `evidence/c5_relu36_live_20260905_v2.json` SHA
`b45d017ac76a55fdc535d411f4c1ee9aa93496b0d603d18da1b286dc02d7e2c3`.
Tests: 59 passed. Worker/tests exit zero, source/provenance drift false.
Supervisor wall 88.125184 seconds; qualification wall 76.914402 seconds.

## Numerical and construction evidence

All 434 registered numeric roots include 34 Facts, 64 constraints, 36 Bounds,
16 affine expressions, six source HZ objects, the 81-layer graph, model
registered state and input/output specifications. All incoming fingerprints
remain unchanged. Arbitrary library/Python heap accounting remains outside
the frozen numeric metric; shallow metadata is explicitly separate.

The probe requests 38 rows, rejects admission (21,901 positive Gc/Gb nnz vs
25,095 threshold), and the native full request also contains 38 rows. Both
requests retain all four sources including the ReLU28 zero value map with
nonzero latent/predicate dimensions. All eight operator comparisons and all
HZ numeric/frame/exactness fields match the original ordered row oracle.
Cumulative channel products: 9,939,968; every stage/branch quarter-work gate
passes. Traced candidate peaks are at most 22,094,325 bytes plus 253,792 tracer
metadata; conservative resident-growth bounds are at most 30,846,976 bytes.
Both construction stages pass the unchanged 1 GiB bound.

## Complete candidate versus a certified reference lower bound

All five reachable implicit operators would expand to 80,286,208 nnz. The
fixed largest-operator witness is Conv10 (identity selection, layer id only
recorded as provenance). Its REAL independently built CSR has 26,214,400 nnz,
314,673,156 owner bytes, and every row is verified against the implicit map.
Its construction peak is 530,722,210 traced bytes plus 9,408 tracer metadata,
and conservative resident growth 523,501,568 bytes: below 1 GiB. The witness
is necessarily included in the unchanged fully expanded reference. Other
reference roots can only increase the union-of-owners metric.

| Boundary | Complete candidate bytes | Reference LOWER BOUND bytes | Candidate entries | Reference LOWER BOUND entries |
| --- | ---: | ---: | ---: | ---: |
| Entry | 114,971,168 | 314,673,156 | 12,677,772 | 26,214,400 |
| Probe retained | 116,834,352 | 320,439,464 | 12,816,509 | 26,677,342 |
| Follow-up retained | 116,834,352 | 320,439,464 | 12,816,509 | 26,677,342 |
| Both retained | 118,697,536 | 322,302,648 | 12,955,246 | 26,816,079 |

Thus complete candidate < measured reference subset <= complete reference
for both metrics at every boundary. The right column is NOT a measured total
for the reference. The physical gate follows from a conservative inequality,
not a logical-work estimate or an omitted candidate component.

## Scope and next step

V2 deliberately stops before ReLU36; prior fresh integrated V2 provides the
separate actual native ReLU36 publication snapshot. These are prerequisite
proofs for one unchanged candidate, not joined fragments of a verdict. No
terminal witness or CERT/ADV is claimed. The ReLU36 boundary prerequisite is
now discharged, permitting a separately preregistered fresh ReLU63 prefix.
The capability concurrency gate, same-structure/family and full 2413 replays
remain outstanding. Formal 1870/2413, separate E0 61/400 and defaults unchanged.
