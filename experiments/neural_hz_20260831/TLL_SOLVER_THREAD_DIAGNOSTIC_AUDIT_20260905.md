# TLL thread hypothesis: complete negative diagnostic

2026-09-05, `redu-hz`, base `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
The previous status-only turn was no progress. This continuation completed two
whole-family arms, after revalidating the 317-file checkpoint. No production
source, old record, archived authority, HZ rule or solver tolerance was edited.

| Fixed policy | CERT | Validated ADV | UNKNOWN | Formal TLL lost | Batch wall |
|---|---:|---:|---:|---:|---:|
| existing automatic threads | 11 | 16 | 5 | 0 | 142.630 s |
| explicit one HiGHS thread | 11 | 16 | 5 | 0 | 147.636 s |

Both arms cover iid0..31 once, four workers, 45-second solver budget and 16 GiB
per worker. Each made 64 MILP calls. All **64 paired numeric fingerprints**
match: objectives, integrality, variable bounds, complete CSR arrays and row
bounds. All 128 calls preserve their input fingerprint before/after execution.
This proves current-arm coefficient identity, not identity with the historical
29/32 candidate, whose exact numeric MILPs were not captured.

Embedded SciPy HiGHS is 1.12.0 in both arms. Observed process thread peaks
including the one observation thread are 12 and 3. Solver CPU totals are
356.347 s and 349.046 s; summed solver wall times are 357.251 s and 369.772 s.
Explicit pre/post instrumentation costs total 0.0995 s and 0.1068 s; the
observation thread also runs during solves, so these are not all its costs.
Fixed auto-then-one ordering and instrumentation prohibit a definitive speed
claim. There is no observed benefit warranting a production thread change.

The same two old candidate rows fail in both arms:

- iid8: base feasibility succeeds, then property feasibility times out near
  45 seconds. More idle native threads do not explain away this result.
- iid26: both arms return identical solution hashes for both MILPs. The
  property call has a `4.440892098500626e-16` variable-bound violation, and the
  reconstructed input is rejected by the unchanged strict concrete check.
  The point is not clamped/repaired and earns no solve.

Reported invalid ADV is **0**; rejected solver proposals are **1 per arm**.
The raw verdict remains UNKNOWN for iid26. Independent native-ONNX/original-
VNNLIB, zero-tolerance replay accepts **16/16 ADV in each arm** (32 records,
not 32 distinct solved instances). No source/model/spec drift was detected.

The thread hypothesis is closed as a recovery route. This does not establish
the cause of the historical candidate regression. No thread option is enabled,
no uninstrumented speed qualification is justified, and no selected retry,
seed search or numerical rescue follows. TLL remains an unpromoted 27/32
capability checkpoint, +10 relative to formal 17/32, not a recovered 29/32.

Sources: `results/tll_solver_threads_20260905_v1/comparison.json`, both complete
arm directories and `evidence/tll_threads_adv_onnx_replay_20260905_v1.json`.
The latter SHA-256 is
`f91603a050ad130ca3c42a78eb0c7d3c54ff9212acad28d695bb290179f2c849`.
The 12 trace/supervisor tests pass. Formal 1870/2413 and E0 61/400 are unchanged;
the full objective and its complete 2413-case gate remain active.
