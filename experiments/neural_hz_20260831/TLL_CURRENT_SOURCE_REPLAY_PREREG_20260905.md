# Current-source TLL qualification replay

Date: 2026-09-05; branch `redu-hz`; base
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

C3 has closed by its frozen negative shape condition. Resume the existing
signed/dead-ReLU candidate in `TLL_FAMILY_GATE_V2.md`, unchanged: exact signed
classes; delete proven dead pairs; delete proven dead cardinality-four classes
only at source width >= 8192. All selectors remain structural and default-off.
The BN loader option stays off for this replay.

Run all 32 TLL rows once under the current source, four workers, one numerical
thread each, sparse representation, 45-second solver budget and 16 GiB address
space per worker. A 240-second supervisor wall limit is explicit and yields
TIMEOUT with no credit. This is a qualification replay, not a controlled speed
comparison. The same fixed options apply to every row, including the three
previously UNKNOWN rows; no conditional rescue or selected rerun is permitted.

Before launching, freeze every model and actual converted specification hash,
the benchmark CSV hash, current source hashes, worker configuration, all 17
formal solved identities and the 29 prior candidate solved identities. Each
job creates its result/log and supervisor record exclusively. Preserve errors,
timeouts, absent/truncated records and invalid witnesses with zero score credit.
Only an ADV with a concrete-network validation may count as solved.

Advancement requires all 17 formal solved rows, all 29 prior candidate solved
rows, zero invalid ADV and no provenance drift. Failure stops this candidate's
promotion for diagnosis; it does not loosen the rule. Success allows the
remaining non-TLL retained shadows and a separately controlled four-concurrent
performance comparison, then the full 2,413-row replay. Formal score remains
1870/2413 throughout this family qualification, and external E0 remains 61/400.
