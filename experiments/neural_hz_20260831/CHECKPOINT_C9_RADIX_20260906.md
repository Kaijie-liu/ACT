# C9 checkpoint — 2026-09-06

Branch redu-hz; goal active. Formal1870/2413 and independent E061/400 unchanged.
This checkpoint seals the C9 sources and all three completed result directories.

- Predicate-only prerequisite: passed; exact inherited predicates and native
  coefficient retention, not a complete-suffix result.
- Complete integrated V1: failed at its final reference HWM gate. Successful
  construction, all-row audit and native ingestion evidence are retained, but
  the transaction remains CLOSED/FAILED.
- Fresh read-only qualification of V1's saved complete-suffix checkpoint:
  CHECKPOINT_QUALIFIED. 355 tests, exact two-level reconstruction, native
  11160330/11160330 nonzeros unchanged; combined original/candidate numeric
  storage412257500 bytes and39363874 entries strictly below the same
  reference lower bound629346312 bytes and52428800 entries.

See C9_COMPLETE_SUFFIX_AUDIT_20260906.md for provenance, numerical/resource
limits and distinctions between the failed construction transaction and
successful read-only checkpoint qualification. All C9 sources already used in
these runs are frozen. No experiments from this batch remain running.

Next: actual live ReLU78 integration and post-consumer exactness, phase-frame
and numeric-storage gates; ordinary terminal verification only after those
pass. Then the registered CIFAR targets, same-structure shadows, family and
full2413 replay. A checkpoint alone cannot promote a candidate or a score.
