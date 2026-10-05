# C9 complete suffix: immutable three-transaction audit

Recorded 2026-09-06, branch redu-hz. This is an additive trial entry; previously
hashed sources, trial logs and failures are unchanged. Formal score remains
1870/2413; independent E0 remains 61/400. No candidate promotion is authorized.

## Predicate prerequisite: passed

results/c9_predicate_rows_20260905_v1/ preserves all original inherited
predicates. 328 tests passed. One wide logical row uses five radix auxiliaries;
the native backend retains all 697160 nonzero matrix coefficients, whereas
the original experimental-prefix input lost 95731 of 697150. This comparison
concerns the experimental C5 prefix, NOT a new audit of the formal baseline.
See C9_PREDICATE_ROWS_AUDIT_20260905.md for exact budgets and proof scope.

## Complete construction transaction V1: failed and closed

results/c9_integrated_suffix_20260905_v1/ has 355 passing tests but worker exit
1, no result.json, no source/provenance drift. Wall time 133.48345008771867 s.
Construction itself completed in 86.24280835408717 s, with conservative resident
growth upper bound 694480896 bytes, inside the unchanged 1 GiB cap. Complete
work upper bound 234443780 and largest branch 139815612 remain within their
registered bounds. All 243162 main definitions, 28 radix definitions, 1150 old
equalities and 2300 old inequalities were independently audited. Five wide
logical rows use 28 radix auxiliaries and three relays. Native ingestion
retained 11160330/11160330 coefficients with zero differences.

The saved checkpoint is 255558854 bytes, SHA-256
616a99a92e6e8ad3b261649d5b3613e4248fcb176dfa384a77fe51731a459962.
The later reference construction FAILED its frozen conservative HWM gate:
1099083776 > 1073741824 bytes. Earlier native ingestion had raised this
process's lifetime HWM. A smaller tracemalloc measurement does not override
that failure. The complete transaction remains failed, not retroactively passed.

## Fresh read-only qualification of that same checkpoint: passed

Preregistered separately in
C9_INTEGRATED_V1_CLOSED_CHECKPOINT_PREREG_20260905.md. No candidate HZ was rebuilt.
results/c9_checkpoint_qualification_20260905_v1/ has 355 passing tests, both exit
codes zero, no source/provenance drift, wall 60.42483157385141 s. result.json SHA:
f49458396d1862e4e0eccbb60bfa6393025b9fcbe8fe3c29a44ab483ab130c9c.

Both the full original native snapshot and separately serialized candidate
remain loaded, with duplicate numeric owners charged rather than merged:
475 numeric roots, 412257500 bytes / 39363874 entries. BOTH measures are
strictly below the SAME phase_selective_expanded_v1 reference lower bound,
629346312 bytes / 52428800 entries. This is a lower-bound witness from the
same two largest distinct-content reachable native Conv leaves, not a claimed
measurement of the complete expanded reference. Every reference row was
checked bitwise. Both unchanged construction gates passed (resident-growth
upper bounds 524480512 and 523857920 bytes). Reference construction preceded
native model loading; no later measured construction follows loading.

Independent audit again checked 11160274 logical nonzero coefficients, all
243162 main and 28 radix definitions, all inherited predicates, unique
extension and original-prefix projection. It streamed 75888888 original
operator entries. Equality is over the exact-real uncomposed stored affine
program, not bitwise equality to a differently rounded composed matrix.

The unchanged native backend retained 11160330/11160330 matrix coefficients;
all matrix, row/column bounds and integrality checks passed. No threshold or
solver option changed. No presolve, optimize or solve was called. Diagnostic
peak RSS including reference witnesses and native loading was 3006632 KiB;
this is NOT a claim that whole-process peak memory is below 1 GiB.

## Scope and next action

CHECKPOINT_QUALIFIED is exactness, numeric-storage and ingestion evidence for
the saved complete pre-ReLU78 suffix. It does not establish live ReLU78 cache
publication, global phase-slot disjointness in that live transfer, post-ReLU
native fidelity, terminal verdict, same-structure shadows, 400/2413 retention,
or concurrency speed. All those remain unproved. The next experiment must
preregister an identity-free structural runtime rule, qualify its live
consumer and whole live state, and only then attempt ordinary terminal MILP.
No score change, default enablement, or historical-data mutation is justified.
