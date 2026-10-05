# Checkpoint114 / C117

Overall Neural-HZ goal ACTIVE; C117 v1 CLOSED with a complete structural
negative and a failed full-ledger gate. Branch `redu-hz`, commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.

- All 3,341 tests / 144 files pass in 53.374763673 s including collection.
- Fresh original network; all 36 nodes, 24 distinct operators, 36 distinct
  complete programs. No whole-node CSE candidates. No archived HZ restore.
- 8,317,921 non-source edges match complete packed consumer occurrences.
  Dominant Conv block: 3,986,304 edges; two largest Conv blocks: 49.3478% of
  raw affine edges. These are not new reductions or final native predicate nnz.
- Graph and ledger stages both pass measured 1 GiB transient limits, but
  full ledger rejects mixed dense/CSR roles on the same op11.data allocation.
  Full 64M-entry ledger and final expression/frame preservation remain unproved.
- Shared diagnostic work 253,332,606/256M; complete census/arrays, prefix,
  test inventory, failure and provenance receipts saved. Session 51701 exit 1.
- No new solver call, witness, source/LIVE admission, formal gain or default.

[Audit](C117_AFFINE_BLOCK_CENSUS_AUDIT_20260922.md),
[next structural questions](C117_STRUCTURAL_HANDOFF_20260922.md).
Results: `results/c117_affine_block_census_20260922_v1`.
Separate terminal integrity and checkpoint SHA256SUMS seal the archived state.

Formal 1,870/2,413 and all 13 family records unchanged; E0 CIFAR 25/200 and
Tiny 36/200 unchanged. Production and historical archives were not edited.
Next work is a newly preregistered exact block-algebra question, not a retry
of unchanged whole-program CSE or a relaxed memory/solver/permission boundary.
