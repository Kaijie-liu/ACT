# Checkpoint64: exact radix16 coefficient lift, physically rejected

Append-only C55 mathematical/physical negative checkpoint on redu-hz. See
C55_BINARY64_LIFT_AUDIT_20260912.md, C55_ARCHIVE_INTEGRITY_20260912.json,
C55_GAUGED_CONSUMER_HANDOFF_20260912.md and the companion SHA256SUMS.

22 focused tests pass; all three source/actual-equality/inverse/restore proofs
complete. Chain and Conv/ReLU fail full numeric-entry and combined-storage
gates. Shared Add passes but cannot be selected alone. The worker terminates
with exit2 and acceptance=false; no actual solver/network execution occurred.

C54 checkpoint63 was completed on resumption before C55; its positive v2
prototype is not overwritten by this unsuccessful lowering attempt.
Production/source-history boundaries and all baseline/promotion requirements
remain intact. No instruction-audit proposal was applied.

The preceding read-only instruction audit was not Neural-HZ experimental
progress. THIS turn makes progress by completing C54 archiving and establishing
an exact binary64 realization with a reproducible physical rejection that
changes the next design. The goal remains ACTIVE, not achieved or blocked.
Formal1870/2413 and separate E0 CIFAR25/Tiny36; formal gain0.
