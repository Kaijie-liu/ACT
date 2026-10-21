# Guarded GenBaB specification controls passed

The once-only standard-library batch passed25/25 fixed controls at execution
revision `9b1d0790f4a39bc5f2a5070d97e132c0c172981c`. The existing supervisor
recorded exit0, identity-bound cleanup with no live owned-session members, and
matching controller FD counts. This validates specification construction and
structural result reception, not execution of GenBaB or a network proof.

The [protocol](genbab_spec_control_20261021.md),
[freeze](genbab_spec_freeze_20261021.json) and
[lossless deduplicated records](genbab_spec_records_20261021.json) preserve the
complete roster. Freeze SHA256:
`82b36d359c2dd1710996efaa5015f73c649def3c1a4c477bfa20b05866fe4d8b`.
Original terminal SHA256:
`f41ac530f76dd3543d5b0ab10ef09822bff5d14f3d8dab53f9c9d674954232d6`.

## Outcome and limits

The controls exercised guard direction without imposing a selected-score order,
tie inclusion, signed constants, all output coordinates, AND/OR grouping,
binary64 conversion, and full pair/property coverage. Missing or contaminated
receipts were not accepted; SAT remained a replay candidate. The full8-expert/
10-class fixture contained252 clauses and3,276 atoms. These are small logical
matrices, not full-source HybridZ capacity evidence.

The same deadline reaches specification construction and reception. The worker
ran under8GiB AS; RSS was telemetry. Script entry to just before terminal export
was0.393038s. The pre-entry Python startup and final export/exit tail are not
included in that number, and no speedup or complete-request cost follows.
Original records occupy37,078 bytes; the deduplicated archive13,332 bytes.
All25 original prefixes reconstruct byte-for-byte from the final records.

The [saved-only auditor](../scripts/audit_genbab_spec_saved.py) passed source,
freeze, prefix, outcome, deadline, limit, session and artifact checks without
rerunning the controls. Its result is a structural audit, not an independent
proof of solver bounds. The separate sigmoid pair graph was reviewed statically;
neither torch, an external numerical backend, a trained model nor a GPU ran.
Real-stage usage remains0 stages/0 calls/0 execution seconds.

## What the next native comparison must address

The pinned source review found that `VerificationSpec` converts inputs/clauses
using the default dtype at construction time. The native bridge must establish
float64 first and verify actual endpoints and matrices. It must inspect the
solver status, not `success`, which includes both verified and falsified results;
avoid the API lower-bound helper that casts tofloat32; bind configured nonlinear
BaB separately from observed incomplete/nonlinear execution. The standard-library
receiver cannot independently establish these runtime facts.

The existing clean-only metadata is insufficient to select new inputs without
new forwards: the saved float64 three-model scans cover4000–4086 and4087–4389,
and every jointly correct row already entered the old30/100 cohorts. Selection
therefore needs its own finite, charged forward roster. It must not hide a
128×3 scan inside two nominal preparation slots. The old exclusion scanner also
misses later diagnostic schemas/directories; resolve those identities before
freezing new indices. The old external runner's free input-preparation boundary
must not carry over to the new300-second comparison.

Next: finish a schema-aware history exclusion inventory, preregister a bounded
selection/query ledger and same-object input contract, and freeze the isolated
CPU native environment/adapter compatibility stage. No repair of the old RSS
chain, A2 retry or another specification-control batch is needed.
