# C54 v2: one joint-latent CSR and a canonical packed inverse record

Frozen v1 diagnostic completed all source/Fraction proofs and48 feasible toy
points, with107-417 bit exact coefficients and no binary64 rounding. It failed
the physical gate: chain combined accounting126096->127968B; Conv/ReLU numeric
entries1230->1242 and combined128148->130304B. Shared Add passed. Both1GiB
memory gates passed, but overall v1 remains CLOSED/NOT ACCEPTED.

New physical layout, same first-write mathematics, arithmetic limits, original
source programs, fixed cohorts and binary64 reference comparator:

- Store EQ, INEQ and output rows in one CSR, with immutable row partitions.
  Columns[0,n_cont) are continuous and[n_cont,n_cont+n_bin) are binary.
  This is joint storage, not merging continuous and discrete domains.
- Concatenate EQ RHS, INEQ RHS and output centers in the same row order.
- Pack each inverse's unsigned32 root and unsigned32 exact-scalar ID into
  one uint64 record, with checked range and a uniquely defined decoder.
  Both records occupied8 bytes already; this removes a redundant array and
  genuinely encodes one fixed-width record, rather than dropping an index.
- Keep the exact scalar pool, global IDs, removed UID/column pairs and EQ/INEQ
  labels complete. No scalar approximation, source omission or comparator change.

The v1 first-write builder is reused unchanged. Its temporary split storage is
packed and all23 unshared numeric buffers must be weak-retired before returning
the v2 state. Retained global/UID/scalar arrays are explicit shared owners.
The measured window includes the v1 construction, packing and retirement;
v2 does not claim an already fused real-network source writer.

A checked exact-scalar split view is used ONLY for independent Fraction audit
and toy evaluation. It is NOT a binary64/native lowering adapter. Its temporary
arrays and work are inside the prototype measurement. Packing cannot substitute
for original-network binding, terminal precision handling, native/frame/witness
proof, current-C31 comparisons or a full verification request's LIVE boundary.

Require the same four strict physical comparisons on ALL three fixed cohorts.
Run a new v2 focused suite with the same mathematical/physical/inverse guards;
the frozen v1 tests remain failed and archived, not reclassified or modified.
No claim of complete old-suite qualification follows. No actual target run.

All previous CPU1/GPU0,AS16GiB,64M entries,whole256M/nested200M,60s focused
tests,240s worker and both1GiB limits stay. Exact scalars remain<=512 bits and
in the unchanged magnitude window. Each new source/version/output is exclusive.
The measured transaction exports protocol5 and replays through unchanged C41:
recompute all6 source proofs,48 toy points and all physical gates; retain and
measure both complete fresh and restored archives. No source or restoration
work is moved outside the measurement. No original pre-pickle view-alias claim.
Formal1870 and separate E0 CIFAR25/Tiny36 unchanged; no forbidden rescue,
binary pivot, convex replacement, native/default promotion or old-result write.
