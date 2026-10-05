# C10 fused alias emission V1 — same S0 structure, new transaction

2026-09-08, redu-hz. The preceding interrupted design turn did not implement
anything or launch a job. All C10 quotient V1 outcomes remain sealed. This
version moves that exact continuous-only quotient INSIDE affine construction,
before any complete HZ matrices are materialized. No new structural family,
solver rescue, production enablement or score promotion is authorized.

## Fixed identity and physical realization

Use the original C9 uncomposed affine DAG and exact radix row encoder, preserving
all original predicates and global factor indices. Classify its owned temporary
rows, not a second completed HZ. The alias criteria, descending independent
frontier, exact products/window and exact collision addition are those of C10
quotient V1. Root output slots, original prefix, radix and all binary factors
are protected. Do not restrict selection to power-of-two ratios or a chosen iid.

Discover eligible aliases from their registered two-column homogeneous rows,
scan owned continuous incidences once, and cache only alias-incident products.
After selecting the frontier, rewrite only hit rows, erase selected definitions,
and emit the final CSR matrices ONCE. Temporary original row buffers are local
construction workspace, not retained certificate/source HZs. No input source,
native prefix cache or original operator may be deleted or overwritten.

Reuse the two existing int64 logical-row maps as a tagged reconstruction record:
nonnegative eq_roots = physical row and eq_scales = dyadic shift as before;
negative eq_roots = -(parent_global_column+1), eq_scales = float64 ratio bitcast
to int64. The logical row index identifies the erased MAIN column. Original
predicate rows cannot carry alias tags. This representation adds NO persistent
reconstruction arrays and cannot hide a full old HZ. Reconstruction is still
the exact unique x_j = ratio*x_parent; selected parents are never selected.

## Bounded exact arithmetic and work

Specialize the already-tested odd-significand product criterion to physical
coefficients in [2^-20,2^40] and alias ratios in [2^-60,1] in magnitude. Products
are then normal/nonzero/nonoverflowing BEFORE the unchanged output-window check.
Precompute the ratio odd significand/bit length once. Check the remaining
operand and sum of odd bit lengths; the 54-bit boundary uses an exact uint64
product. This removes the general routine's repeated ratio decomposition and
subnormal-grid tests, not its exactness requirement. Independent Fraction and
the general C10 criterion must agree on proof fixtures and boundary cases.

Frozen ceilings remain whole work256M, largest branch200M, entries64M,
construction1GiB under the unchanged measured_build gate, radix auxiliaries16384,
extra radix entries131072, extra radix work16M, worker240s, tests60s, address
space16GiB, CPU1/GPU0. Keep original C9 affine/old-predicate base-work charges.
Use ONE coupled extra-work counter, checked before each operation, with capacity
min(256M - whole_base, 200M - largest_branch_base). Radix still independently
cannot exceed16M. Do not reserve a second16M in addition to actual radix work;
the coupled pool may fail closed earlier if alias work has consumed room.
This is a new preregistered allocation policy, not a reclassification of C9.

Alias work charges:32 per MAIN definition (including metadata/frontier), one
per owned continuous coefficient inspected,32 per remaining alias occurrence
for specialized exact products, and4*w*ceil(log2(max(2,w))) for every rewritten
row, plus32 per coefficient in a collision group. Reuse cached exact products
at emission: do not calculate them a second time for free. All sorting/collision
charges precede rewriting. Original row-to-matrix copying remains covered by
the original C9 emission accounting. Audit/hash/native diagnostic work is
reported separately, as in the predecessor; no full-pipeline speedup is inferred.

## Acceptance ladder and retained evidence

Proof fixtures must cover tags, shared frames, all binaries/old predicates,
signed and non-power-two aliases, chains, sibling collisions, rejected inexact
addition, retained radix definitions, default-off/mutation/cap guards and exact
original affine output recovery. Then one original ADD75 checkpoint build
(d08086844eacebbd69c2ddc3c4ffc77ebf1aabc90539e95547744da929273fed), through
its registered suffix expression, with all output rows and the original global
frame widths. This is a fresh construction from the expression, not reuse of
the old completed HZ. Compare against the sealed C9 live preactivation HZ only
as an independent audit artifact. Its full numeric payload is not hidden in
candidate accounting; diagnostic coexistence and RSS must be reported.

The oracle loader retains exactly the complete preactivation HZ and its five
original row maps from the live checkpoint; unrelated postactivation/runtime
objects in that file are deserialization temporaries, discharged before the
construction measurement starts. ALL objects in the input ADD75 snapshot,
including its native prefix caches, remain retained unchanged. The full
offline union includes that snapshot, the new expression/graph/HZ/tagged maps,
and the complete external original preactivation HZ/maps. Report this union
explicitly and compare it against the same frozen two-leaf reference rule.
Build the reference before native ingestion to avoid the already-diagnosed
native diagnostic lifetime-HWM interaction; do not reset the memory gate.

Require original exact affine/predicate proof, exact equivalence to the sealed
C9 preactivation HZ through an independent quotient oracle, strictly fewer
component bytes AND entries including ALL tagged maps and original graph,
and no native coefficient/bounds/integrality loss. Then perform the same
complete reachable-root/frozen-reference qualification; full live integration
and terminal calls require their own subsequent preregistration. No offline
success counts as fresh live publication, capability or formal gain.

Exclusive directory results/c10_fused_emission_20260908_v1/. Freeze all inherited
and new sources/configuration/provenance before its first target run; retain
tests/events/results/failures/exit hashes. Failed target versions stay closed;
do not change gates or rerun the same name. Formal1870/2413 and independent
E061/400 remain unchanged until their respective complete replay requirements.
