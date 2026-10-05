# D003 v1 preregistration — bounded exact reference, not benchmark admission

2026-09-28; branch `redu-hz`; commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`.
Read with `D003_REFERENCE_DESIGN.md`, D001 definition and D002 scope proofs.
All code and this document are frozen before any new module import, pytest
collection, numerical evaluation or model execution. Static AST/source review
does not qualify as a pass. This is one attempt; failed bytes/results remain.

## Falsifiable question

Does the phase-dependent continuous-affine DAG reference preserve the entire
original input/phase/output relation through ordinary mixing affine, ReLU,
same-frame residual Add, Concat and original EQ/LE constraints, and does its
complete finite-binary linear terminal representation represent that same
relation? A pass establishes implementation evidence for the paper induction;
it establishes neither new abstraction power nor a research novelty claim.

The exact reference always retains original free bits and all ReLU bits,
including stable and unconsumed gates. Binary values are integral 0/1, not a
relaxation. A ReLU node evaluates to beta*f with the sign guard, not max(f,0)
without the original phase assignment. Its four-row lowering retains p+Q
continuous variables and all B bits. Boundary f=0 admits both bit values.
Intermediate predicates and original-input inverse remain visible.

No model/property solve, model/HZ payload deserialization, production edit,
default enablement, benchmark score update, old archive edit, commit or push.
There is no attack, split, dual/backward rescue or instance-identity dispatch.
The new reference/diagnostic has no solver. The retained inherited tests
include their existing small LP/MILP calls; global solver_calls=0 is NOT claimed.

## Frozen populations and gates

- Inherit **all 159 files / 3660 node IDs** from qualified C130. Add exactly
  **24 explicit D003 tests**, total **160 files / 3684 node IDs**. No skips,
  drops, xfails or altered old tests; actual collection and JUnit population
  must match the preregistered list exactly. Combined collection+execution
  wall time **<=60 seconds**, no retry of a failed frozen version.
- Child address space **16 GiB**; one allowed CPU via process affinity plus
  single-thread environment; GPU disabled. The 60-second gate is unchanged;
  explicit affinity/resource enforcement is not a relaxation.
- D003 reference limits are bounded prototype rejection limits: p<=16,
  free binary<=8, total binary<=64, nodes and vector/output handles<=256,
  predicates<=128, fan-in<=32, nonzero affine edges<=4096, reduced rational
  numerator/denominator<=512 bits. They do not enlarge old source budgets.
  Invalid operations reject without partial graph mutations or verdicts.
- Run one complete small mixed-residual fixture diagnostic only after the
  full test gate. Worker wall **<=240 seconds**; actual retained entries
  **<=64,000,000**, RSS high-water growth **<=1 GiB**, and traced peak plus
  tracer metadata **<=1 GiB**. All complete candidate and independent evidence
  roots remain held during measurement and evidence encoding.

## Independent comparator and fixed diagnostic

The test fixture and independently spelled-out comparator use 2 continuous
inputs, 1 original free binary, 3 ReLUs, 13 nodes, 18 affine edges and 4 original
predicates. The native reference has 5 continuous variables, 4 binaries,
16 rows and 50 constraint nonzeros (66 including RHS entries).
There is no speed or compression gain hypothesis for this control.

The worker retains the builder, frozen element, complete lowered reference,
independent ordinary direct evaluations, independent linear forms/rows/bounds,
and fixed sample inputs, complete bit vectors, candidate evaluations and full
native assignments together. Exact comparisons include every node and output,
all native rows and bounds, feasibility and original input reconstruction.
Hand-picked finite regression points are not a proof of full-set equivalence
or a search/attack. Nodewise induction plus the four-row binary case proof is
the mathematical argument; tests can falsify its implementation.

Fixed diagnostic `(input, free_bit, zero_phase)` list: `((0,0),0,0)`,
`((0,0),0,1)`, `((1/2,1/2),1,0)`, `((-1/2,1/2),0,0)`, `((1,-1),0,0)`.
Bits come from independent direct preactivation signs, with the chosen bit
only when exactly zero; this is not phase enumeration or solving a property.
All data evidence and its encoding are measured together. A 65,536-byte
final-summary reserve is added to both memory checks; no whole-source memory
qualification follows. Module/class singletons are outside the per-instance
held ledger, while their import allocations are included in the transient trace.

The full held-object traversal uses actual dataclass fields, builder dictionary,
containers and Fraction numerator/denominator, deduplicating object identity.
It is a descriptive Python-object ledger for this reference fixture, NOT the
old native-HZ NumPy storage qualification or a replacement for that ledger.
Evidence encoding is inside the memory trace; proof/diagnostic cost is included.
D003's coarse scalar/retained bounds are documented in the design; no old
source-construction budget credit or coupled-work tariff is claimed.
The old native [2^-20,2^40], source/branch and native-row admission gates still
apply to any future ACT integration; Fraction-only reference success does not
mean passing them. No C130 complete-source worker is replaced or requalified.

## Provenance, single-use execution and interpretation

Run directory: `results/d003_reference_semantics_20260928_v1`, created exclusively.
`run_d003_reference_v1.py` binds the C130 preregistration, inventory, exit,
terminal audit and 64-file seal by their recorded SHA256 identities. Before
the first import it authenticates all inherited .py/.so identities (safe
superset of test dependencies), every old test, the Python executable, all
13 production-provenance files, and all new D003 source/design/prereg files.
It streams and hashes the 3 inherited original input files without decoding.
It records hashes before and rechecks them after execution. The entire old
manifest is bound but its non-executed historical result payloads are not
individually rehashed; this is NOT a fresh whole-archive integrity audit.
Hash traffic is outside the numerical test clock, not claimed as free runtime.

The runner saves collection, exact inventory, test output/JUnit, complete
diagnostic evidence, measurements and final exit state without overwriting.
Any mismatch, exception, timeout or gate failure leaves this version failed;
the diagnostic does not run after a failed test gate. Reports distinguish
semantic regression success, fixture memory success, source qualification,
novelty and benchmark gains. Only the first two can pass in D003.

Formal baseline stays **1870/2413 = 1063 CERT + 807 validated ADV**; separate
E0 stays **CIFAR100 25 + TinyImageNet 36 =61/400**, not added to 1870.
Neither population is replayed or requalified by these tests. All 13-family,
every-old-solve and full-replay guards remain prerequisites for later promotion.
Next research must test a guard-aware definition-level simplification against
strong ordinary-HZ/shared-circuit comparators on real common structures; a
passing unnormalized DAG alone is not the requested Neural-HZ breakthrough.
