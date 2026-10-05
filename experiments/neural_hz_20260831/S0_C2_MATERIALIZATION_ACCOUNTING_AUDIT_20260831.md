# S0-C2 Isolated Materialization and Whole-State Accounting Audit

Date: 2026-08-31  
Branch: `redu-hz`  
Repository base commit: `f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`

This audit covers two disconnected, default-off prototypes that remain useful
after S0-C2 closed as a real-graph zero-hit.  They are reusable safety
infrastructure for S0-C3 or a later structural rule.  They are not production
integration, target evidence or score evidence.  The formal baseline remains
1,870/2,413 and E0 remains 61/400 with Neural-HZ gain zero.

## Real-CSR artifact transaction

Files and current SHA-256:

```text
613c81fc8e8b157f96c2d86f685044d85d202d1b99b7be7c66af76736d0feb4  s0_c2_emission_artifact_transaction_prototype.py
6954f80df55081739107c861003952f23adf623f38b82e633c904fdb08f4cab3  test_s0_c2_emission_artifact_transaction_prototype.py
```

The prototype establishes a request-local ownership/accounting boundary:

- input `Q` and output matrices are rebuilt as canonical, owned, read-only
  float64 CSR buffers;
- the cache key independently encodes descriptor snapshot, support, reverse
  prefix and the complete canonical `Q` digest;
- a cache hit returns the identical strongly retained artifact and waives no
  work unless that real artifact still exists and revalidates;
- actual result nnz, all CSR buffer entries/bytes, retained input/output bytes,
  cumulative work and controlled transient are recorded;
- request token, build-open state and seal state prevent cross-request reuse;
  and
- ordinary failure restores the pre-build ledger; asynchronous interruption
  discards the incomplete build and propagates.

A main-agent audit found a pre-callback allocation gap: `_build_open` could be
set before nonce allocation outside the rollback region.  The allocation now
occurs inside the guarded region, and synthetic `MemoryError` and
`KeyboardInterrupt` tests prove no half-open build survives.

The descriptor callback remains a declared trust boundary.  Its authenticated
handoff prevents callers from supplying an arbitrary ledger, but does not
prove that V2's actual gather/compose work was instrumented correctly.  The
descriptor/support/prefix snapshots are supplied rather than derived from a
live production expression.  Consequently the 15 passing tests establish
transaction behavior only; they do not establish real emission or a target
hit.

## Whole-state numeric-storage ledger

Files and current SHA-256:

```text
d9fd6948c0d8e7fc4e5399846ec650d0b9b28f3044ca97dbee242d4002098c2e  s0_c2_whole_state_ledger_prototype.py
e2da78953b6085d2c23cc2c85512b0757be1eeb84852f1b6facd89f82e3cd503  test_s0_c2_whole_state_ledger_prototype.py
85e7f64b2e1623f53d4eb2a9efc5082ae9b1114f6213278d460a8ddad2defe44  test_s0_c2_whole_state_ledger_adversarial.py
```

The ledger snapshots only explicitly registered strong roots.  It recognizes
production-shaped sparse HZ values and every equality/inequality predicate
buffer, affine expressions and operators, all supported precomputed ReLU
tuple slots, phase bounds, descriptors, artifacts, active transactions and
pending strong-retention plans.  Weak references are observed but never
promoted to strong roots.

Candidate substitutions are identity-CAS operations on a private mapping
snapshot.  The unchanged baseline and staged candidate independently execute
the same consumer-GC steps before measurement.  Thus an ordinary
last-consumer release cannot be booked as candidate reduction.  A remaining
or pinned successor keeps the predecessor roots on both sides, covering the
second successor of Tiny iid143 ADD32.  Acceptance requires both strict
inequalities:

```text
candidate.resident_bytes   < baseline.resident_bytes
candidate.resident_entries < baseline.resident_entries.
```

Equal metrics reject.  CSR entries count numeric data, while index/indptr
still contribute physical bytes.  Torch aliases are deduplicated by complete
untyped storage without publishing pointer values.

### Material defect found and repaired

The first ledger version keyed NumPy storage by final owner plus visible view
span and charged only `view.nbytes/view.size`.  A small slice strongly retains
its complete base allocation, so that rule could materially undercount the
candidate's resident state.  The repaired byte rule keys the final owner and
charges `owner.nbytes` once.  Dense entries use `owner.size`; the frozen CSR
entry convention remains `matrix.data.size`, and a CSR data view that does not
cover its complete owner now fails closed rather than substituting owner size.
View spans remain alias evidence.  Exact or disjoint dense views deduplicate
one owner; partially overlapping, gapped, external-buffer and
incompatible-dtype views fail closed.  New tests prove that a four-element
slice of a 1,024-element dense owner is charged as the complete owner and that
disjoint views retain exactly one complete allocation.

A second audit then closed two more fail-open paths.  Every distinct view span
is now registered before later overlap comparisons, so a third view cannot
hide a partial overlap with the second.  Direct substitution of graph
consumer-managed `sparse_hz`, `affine_expr` or `phase_bounds` roots is
forbidden; only the identical baseline/candidate consumer-GC simulation may
release them.  This prevents a candidate from deleting ADD32 state while its
second successor still has a live consumer.  Request-local `active`,
descriptor and artifact roots remain available for hypothetical candidate
substitution.

A final compound-root audit closed a further alias/provenance omission.  A
numeric object may be reached through more than one registered strong-root
role, so seeing the object once cannot suppress traversal of the later role's
numeric children.  Object storage is still deduplicated, but every role is
traversed for provenance and numeric descendants; an active-recursion guard
rejects strong-root cycles instead of accepting an incomplete snapshot.  The
adversarial suite covers alternate-role reachability, nested compound aliases
and cycles.

Ordinary mapping/allocation failure returns a stable rejection without input
mutation.  `KeyboardInterrupt` and `SystemExit` propagate without mutation.
The repaired ledger passes 26 primary plus 33 independent adversarial tests,
59/59 total.

## Combined result and remaining boundary

The descriptor/compiler, adversarial planner, artifact transaction and ledger
selection used in this audit pass 183 tests together.  All files remain under
`experiments/neural_hz_20260831/`; ACT production imports none of these two new
prototype modules.

The following claims remain explicitly open:

- the generic ledger does not yet traverse the concrete V2 descriptor and
  `EmissionArtifact` dataclasses through production adapters;
- a pure snapshot comparison is not an atomic publish transaction and does
  not measure allocator overhead or peak RSS;
- local caller/substitution objects coexist during simulation and are not a
  proof of post-publication root release;
- the callback is not an audited V2 materializer, and actual cumulative
  four-branch emission has not been measured;
- no Tiny/CIFAR run, whole-state target reduction, concurrency result,
  family replay or full 2,413 replay exists.

Therefore C1 gain = 0, C2 gain = 0, C3 gain = 0, and the formal score remains
1,870/2,413.  The isolated BatchNorm V2 clone certificate now proves the
19-edge graph correction mechanically, but production loader correction and
an immutable live operator-lineage adapter with payload CAS remain open.
These prototypes cannot advance to a target until that complete runtime
bijection succeeds.
