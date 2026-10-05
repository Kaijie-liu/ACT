# S0-C2 Implementation Clarification V1

Recorded on 2026-08-31 on branch `redu-hz` at repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, after isolated pure-planner
unit tests and before any S0-C2 target benchmark, real materialization or
production edit. This is a non-relaxing implementation clarification of
`S0_C2_RESIDUAL_DISTRIBUTIVE_PREREG_V1.md`; it does not change the target,
budgets, advancement order or closure conditions and creates no score.

The formal baseline remains exactly 1,870/2,413 with all 13 family counts as
hard lower bounds. E0 remains 61/400 with zero Neural-HZ gain credit.

## Frozen occurrence semantics

The latest common ADD is proved only by one recursively immutable, nonempty,
non-boolean equality token. `None`, bools, mutable containers, NumPy arrays and
arbitrary objects are not occurrence identities. The same rule applies to the
shared HZ frame token: an expression and all sources must encode to the same
stable frame payload. Raw Python `==` is not an authority because it can
accept unframed values, mutable containers or array-valued comparisons.

The ADD mark may occur at a different operator index in different branches;
that is expected because branch lengths differ. Tuple order records graph
occurrence order. The last mark must be the common ADD. A nonidentity branch
segment begins strictly after its preceding mark and ends at that ADD. It must
be exactly one implicit Conv followed by one or more exact channel-stationary
diagonals. Any incomplete segment rejects the complete request.

## Frozen shared suffix semantics

The common suffix is the exact sequence from the latest ADD through the
shared outer Conv:

```text
ADD -> zero or more common channel diagonals -> shared outer Conv.
```

For V1, every participating term must reference the same outer Conv object
and the same common diagonal objects. Object identity proves one graph event;
fresh current-payload snapshots additionally prove that no stale constructor
key is being used. Equal-valued separately allocated operators are not enough.

Diagonals after the outer Conv are term-local `Dout` keep/mask maps. They may
differ across terms, are excluded from the composed descriptor key, retain
their original object identity and determine each term's actual selected-row
support. Descriptor-level support is the sorted union across all uses of that
same current semantic descriptor.

An identity skip has an empty segment immediately before the common ADD. It is
not composed. Its entire original prefix, common diagonal sequence, outer Conv
and term-local `Dout` remain unchanged. A zero-support term also remains an
original term with its original source and predicates. Neither category can be
deleted or used to waive global bias propagation.

## Pure planner boundary

The isolated pure planner may prove only:

- one latest common ADD and one uniform all-or-nothing branch set;
- current-payload descriptor grouping and deterministic request order;
- source/term/bias/frame identity preservation in the prospective plan; and
- conservative per-descriptor and cumulative arithmetic reservations.

It deliberately has no compile, execute, rewrite, fallback or publication
interface. Its plan still holds live operators and the original expression,
so it is neither a frozen compile payload nor a success-state object. A future
materializer must re-snapshot and privately clone payloads, then discard every
plan/original-term reference before the persistent-state comparison.

The current C1 estimator does not prove the one-quarter work gate, actual
emission, canonical CSR nnz/bytes, peak RSS, old-root release, strict
whole-state reduction, production lineage, four-concurrent behavior or a Tiny
iid143 hit. All remain no-claims.

## Real emission boundary

The descriptor-only V2 transaction cannot be promoted by summing prospective
emission estimates. A real request-local materialization transaction must:

1. seal all unique descriptor compilations;
2. canonicalize and privately own the real left factor `Q`;
3. key a strong CSR artifact by the descriptor's current semantic payload,
   actual support, reverse-prefix semantics and canonical `Q` digest;
4. perform reserve, real `gather_rows/left_compose`, canonicalization, actual
   ledger validation and artifact publication inside one atomic interface;
5. retain and return the identical CSR object on a true cache hit, or execute
   and charge again when no artifact exists; and
6. enforce the 256M gate over all unique descriptor contractions plus every
   newly constructed artifact in the complete request.

A key without a strongly retained, integrity-checked artifact is never an
emission cache hit. Actual canonical `nnz`, data/index/indptr bytes and all
construction transients must be measured, not inferred from a support key.

## Persistent whole-state boundary

The existing shadow live-cache ledger remains diagnostic only. It double
counts shared predicate backing buffers and omits top-level phase bounds,
precomputed expected bounds, active/transaction/artifact roots and several
aliases. It cannot be renamed into the strict gate.

The persistent comparison belongs after current-layer publication and the
same simulated consumer-GC step on both sides. Baseline and candidate virtual
root views must include all HZ, affine-expression, precomputed-ReLU and phase
bound caches, pending result roots, descriptors/artifacts and any externally
pinned terminal HZ. Substitution is path-specific and guarded by expected
object identity; a global operator-id replacement is forbidden.

NumPy and Torch views are deduplicated by backing storage, not wrapper object.
Physically distinct equal-content allocations count separately. Both

```text
candidate.resident_bytes   < baseline.resident_bytes
candidate.resident_entries < baseline.resident_entries
```

must hold at the same boundary. Numeric storage is reported separately from
Python object counts and measured peak RSS. Any still-reachable plan or alias
that retains an old term is charged in full and may cause rejection.

## Advancement status

Only isolated planner, artifact-transaction and whole-state-ledger work is
authorized while Trial 9 owns the nine-file production source freeze. The
next target remains TinyImageNet iid143 at ReLU36, and it may run only after
real emission, full HZ/predicate/witness equivalence, strict whole-state gates
and fail-closed atomic publication exist. Until then, S0-C2 formal and E0 gain
remain zero.
