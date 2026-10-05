# S0-C3 Same-Frame Residual-Distributive D* Preregistration V1

Preregistered on 2026-08-31 on branch `redu-hz` at repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, before any S0-C3 target run or
production integration.  This is an independent candidate rule.  It does not
reinterpret or reopen S0-C1 or S0-C2: their Tiny iid143 ReLU36 real-graph hit
counts remain zero and their gain remains zero.

The formal baseline remains exactly 1,870/2,413: 1,063 CERT plus 807
concretely validated ADV.  Every solved count in all 13 families is a hard
non-regression constraint.  E0 remains 61/400 independently replayed
historical ADV with zero Neural-HZ gain credit.  This document creates no
score and authorizes no benchmark run while the graph-faithfulness gate below
is unresolved.

Frozen identifiers are:

```text
rule_id        = s0_c3_same_frame_residual_distributive_identity_middle_v1
descriptor_tag = s0_c3_current_snapshot_composed_descriptor_v1
payload_prefix = S0C3D1
```

## Exact set theorem

All sources share one exact HZ frame and the same concrete factor witness
`(xi,z)`.  Immediately before the newest common ADD, write

```text
r = sum_i Pi_i C_i u_i + sum_j u_j + d,
Pi_i = D_i,m ... D_i,1,  with m >= 0.
```

`C_i` is a complete branch's unique inner Conv.  Every `D` is an exact,
finite, channel-stationary diagonal map.  For `m=0`, `Pi_i` is the fixed empty
product identity.  The second sum contains only branches whose entire segment
before the ADD is graph-certified empty; `d` is the expression's single
already-propagated global bias.

Let the exact common post-ADD diagonal product be `Gamma`, also allowing an
empty product, and let the shared outer Conv be `(B,beta)`.  Then

```text
B Gamma r + beta
  = sum_i (B Gamma Pi_i C_i) u_i
  + sum_j (B Gamma) u_j
  + B Gamma d + beta.
```

This is equality over the same nonconvex set, not a relaxation:

- `c/Gc/Gb`, continuous and binary factor slots, phase identities and source
  order are preserved;
- `Ac/Ab/b/Auc/Aub/ub`, `frame_id` and `exact` are preserved;
- equal-valued but distinct source occurrences are never merged by value;
- the same concrete `(xi,z)` reconstructs the same network input and output;
- zero-support terms and their predicates remain present; and
- the global bias passes through the shared map once and `beta` is added once.

Term-local output diagonals remain outside the distributed core.  Output row
masks affect term maps/support only and may not mask, duplicate or discard the
global bias.  The theorem is exact over reals; non-dyadic binary64
reassociation is not claimed bitwise identical.

## One structural rule

S0-C3-v1 has one grammar and no fallback menu:

```text
prefix_i
 -> ImplicitConv(inner_i)
 -> exact channel-stationary diagonal*
 -> latest common ADD
 -> exact shared channel-stationary diagonal*
 -> one shared ImplicitConv(outer)
 -> term-local output diagonal*.
```

The pre-ADD segment is classified once from immutable lineage:

- empty segment: graph-certified identity skip, retained unchanged;
- `Conv,D*`: complete branch, selected as part of the full transaction;
- any other nonempty segment: reject the complete request.

At least one complete branch must have nonzero active support.  Every complete
active-support branch is selected; resource budgets cannot choose a favorable
subset.  A complete zero-support term remains unchanged, but malformed
lineage on it still rejects the request.  A second ADD, reshape-family event,
nonlinearity, pooling, Dense, ConvTranspose, permutation or unknown event in
the core rejects the whole request.

The newest common ADD, shared outer Conv and common post-ADD suffix must match
both graph occurrence identity and current semantic payload snapshots.  Value
equality alone is insufficient.  Trigger inputs are limited to exact graph
events, operator types/shapes/current payloads, support and frozen resource
metrics.  iid, family, property, layer number, margin, verdict and elapsed
behavior are forbidden selector inputs.

S0-C3 never tries S0-C2 first and never reclassifies a C2 rejection.  C2 keeps
its frozen `Conv,D+` grammar and its empty-middle rejection tests.

## Mandatory graph-faithfulness certificate

An omitted operator tuple entry is not evidence of identity.  Before S0-C3
may plan any target, a generic certificate must prove all of the following:

1. every layer `in_vars` producer agrees with `net.preds`, including exact
   producer occurrence rather than shape or value equality;
2. every multiplicative graph event on an ADD branch has exactly one matching
   semantic term operator in ordered lineage;
3. a decomposed BatchNorm BIAS consumes its paired SCALE output and the SCALE
   is represented in graph path and operator lineage;
4. BIAS-only events may change the single expression bias without occupying a
   multiplicative operator slot, but must remain explicit lineage transitions;
5. a genuine empty `D*` segment is certified only when no SCALE, BatchNorm,
   MUL or other linear map occurs between Conv and ADD; and
6. a known but omitted SCALE rejects as `unaccounted_linear_event`, even if
   its current numeric payload happens to be all ones.

Identity canonicalization, if added later, must be a separately preregistered
uniform transform with event-preserving provenance.  Tuple absence, dangling
successors or the desired target shape cannot serve as identity evidence.

Tiny iid143 currently fails this gate: its variable IDs show Bias31 consumes
Scale30 output and Bias35 consumes Scale34 output, while `net.preds` makes the
Scale nodes successorless siblings.  Both Scale payloads are nonidentity.
Therefore no existing Tiny143 tuple is an S0-C3 hit.  A translator repair is
an independent correctness change and must retain all 1,870 baseline rows
before a corrected tuple can become target evidence.

## Descriptor and lineage separation

The S0-C3 rule namespace and use-lineage are distinct from S0-C2.  Within C3,
an empty diagonal product and an explicitly represented all-ones diagonal may
compile to the same numeric descriptor because they denote the same linear
map.  A descriptor/cache hit never authorizes structure: every use must still
retain and revalidate its own graph occurrence certificate,
`pre_add_diagonal_count`, source order and identity-skip classification.

If an implementation cannot keep numeric reuse separate from structural
authorization, its descriptor key must conservatively include an
empty/nonempty bit.  Hash or key equality alone is never semantic evidence.

## Atomic implementation and whole-state boundary

Every materialization uses a fresh request-local staging transaction:

1. snapshot the expression, all sources/value and predicate buffers, bias,
   factor widths, frame/exact bits, operators, lineage and support;
2. validate the graph-faithfulness certificate and unique full branch set;
3. compile descriptors in deterministic semantic-content order;
4. reserve and charge actual `gather_rows/left_compose` emission around the
   real materializer for each support/reverse prefix;
5. retain and charge a real CSR artifact before any reuse can waive emission;
6. materialize a complete shadow expression and compare exact HZ, predicate,
   frame, bias and witness semantics;
7. compare baseline and candidate complete strong-root state after the same
   consumer-GC event, including all remaining successors; and
8. publish aliases, descriptors, artifacts and counters atomically only after
   every gate succeeds.

Ordinary structural/resource rejection and ordinary `Exception` discard the
staging request and return the identical original production object.
`KeyboardInterrupt` and `SystemExit` discard staging and re-raise.  Failure of
the unchanged fallback remains UNKNOWN, never candidate gain.  A plan or CAS
object that still strongly retains old numeric state must be charged.

NumPy views are charged by their complete final owner allocation, not their
visible slice.  Torch views are charged by complete untyped storage.  CSR
bytes include `data`, `indices` and `indptr`, while entries remain the actual
`matrix.data.size`; a short data view with an ambiguous larger owner fails
closed.  All aliases, descriptor/artifact roots,
precomputed ReLU states, phase bounds and the second ADD32 successor remain in
scope.  Pure hypothetical accounting is not atomic publication or RSS proof.

## Frozen resource and performance gates

The complete request retains the prior numeric limits without adjustment:

- each descriptor contraction: at most 200,000,000 products;
- contraction plus actual emission: at most 256,000,000 products;
- coefficient entries: at most 2,000,000;
- descriptor resident payload: at most 64 MiB;
- controlled numeric transient: at most 1 GiB;
- each retained CSR artifact: at most 64,000,000 nnz;
- fused work: at most one quarter of registered unfused work;
- `after.resident_bytes < before.resident_bytes` at the same GC boundary;
- `after.resident_entries < before.resident_entries` at that boundary; and
- four-concurrent `baseline_time/candidate_time >= 1.0`.

Equal bytes or equal entries rejects.  Prospective emission, logical expanded
nnz, descriptor-only arithmetic, Python object-size estimates and peak RSS are
separate diagnostics and cannot satisfy either strict resident gate.

## Fixed target order

No target execution is authorized before graph-faithfulness and translator
retention tests pass.  After that prerequisite, advancement is fixed:

1. pure set theorem, lineage, graph-certificate and transaction tests;
2. TinyImageNet iid143 ReLU36 with exactly 2,180 registered phase rows;
3. Tiny iid143 ReLU63 only after ReLU36 passes every gate;
4. preregistered same-structure Tiny/CIFAR residual shadows and zero-hit guards;
5. affected-family retained-set replay plus four-concurrent non-regression;
6. all 13 families and all 2,413 rows under one flag/config/path.

At ReLU36 all four terms must be classified unambiguously as complete
`Conv,D*` branches or genuine empty-segment identity skips.  One ambiguous
term rejects the full request.  There is no jump to ReLU71 or another iid
after a target failure.

## P0 proof matrix

The pure test surface must include:

- certified `Conv -> ADD -> outer` and `Conv -> D -> ADD -> outer` accepts;
- mixed empty/nonempty `D*` complete branches selected atomically;
- true empty ADD-side identity skip retained unchanged;
- D-only, two-Conv-on-one-side, nested ADD, reshape/nonlinear and unknown
  events rejected;
- occurrence-distinct but value-equal ADD/outer/suffix rejected;
- graph SCALE omitted from tuple rejected as `unaccounted_linear_event`;
- paired BIAS without SCALE predecessor and `preds`/producer mismatch rejected;
- explicit all-ones D accepted without erasing source or event lineage;
- empty and explicit-one descriptors optionally reused while use-lineage stays
  distinct;
- dense/CSR exact-set oracles, including a dyadic two-branch bias-once witness;
- channel-nonstationary diagonal and malformed zero-support term rejected;
- every resource boundary tested at the limit and one unit beyond;
- ordinary allocation/compilation failure rolls back; asynchronous
  interruption re-raises after discarding staging;
- C2 continues to reject its existing empty-D fixture; and
- APIs/selectors containing iid/family/layer/margin/verdict fail tests.

## Closure and promotion

S0-C3 closes with gain zero if graph bijection cannot be proven, the corrected
target has no grammar hit, exact HZ/frame/predicate/bias/witness semantics
differ, any whole-state/resource/concurrency gate fails, or any retained
baseline row regresses.  A translator repair that changes a verdict cannot be
booked as an HZ gain until the complete retained baseline is re-established.

Only one complete 2,413 replay under the same default-off candidate flag may
promote S0-C3.  Promotion requires all 1,870 baseline rows and all 13 family
counts retained, invalid ADV/SAT zero, and at least one new CERT or
independently concrete-network-validated ADV.  UNKNOWN, ERROR, theoretical
capacity, synthetic capability and speed alone never change the headline.

## Current status and no-claims

The set theorem and grammar are preregistered; the graph-faithfulness
prerequisite is not yet satisfied.  There is no production integration,
Tiny/CIFAR target run, same-structure shadow, speed result or formal/E0 gain.
Trial 9 continues under its existing source freeze and automatic exit sealer.
`/data1/Kane/HyZor` and all historical results remain read-only; new evidence
uses exclusive files only under this experiment directory.
