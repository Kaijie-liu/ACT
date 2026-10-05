# S0-C2 Residual-Distributive Composed Stencil Preregistration V1

Preregistered on 2026-08-31 on branch `redu-hz` at repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`, before any S0-C2 benchmark run
or production integration. This is a new candidate version, not an
interpretation, amendment or threshold change for S0-C1. S0-C1 remains bound
by its graph-level rule that fusion never crosses ADD and is a structural miss
at Tiny iid143 ReLU36.

The formal baseline remains exactly 1,870/2,413: 1,063 CERT plus 807
concretely validated ADV, with every solved count in all 13 families a hard
non-regression constraint. E0 remains 61/400 independently replayed historical
ADV with zero Neural-HZ gain credit. This document creates no score.

## Exact set theorem

All terms in a production `SparseHZAffineExpr` share one exact latent frame.
After the existing same-frame column alignment, source `i` denotes

```text
s_i(xi,z) = c_i + Gc_i xi + Gb_i z

Ac_i xi + Ab_i z = b_i
Auc_i xi + Aub_i z <= ub_i
xi in [-1,1]^nc, z in {-1,1}^nb.
```

The delayed residual expression is

```text
E(xi,z) = sum_i T_i s_i(xi,z) + d,
P = conjunction_i P_i.
```

For one shared outer linear map `B`,

```text
B E = sum_i (B T_i) s_i + B d.
```

If the outer graph operation has bias `beta`, the sole global bias is
`B d + beta`. It is never duplicated once per residual term. This is equality
over the same continuous and binary factors, not a relaxation:

- `Gc/Gb` factor columns and phase-slot identities are unchanged;
- `Ac/Ab/b/Auc/Aub/ub`, `frame_id` and `exact` are unchanged;
- every original source and term remains present and ordered;
- equal-valued distinct sources are never merged by value;
- the same `(xi,z)` is a concrete factor witness before and after; and
- any FALSIFIED result still requires independent concrete network/VNNLIB
  replay before it can count.

`keep_rows` masks only the linear maps. The already propagated expression bias
is not masked. A term whose selected linear output is zero is retained because
its source may uniquely carry predicates constraining factors used elsewhere.

The algebra is an exact real-number theorem. Non-dyadic binary64 reassociation
is not claimed bitwise equal; deterministic evaluation order, dyadic equality
oracles, finite checks and the existing soundness boundary remain mandatory.

## One structural rule

S0-C2-v1 uses exactly one branchwise schedule: distribute the newest shared
outer Conv through exactly one latest common residual ADD, compose every
eligible nonzero-support branch, preserve recognized identity skips, and
reject the entire request if any other branch is structurally ambiguous.
There is no runtime menu between S0-C1, C2, checkpointing or partial subsets.

Each term carries immutable graph lineage:

```text
BoundaryMark(
    kind,              # ADD, RESHAPE, FLATTEN, SQUEEZE, UNSQUEEZE, ...
    occurrence_key,    # equality-only token for one graph event
    operator_index,
)
```

Lineage `None`, malformed marks or unstable occurrence identity reject the
whole request before planning. Marks are ordered by graph occurrence and
operator position. The numeric value of an occurrence token is never a
selector; only exact equality proves that terms passed through the same ADD.

The sole accepted non-identity branch shape is:

```text
prefix_i
 -> ImplicitConv(inner_i)
 -> one or more exact channel-stationary diagonals
 -> latest common ADD mark
 -> zero or more exact common channel-stationary diagonals
 -> one shared ImplicitConv(outer)
 -> zero or more exact output diagonals.
```

The proposed core must contain exactly one ADD mark, and that mark must be the
latest mark for every participating term with the same occurrence key. Any
other ADD, reshape-family, nonlinear, pooling, Dense, ConvTranspose,
permutation or unknown boundary/operator inside the core rejects the entire
transaction. The shared outer Conv and post-ADD suffix must have identical
current semantic snapshots, not merely stale constructor keys.

An identity skip is permitted only when its current segment before the common
ADD contains no operator; its shared outer map is kept unfused. A nonempty
branch that lacks the complete registered inner-Conv/diagonal chain rejects
the request. Zero-support terms remain byte-for-byte unchanged. Every
structurally eligible nonzero-support branch is selected; budgets cannot be
used to choose a favorable subset.

Triggering may inspect only operator types/shapes/current payloads, exact
diagonal stationarity, boundary kind/equality, support and frozen resource
metrics. Benchmark, family, iid, property, layer number, margin, historical
verdict and elapsed behavior are forbidden selector inputs.

## Atomic implementation boundary

S0-C2 uses a fresh request-local staging transaction for each real
materialization:

1. snapshot the expression, bias, term/source order, all value/predicate
   buffers, frame/exact bits, operators, lineage and actual support;
2. determine the unique full eligible branch set and stable semantic requests;
3. compile every unique descriptor in deterministic content order;
4. at the real materializer, reserve emission around the actual
   `gather_rows/left_compose` artifact for the current support and reverse
   prefix;
5. either cache and return the same CSR artifact, counting its bytes/nnz in
   physical state, or charge every execution; a key without an artifact never
   waives emission work;
6. materialize the entire shadow expression and compare complete HZ/frame/
   predicate/bias/witness semantics;
7. compare before/after reachable physical state at the same consumer/GC
   boundary using hypothetical root substitution; and
8. publish every alias, descriptor, artifact and stable counter together only
   after all checks pass.

Normal structural/resource rejection and ordinary `Exception` return the
original production expression object and run the unchanged capped baseline
materializer. `KeyboardInterrupt` and `SystemExit` discard staging state and
re-raise. No plan retaining old terms may survive success. Failure of the
baseline fallback remains UNKNOWN, never a candidate gain.

The current hardened V2 transaction is descriptor-only. Its emission number
is prospective and explicitly deferred. It is not S0-C2 materialization
evidence until step 4 above exists and is tested.

## Frozen resource and representation gates

Every unique descriptor and the complete request retain the S0-C1 numeric
limits without adjustment:

- descriptor contraction products at most 200,000,000;
- complete contraction plus actual emission at most 256,000,000;
- coefficient entries at most 2,000,000;
- descriptor resident payload at most 64 MiB;
- controlled numeric transient at most 1 GiB;
- each materialized result at most 64,000,000 nnz; and
- fused work at most one quarter of the registered unfused work.

Additionally, identity-deduplicated whole reachable state must satisfy both:

```text
after.resident_bytes   < before.resident_bytes
after.resident_entries < before.resident_entries.
```

The ledger includes HZ value maps, all predicate matrices/vectors, biases,
phase bounds/metadata, active expressions, operators, descriptor arena,
precomputed/cache roots, transaction objects and any still-reachable old
expression. Logical expanded nnz and work ratios are diagnostics only. Numeric
buffer accounting is reported separately from measured peak RSS. Four
concurrent capability replay must satisfy `baseline_time/candidate_time >= 1`.

## Fixed target and advancement order

The first and only target before any shadow is TinyImageNet iid143 at ReLU36.
The content-addressed Trial 8 evidence records:

- `Conv29/Scale30/Bias31 -> ADD32 -> Conv33/Scale34/Bias35 -> ReLU36`;
- four lazy terms at ADD32 and no lazy checkpoint;
- 25,088 ReLU rows with `N/P/U = 20,648/2,268/2,172`;
- exactly 2,180 phase-probe rows (`2,172 U + first 8 P`); and
- formal verdict UNKNOWN and Neural-HZ gain zero.

The earlier per-descriptor arithmetic preflight is only a ceiling:

```text
unfused products        2,445,737,984
contraction               169,869,312
prospective emission       19,107,328
prospective total         188,976,640
descriptor resident        10,616,920 bytes.
```

It does not prove real matches, cumulative branches, actual emission or
whole-state reduction. Advancement is fixed:

1. pure lineage and exact-HZ theorem tests;
2. Tiny iid143 ReLU36 only;
3. Tiny iid143 ReLU63 only after ReLU36 passes;
4. registered same-structure Tiny/CIFAR residual shadows and zero-hit guards;
5. affected-family retained-set replay with four-concurrent non-regression;
6. all 13 families and all 2,413 rows under one flag/config/path.

No direct jump to ReLU71, terminal verification or another iid is permitted
after a ReLU36 failure.

## Immediate closure conditions

S0-C2-v1 closes without changing its rule, budgets or target if any occurs:

- real lineage yields zero cross-one-common-ADD hit at ReLU36;
- a hit requires two ADDs, reshape/nonlinear crossing, an unknown boundary or
  an unregistered partial branch subset;
- source, term, bias, frame, continuous/binary width, phase slot, predicate or
  concrete-witness semantics differ;
- any descriptor, transaction, nnz, transient, RSS, whole-state or concurrency
  gate fails;
- emission is waived without reusing the identical retained CSR artifact;
- the fallback produces ERROR or an invalid witness; or
- any one of the frozen 1,870 results or any family solved count regresses.

Only a complete 2,413 replay retaining all 1,870 baseline rows, invalid ADV
zero, and adding at least one new CERT or independently validated ADV can
change the formal score or default-enable S0-C2.

## Current status and no-claims

This preregistration authorizes only isolated math/lineage work while Trial 9
owns the production source freeze. It does not claim that production lineage,
real emission, batch-two plumbing, whole-state reduction, a Tiny iid143 hit,
speedup or any formal/E0 gain exists. `/data1/Kane/HyZor` and all historical
results remain read-only; new evidence uses exclusive files only under this
experiment directory.
