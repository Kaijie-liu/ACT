# S0-C1 Production Patch Map V1

Recorded on 2026-08-31 on branch `redu-hz` at repository commit
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This is a read-only mapping from
the frozen S0-C1 preregistration to the current production `SparseHZ` plumbing.
It authorizes no production edit while Trial 9 owns its nine-file source
freeze, creates no formal or E0 credit, and leaves the formal baseline exactly
1,870/2,413.

## Decision boundary

The hardened isolated V2 mathematical core is eligible for reuse by a future
candidate, but strict lineage proves that S0-C1 itself is zero-hit at its fixed
Tiny ReLU36 target. It is therefore not authorized for a production attempt.
The synthetic pure planner and adapter are not production evidence. In the
current real plumbing they would either miss every Scale/BN tail, lose the
registered ADD/reshape boundary, or prove only a local metric. Each item below
is a reusable prerequisite, not optional cleanup or permission to rescue C1.

The production attempt must preserve the exact nonconvex Hybrid-Zonotope
object: continuous and binary factors, equality and inequality predicates,
shared latent identity, frames, phase slots, concrete-witness reconstruction
and fail-closed behavior. It may replace only a proven-equivalent tuple of
linear source-map operators. It may not delete a source, a term or a predicate
row.

## Frozen rule and real operator order

`SparseHZAffineTerm.operators` is appended in source-to-output order and
`_lazy_materialize` consumes it in reverse by left composition. Consequently
the registered terminal tail remains:

```text
ImplicitConv(inner)
 -> one or more exact NCHW channel-stationary diagonals
 -> ImplicitConv(outer)
 -> zero or more exact output diagonals
```

The group-intersection coefficient formula in hardened V2 implements this
order exactly. Dense, ConvTranspose, pooling, permutation, nonlinear or
unknown operators are barriers. A resource rejection is also a barrier for
the candidate request and must re-enter the unchanged baseline materializer.

## Mandatory production changes

### 1. One production module, no experiment import

Add `act/back_end/hybridz_tf/composed_conv2d_stencil.py` containing the
hardened operator, a typed semantic Conv snapshot, strict diagonal
normalization, the real-term tail recognizer and a request-local transaction.
Production must not import code from `experiments/`.

The planner and compiler must use the same immutable kernel/mask snapshots and
typed binary keys. After compilation, the request key is rechecked before any
publication. The constructor-time `ImplicitConv2DOp.content_key` is not an
authority because its private payload buffers are currently writeable.

### 2. Recognize the actual Scale/BN representation

Production Scale and BatchNorm append SciPy CSR matrices, not
`DiagonalLinearOp`. The recognizer accepts a CSR middle/output operator only
after proving all of the following without a dense conversion:

- square shape equal to the flattened NCHW state;
- numeric finite data, canonical duplicate summation and no off-diagonal
  nonzero;
- an exact broadcast of one channel vector across every batch and spatial
  position; and
- the registered middle/output position in the tail.

Any non-diagonal entry, nonstationary value, unexpected shape or allocation
failure rejects the whole candidate request. Phase masks are not globally
converted to a full-length diagonal descriptor.

### 3. Make ADD/reshape lineage visible

The current term carries only `source` and `operators`; ADD concatenates terms
and reshape-like graph operations return the expression without a marker.
Thus a prohibited cross-boundary tail and a legal straight-line tail can have
the same operator tuple. Synthetic `Barrier` objects do not prove the real
rule.

Add `barrier_cuts: tuple[int, ...] | None` to `SparseHZAffineTerm`. `None`
means lineage was not established and rejects the whole candidate transaction;
new production terms explicitly start with `()`. A cut `c` denotes
`operators[:c] | boundary | operators[c:]`. Cuts are non-bool integers,
strictly increasing, unique and within `[0, len(operators)]`.

- a new materialized/nonlinear source starts with `()`;
- appending a linear operator preserves all cuts;
- ADD maps each outgoing term to a new term with its current operator length
  appended as a cut, then concatenates terms without mutating either input;
- reshape, flatten, squeeze and unsqueeze return a new expression whose terms
  append their current operator length; and
- repeated boundaries at the same operator position retain one cut.

A proposed core `[inner_index, outer_index]` crosses a boundary exactly when
some cut satisfies `inner_index < c <= outer_index`. The matcher rejects that
term before reservation. Cuts at or before the inner operator and cuts after
the outer operator do not cross the core. This distinction is necessary: a
single `composition_floor` is sound but incorrectly rejects a complete tail
that lies before a later ADD.

After replacing the core by one descriptor, retain cuts at or before the inner
index and shift every cut after the outer index left by
`outer_index - inner_index`. A cut inside the core cannot exist because the
request would already have rejected. Descriptor keys do not include lineage;
eligibility is proved per term before otherwise identical legal descriptors
may be interned.

The graph topology already proves that strict S0-C1 v1 has zero eligible
ReLU36 tails on Tiny iid143: Conv29 precedes ADD32, Conv33 is the only Conv
between ADD32 and ReLU36, and Trial 8 records no intervening lazy checkpoint.
Every two-Conv terminal core therefore has the ADD32 cut strictly inside it.
Trial 9 may provide the authoritative runtime tuple census but cannot change
this topology. S0-C1 v1 must close as a structural miss; allowing the outer
Conv to distribute across residual terms requires a separately preregistered
S0-C2 theorem and cannot be described as an S0-C1 interpretation.

### 4. Preserve expression, term, bias and predicate semantics

The real materialization is

```text
D_keep * W * source + expr.bias,
```

not `D_keep * (W * source + bias)`. Conv and BN biases have already been
propagated into the global expression bias. Middle/output zero diagonals or an
empty selected linear support therefore do not authorize masking, recomputing
or deleting bias.

On success, preserve `frame_id`, `n_out`, term order, every source object and
all prefix/output operator objects outside the replaced tail. Preserve the
original normalized one-dimensional float64 bias object. On rejection, return
the original `SparseHZAffineExpr` object itself; reconstructing an equivalent
dataclass is insufficient because its current `__post_init__` creates a new
bias view.

A term with zero linear output also remains present. Its source may be the
only carrier of equality or inequality predicates constraining factors shared
by another term. Baseline materialization still merges those predicates, and
dropping the term would relax the set.

### 5. Use one request-local, all-or-nothing transaction

For each real `_lazy_materialize` request:

1. take semantic snapshots and build a pure plan;
2. create a fresh local V2 transaction;
3. reserve and compile every unique descriptor sequentially;
4. reserve emission for the actual `keep_rows` support of this request;
5. validate exact materialization, identities and whole-state metrics; and
6. publish all rewritten cache aliases only after every check succeeds.

Failure at any point discards the local transaction and re-runs the original
expression through the unchanged baseline core. `CandidateV2Reject`,
`MemoryError` and candidate cap failures are normal fail-closed fallback
reasons. `KeyboardInterrupt` and `SystemExit` first roll back and then
re-raise. A failed request publishes no descriptor, arena reference, cache
alias, counter or provenance record.

Descriptor contraction may be interned by semantic content after success, but
emission permission is support-specific. Every materialization must reserve
against its current actual support, or the descriptor must have prepaid full
support. A small phase probe can never permanently authorize a later larger
support.

The hardened isolated V2 now commits descriptor contraction/resident state
only. Its emission count is explicitly prospective and deferred because it
does not construct a CSR artifact. A future materializer may waive a repeated
emission charge only by returning the identical retained artifact for the
same descriptor/support/reverse-prefix semantic key and counting its bytes and
nnz in physical state. A support key without an artifact is never a cache hit.

Do not retain a `RewriteDecision` or plan after success: it holds the original
terms/operators and would keep the supposedly replaced objects reachable.

### 6. Prove the whole reachable physical reduction

Move the shadow worker ledger into a production helper such as
`HybridzTF._sparse_reachable_numeric_ledger(extra_roots, substitutions)`, but
strengthen it before use. Traverse at least:

- `_sparse_hz_cache`, `_sparse_affine_expr_cache` and
  `_sparse_precomputed_relu`;
- top-level `_sparse_phase_output_bounds`;
- active expressions, all term sources, biases and operators;
- live successfully interned descriptors; and
- every candidate object held by the local transaction.

Deduplicate NumPy arrays, CSR buffers, HZ objects and operator payloads by
physical object identity. Physically distinct equal-content objects count
twice; shared predicate buffers count once. Support hypothetical root
substitution so before and after are measured at the same consumer/GC
boundary. The mandatory inequalities are:

```text
after.resident_bytes   < before.resident_bytes
after.resident_entries < before.resident_entries.
```

Logical expanded nnz, scalar work and local operator bytes remain diagnostics;
none substitutes for these inequalities. Numeric-buffer accounting also does
not substitute for measured peak RSS.

### 7. Configuration and stable provenance

Add a single default-false option such as `sparse_composed_conv_tail`. It is
active only when the lazy affine DAG and implicit Conv DAG prerequisites are
enabled. Mirror/reset it in `HybridzTF`; flag-off must execute zero candidate
planner/compiler calls and produce baseline-identical metadata.

Flag-on profiling records only stable values: schema/algorithm version,
accepted and rejected reason counts, semantic descriptor digests, support
digest/count, reserved and actual work, before/after resident entries/bytes,
strict deltas, compilation ledger, controlled transient and measured RSS.
Object `id`, Python `repr` and unstable hashes are forbidden in saved evidence.

## Focused promotion test matrix

All integration tests below use the real `SparseHZAffineExpr/Term` path:

1. Conv -> real CSR Scale/BN -> Conv, including a noncommuting two-channel
   oracle, multiple diagonals and optional output diagonal.
2. Exact `barrier_cuts` validation and remapping: cuts before the inner, after
   the outer and inside the core; repeated ADD/reshape cuts; malformed,
   missing and bool cuts; legal tails wholly before/after a boundary; and the
   real-shaped `Conv29 -> D -> ADD32 -> Conv33` zero-hit guard.
3. Same-source repeated terms, distinct-source same-frame terms and forty
   terms sharing one suffix; term/source/order identity remains exact and one
   unique descriptor is compiled once.
4. Nonzero inner, BN and outer biases; zero middle scale, zero output
   diagonal and zero outer-row mask; bias remains byte- and identity-preserved.
5. A zero-linear-output distinct source carrying unique `Ac/Ab/Auc/Aub/b/ub`
   rows; complete baseline/candidate predicates are identical.
6. Empty, partial and full keep sets, followed by a second materialization of
   the same descriptor with a different support; emission is charged for each
   new support.
7. Ordinary, aligned/misaligned group, depthwise, channel-multiplier,
   asymmetric stride/padding/dilation and batch-two operator algebra. Batch
   two remains an operator-only claim until real lazy Conv stops enforcing
   batch one.
8. General and selected left factors, result/cap boundaries and a monkeypatch
   proving full Conv CSR expansion is never attempted before a gate.
9. Failure during a second descriptor compile, cache publication and every
   reservation mutation; original expr/terms/bias/sources/cache/arena/ledger
   remain identity-equal.
10. `CandidateV2Reject`, `ValueError`, `MemoryError`, `KeyboardInterrupt` and
    `SystemExit` behavior at plan, reserve, compile, validate and publish.
11. Whole-state ledger coverage of shared buffers, phase-bound caches,
    precomputed objects, live arena objects, active expressions and a
    hypothetical root substitution.
12. Nontrivial `Gc/Gb/Ac/Ab/Auc/Aub` equality/inequality sources plus a
    concrete factor assignment or recovered FALSIFIED witness evaluated
    through baseline and candidate maps.
13. Default-off byte identity and zero invocation, then the registered real
    target, same-structure shadows, affected-family retained-set replay and
    finally all 2,413 formal rows.

## File-level patch map after Trial 9 custody closes

- `act/config/config.py`: one default-false option.
- `act/back_end/hybridz_tf/composed_conv2d_stencil.py`: hardened production
  operator, snapshot/diagonal normalizer, planner and transaction.
- `act/back_end/hybridz_tf/tf_cnn.py`: term lineage, real materialization
  wrapper and exact fallback; the old materializer remains the baseline core.
- `act/back_end/hybridz_tf/hybridz_tf.py`: whole-state ledger, live descriptor
  roots, transaction publication and reset/release cleanup.
- `act/back_end/verifier.py`: stable flag-on aggregate provenance only.
- focused production tests: every matrix item above before any benchmark.

`tf_mlp.py`, source HZ construction, nonlinear graph definitions, global
continuous/binary slots, predicate matrices and witness decoding remain out of
scope unless a focused preservation test reveals a genuine blocker. Such a
blocker pauses integration; it does not expand authorization silently.

## No-claim boundary

This map does not claim production readiness, a Tiny iid143 S0-C1 hit,
batch-two production support, a complete RSS bound, or any score gain. It does not treat
synthetic barriers, a per-descriptor resource pass, structural logical nnz or
an implicit-unfused speed comparison as Neural-HZ simplification. Only a
strict whole-state reduction followed by zero-regression family and full
2,413-row gates can change the formal 1,870 baseline.
