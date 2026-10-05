# S0-C1 Production Integration Audit

Read-only audit completed on 2026-08-31 while Trial 9 PID 251298 held the
production-source freeze. No production file was modified for this audit. This
record refines implementation requirements without changing the frozen v1
budgets or comparator amendment.

## Minimal integration surface

Production integration is limited to:

- a default-off `sparse_composed_conv_stencil` configuration flag and its
  HybridZTransformer mirror/reset/profile fields;
- an exact `ComposedConv2DStencilOp` in `exact_linear_op.py` implementing the
  existing operator protocol;
- a pure terminal-tail planner before reverse-prefix accounting in
  `_lazy_materialize`; and
- worker provenance/resource ledgers plus focused tests.

`tf_mlp.py`, ReLU graph construction, source HZs, global slots, predicate
matrices and witness decoding must remain unchanged. If implicit Conv is not
enabled, the new flag records `implicit_conv_prerequisite_disabled` and leaves
the expression unchanged.

The v1 trigger recognizes only, within one affine residual term:

```text
ImplicitConv(inner)
 -> one or more exact batch/spatial-stationary channel diagonals
 -> ImplicitConv(outer)
 -> optional output diagonals
```

It never crosses ADD, ReLU/phase masks, pooling, reshape/permutation, Dense,
ConvTranspose or an unknown operator. The inner mask is full-one or exactly
channel-stationary 0/1; the outer row mask may be arbitrary. Geometry, group
connectivity, shape and all frozen resource gates must be proved before
allocation. A normal reject returns the original term tuple byte-for-byte and
charges no committed budget.

## Required corrections to the isolated prototype

The isolated prototype is mathematically correct on its covered tests, but it
cannot be copied into production unchanged.

### Group-aware contraction work

The prototype compiles each tap pair with dense
`(Cout x Cmid) @ (Cmid x Cin)` multiplication while charging only group-valid
products. For inner/outer group counts `gA,gB`, this can execute approximately
`gA*gB` times the charged work.

Production must contract only nonempty intersections of an outer input-channel
block with an inner output-channel block. For each intersection `J`, outer
channel block `O`, input channel block `I` and tap pair `(t,q)`, it computes

```text
H[t,q,O,I] = sum over m in J of
              B[O,m,t] * sigma[m] * A[m,I,q].
```

Ascending-middle-channel rank-one accumulation is the canonical v1 evaluation
order. It supports misaligned groups, depthwise and channel multipliers while
executing exactly the group-valid product count charged by the gate. The
floating-point claim remains real-algebra equivalence under a frozen runtime;
NumPy/SciPy/BLAS versions and threading are recorded rather than inferring
bitwise equivalence from an `allclose` test.

### Transaction accounting

The prototype validates but does not consume
`transaction_contraction_products`. Production uses one explicit transaction
with:

```text
compiled_content_keys
contraction_used
emission_used
transient_live_bytes / transient_peak_bytes
cached_csr_nnz
```

Contraction is charged once per unique descriptor; emission is reused only for
the same selected support/reverse prefix. Descriptor contraction remains at
most 200M and committed contraction plus emission remains at most 256M. A
rejected reservation cannot consume budget.

### Stable identity and exact metrics

Production content identity is a typed tuple containing the inner and outer
operator content keys, the binary digest of the stationary diagonal and the
group-intersection algorithm version. Python `repr` is forbidden. A selected
support digest is included only if the descriptor itself is support-specific.

`logical_expanded_nnz` is the exact structural count after colliding identical
spatial offsets, not the prototype's emission upper bound. The ledger keeps
four distinct values:

- logical path products;
- emission contributions;
- exact logical expanded nnz after offset collision; and
- actual materialized CSR nnz after numerical zero elimination.

Resident coefficient, pair-index, indptr and mask buffers are recorded
separately. The worker records operator kind, accepted/rejected reasons,
per-layer live-cache peak, controlled transient peak, process `ru_maxrss` and
environment versions.

## Semantic preservation gate

The planner replaces only the linear source-map chain at materialization.
`expr.bias` remains authoritative because Conv/BN bias has already been
propagated layer by layer. Every term retains the same source object and frame;
continuous/binary widths, equality/inequality predicates, global phase slots,
residual ancestry and reverse witness reconstruction remain unchanged.

Focused production tests must cover ordinary/group/depthwise/channel-multiplier
geometry, batch two, asymmetric stride/padding/dilation, every boundary,
channel and outer masks, multiple/zero/negative scales, bias, residual
multi-terms, selected/general left factors, descriptor interning, cumulative
budget boundaries and every fail-closed reject. They must monkeypatch full Conv
CSR expansion to raise, prove default-off byte identity, compare complete
SparseHZ predicates/frames/slots and replay a synthetic concrete witness.

Composed stencil alone may claim exact work compression relative to Trial 9;
it cannot claim a representation reduction. The combined S0 candidate advances
only when the whole-state metric defined in
`S0_C1_COMPARATOR_AMENDMENT_20260831.md` is strictly smaller than the registered
expanded phase-selective comparator and all existing verdict/witness gates are
preserved.
