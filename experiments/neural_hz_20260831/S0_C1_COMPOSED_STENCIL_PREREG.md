# S0-C1 Pre-registration: Exact Channel-Contracted Conv Stencil

Status: design only, default-off, formal gain zero. This candidate may be
implemented only after the active Trial 9 process releases its source freeze.
It is governed by `GOAL_CHARTER.md` and targets the repeated residual-CNN
blocker measured at TinyImageNet ReLU36.

## Repeated blocker

`ImplicitConv2DOp` avoids the full resident Conv CSR, but its current Python
row-oracle composition still enumerates every two-Conv channel path. For two
`128 x 128`, `3 x 3` convolutions and 2,180 selected output rows, the second
composition has a conservative bound of roughly 2.893 billion scalar
products even though the final support has at most about 6.98 million nnz.
This is a structure-level overlap problem, not an iid condition.

## Exact real identity

Let the inner convolution be `A`, the outer convolution be `B`, and let every
operator between them be proved to be one batch- and spatially-stationary
per-channel scale `sigma_m`. For outer tap `t`, inner tap `q`, output channel
`o`, intermediate channel `m`, and input channel `c`, define

```text
H[t,q,o,c] = sum_m B[o,m,t] * sigma[m] * A[m,c,q].
```

The group-connectivity intersection restricts the sum over `m`. For an output
position `(b,o,h,w)`, the effective input offset is

```text
t * dilation_outer * stride_inner + q * dilation_inner,
```

the effective stride is `stride_outer * stride_inner`, and the origin is
`padding_outer * stride_inner + padding_inner`, independently in height and
width. Crucially, an outer tap contributes only when its intermediate spatial
coordinate is valid before the inner coordinate is tested. This condition
preserves the two original zero-padding boundaries. Consequently the operator
is a factored composed stencil, not an ordinary fused 5x5 Conv.

The identity changes only the evaluation order of the same affine map. Source
HZs, frame ids, continuous and binary slots, equalities, inequalities, global
neuron slots, and witness reconstruction are unchanged. Bias remains in the
existing affine-expression bias path and is propagated layer by layer.

## Admissible and rejected chains

Version 1 admits exactly:

- `ImplicitConv2DOp -> channel-stationary diagonal(s) -> ImplicitConv2DOp`;
- ordinary, grouped, or depthwise NCHW convolution;
- legal stride, padding, dilation, and independent batch blocks;
- a full-one intermediate row mask or a 0/1 mask proved constant for every
  spatial position and batch within each channel;
- any outer row mask; and
- a contiguous affine tail inside one residual term.

Version 1 rejects without changing the expression:

- any spatially varying phase or row mask between the convolutions;
- a diagonal that cannot be proved identical across spatial positions and
  batches for each channel;
- ReLU or another nonlinearity, pooling, ConvTranspose, Dense, an unknown
  permutation, or ADD inside the chain;
- incompatible layouts, batches, shapes, or group connectivity;
- non-finite payloads or index overflow; and
- any work, storage, transient-memory, or result-nnz bound violation.

Fusion never crosses ADD. Each affine residual term is planned separately and
the existing same-frame exact addition remains authoritative. A fusion reject
falls back to the separately capped support-sliced composition; if that also
rejects, the existing fail-closed path returns UNKNOWN.

## Uniform trigger and frozen candidate-v1 budgets

The planner may inspect only operator types/content, exact geometry, masks,
and resource bounds. It cannot inspect benchmark/family/model names, iid,
property, margin, or a historical verdict.

Before allocation it computes conservative bounds for unfused path products,
channel-contraction products, row-emission contributions, fused resident
storage, selected result nnz, and the cumulative `_lazy_materialize` transient
peak. Candidate v1 triggers only when all conditions hold:

- `contraction_products <= 200,000,000` per unique fused descriptor;
- cumulative `contraction_products + emission_contributions <= 256,000,000`
  per `_lazy_materialize` transaction;
- fused coefficient entries `<= 2,000,000`;
- fused descriptor resident payload `<= 64 MiB`;
- controlled transient payload `<= 1 GiB`;
- materialized result `<= 64,000,000 nnz`;
- fused work is at most one quarter of the conservative unfused path-product
  bound; and
- the registered whole reachable-state physical metric strictly decreases
  relative to the candidate's unfused phase-selective baseline.

Budgets are cumulative across terms. Content-identical fused descriptors are
interned and compiled once; they do not receive a fresh work budget per term.
These values are frozen for candidate v1 before its first real-network result.
A synthetic test may close v1, but cannot be used to retune it in place.

For the measured Tiny36 shape, the estimated unique contraction work is about
169.9 million products, row emission about 22.5 million contributions, and
resident coefficient upper bound 1,327,104 float64 values (about 10.6 MiB), so
the structure is inside the pre-registered v1 gate. The real run must record
the actual bounds and may still reject.

## Minimal implementation surface

Add a default-off `ComposedConv2DStencilOp` to `exact_linear_op.py` with:

```python
try_build(inner, middle_ops, outer, budget)
matvec(vector)
gather_rows(rows, max_nnz, budget)
left_compose(Q, max_nnz, budget)
```

It exposes the existing operator protocol plus stable `content_key`, logical
expanded nnz, resident entries, and resident bytes. `_lazy_materialize` runs a
pure tail planner before reverse-prefix counting, replacing only admissible
Conv/diagonal/Conv tails with an interned descriptor. Monomial `Q` performs
direct selected-row scaling. General sparse `Q` uses strictly support-sliced
temporary CSR and never calls either Conv's full `to_csr_reference`.

The planner is initially local to materialization. It does not change ReLU
construction, source HZs, affine bias propagation, residual grouping, terminal
solving, or witness decoding.

## Proof and regression gate

Focused tests must compare the fused result with
`outer_CSR @ middle_diagonal @ inner_CSR` on small oracles and cover:

- interior, every edge and corner, odd/even spatial sizes;
- asymmetric stride, padding, and dilation;
- groups, depthwise, and channel multipliers;
- positive, negative, and zero channel scales;
- arbitrary outer masks and all admissible intermediate masks;
- batch size two;
- nonzero bias through the existing bias path;
- residual multi-term expressions with a shared source/frame;
- monomial and general sparse left factors;
- exact preservation of factor widths, predicates, slots, and witness maps;
- each nnz/work/resident/transient rejection boundary;
- every inadmissible spatial mask/operator chain above;
- default-off byte identity; and
- a monkeypatch that makes full Conv CSR expansion raise immediately.

Dyadic oracle cases require element-wise identity. Other finite binary64 cases
must be deterministic and satisfy the repository's registered numerical
equivalence criterion. The real-algebra identity is the mathematical claim;
the PLDI soundness audit must separately state the floating-point evaluation
model and cannot infer bitwise equivalence from a tolerance test.

After focused tests, the fixed expansion order is Tiny iid143 ReLU36, ReLU63,
ReLU71, and terminal; the registered CIFAR targets and S6 guards; SRI-A,
SRI-B, CIFAR2020, and Collins RUL; then the formal Conv cohort. Any failed
level closes candidate v1 before broader replay. No intermediate stop changes
the formal 1,870 score or an external ledger.

## Required result ledger

Each attempted fusion records original/fused product bounds, contraction and
emission work, avoided products, logical nnz, coefficient/index/mask resident
bytes, temporary CSR bytes, cumulative transient peak, final nnz, geometry,
group mode, selected/masked rows, unique-object accounting, and a stable
trigger or rejection reason.

The fused stencil can use more resident bytes than the two raw implicit
kernels. It may therefore claim exact work compression but cannot alone claim
HZ simplification. That claim requires the whole registered reachable physical
ledger, including HZ values/predicates, operators, biases, phase metadata and
caches, to show the strict reduction required by the charter.
