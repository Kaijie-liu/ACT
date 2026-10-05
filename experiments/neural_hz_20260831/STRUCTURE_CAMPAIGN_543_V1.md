# Formal 543-Case Structure Campaign

Locked on 2026-08-31 for the active `redu-hz` Neural-HZ goal. This is a target
map, not a score promotion. The only formal baseline remains **1,870/2,413**:
1,063 CERT plus 807 concretely validated ADV, with 269 UNKNOWN and 274 TIMEOUT.
Every old solved row and every per-family solved count is a non-regression
constraint.

The machine-readable authority is
`manifests/formal_unsolved_structure_manifest_v1.json`:

- file SHA-256:
  `393590e26d3ae4edd50c9a7d2df48980c8b3701237b61c030bfbe4ad36947d8d`;
- payload SHA-256:
  `dd7fd5e63b0b5a4a1b72911424326f338831b07eff0fb64192e18ab82c9dbef9`;
- generator SHA-256:
  `33a53a00848a970251fabe192c6532aea1cd7f5d0dec05bef84fa823a34eb52e`;
- test SHA-256:
  `64c27d1dd043096f2bdb71e9ae69653bb955bee0d713ca9770fb8268dea34cd3`.

The artifact deterministically rebuilds 2,413 composite-authority rows, removes
the 1,870 solved rows, and assigns every remaining row to exactly one cohort
using its ONNX operator structure. It records model/spec hashes, operator
signatures, authority row identity and the 13 benchmark manifests. Family,
model identity, iid, property and verdict are forbidden selector inputs for a
runtime Neural-HZ rule; they appear here only to freeze reporting and replay
coverage.

## Exhaustive partition

| Cohort | Repeated structure | UNKNOWN | TIMEOUT | Remaining upper bound |
|---|---|---:|---:|---:|
| A | Conv/ConvTranspose--ReLU sparse frontier | 79 | 49 | 128 |
| B | TLL symmetric/signed ReLU | 15 | 0 | 15 |
| C | plain dense FC--ReLU | 95 | 168 | 263 |
| D | shared-ancestor Add/Concat/skip | 19 | 2 | 21 |
| E | attention plus residual | 59 | 53 | 112 |
| F | smooth nonlinear tail | 2 | 2 | 4 |
| **Total** | six mutually exclusive structures | **269** | **274** | **543** |

These values are search-space ceilings, not candidate gains, and must never be
summed into the formal score before one promoted full replay. The aspirational
endpoint is 2,413/2,413, reached through multiple independently gated
structural innovations rather than an iid menu.

## A -- phase-sliced implicit Conv HZ (active)

For interval-certified negative, positive and unstable rows `N/P/U`, use the
exact identity

```text
ReLU(z)_N = 0
ReLU(z)_P = z_P
ReLU(z)_U = the existing exact nonconvex HZ ReLU graph on z_U
```

Stable-positive rows stay in an exact affine operator DAG; only the unstable
frontier receives continuous auxiliaries, binary phase factors and predicates.
Conv is represented by kernel, geometry and row mask, and admissible consecutive
Conv tails use the boundary-correct channel-contracted stencil. Residual terms
retain one shared ancestor/frame and are never split into independent convex
copies.

The current `ImplicitConv2D` direct formal reach is **124**: MetaRoom 5 plus
ReluSplitter-CNN 119. The four cGAN large-image rows use ConvTranspose and are a
separate exact-operator extension; they cannot be claimed by the current S0-C1
rule.

The fixed advancement order is:

1. TinyImageNet/CIFAR100 residual scouts and S6 guards;
2. residual/plain-CNN same-structure shadows;
3. MetaRoom 6cnn's five formal TIMEOUT rows;
4. ReluSplitter OVAL 39 and biasfield 80;
5. only after a separate preregistration, cGAN ConvTranspose 4.

The zero-regression set includes all solved rows in those structures, Malware
Conv, the external E0 61 validated ADV, and ultimately the complete 1,870.
Composed stencil alone is only work compression relative to Trial 9; the
combined S0 candidate must pass the whole-state strict physical reduction in
`S0_C1_COMPARATOR_AMENDMENT_20260831.md`.

## B -- TLL signed sharing and dead exact graphs

Byte-exact `x/-x` rows in one frame use

```text
ReLU(-x) = ReLU(x) - x
```

to share one binary ReLU graph. When the unique Dense successor has an
exact-real zero nonlinear coefficient for a private signed class, its private
continuous lift and three local inequalities may be removed while the affine
row and witness mapping remain.

The frozen width-adaptive rule already moves TLL from 17/32 to a reproduced
candidate 29/32, **+12**, with zero invalid ADV. Its remaining gates are
representative non-TLL shadows, four-concurrent no-regression and one complete
2,413 replay. If all pass, the nearest credible formal node is 1,882/2,413;
until then the headline remains 1,870.

## C -- Dense successor-folded phase frontier

This cohort covers Cora 140, ACAS Xu 66, ReluSplitter-FC 56 and SafeNLP 1.
Trial 1/2/3 are not revived: inactive-factor projection slowed retained proofs,
predicate-implied phase fixing had zero hits, and dense ReLU quotienting caused
fill/regressions.

The new structure constructs only the `U` nonlinear graph, carries `P` as an
exact row-masked Dense operator, folds both once into the next Dense consumer,
and garbage-collects an ancestor only after its final consumer. It retains the
existing equality-friendly ReLU graph for `U`; no binary factor is relaxed or
pivoted.

The target order is Cora's repeated 8-Dense/7-ReLU set subgroup, the remaining
Cora point/trades networks, ACAS Xu, ReluSplitter-FC and finally the singleton
SafeNLP UNKNOWN. Same-structure retained guards include SafeNLP, ACAS, Cora,
ReluSplitter-FC, SAT-ReLU and linear Malware rows.

## D -- same-frame branch quotient

LinearizeNN is correctly classified here, not as Conv: its `AllInOne` models
repeat `Gemm/ReLU + MatMul + dynamic Concat`. Cersyve supplies same-frame Add
shadows. The exact identities are

```text
W concat(u,v) = W_u u + W_v v
(T1 s + d1) + (T2 s + d2) = (T1 + T2) s + d1 + d2.
```

The unique affine successor is folded through Add/Concat while the common
source HZ and predicate frame are stored once. The rule never crosses ReLU and
never garbage-collects a branch with another live consumer. It first targets
LinearizeNN 20 and treats Cersyve 1 as a guard/transfer check.

## E -- attention common-mode quotient

ViT/cGAN attention is not an ordinary residual variant. A first exact target is

```text
softmax(alpha * 1 + r) = softmax(r).
```

Only a byte/exact-real common affine component shared along one Softmax axis may
be removed from that Softmax consumer. The upstream latent remains live for
all other consumers and witness reconstruction. The 85/100 solved rows of the
ViT IBP model form the first guard before PGD-model or cGAN transfer. Historical
numeric Softmax fallbacks that regressed a CERT stay closed.

## F -- signed smooth-graph sharing

For same-frame byte-exact duplicate or negated preactivations:

```text
sigmoid(-x) = 1 - sigmoid(x)
tanh(-x) = -tanh(x).
```

One existing nonconvex smooth HZ graph is shared and other outputs are rebuilt
affinely. DistShift's repeated Sigmoid structure is checked first; if no exact
signed hit exists, this version closes rather than loosening equality for two
cGAN singletons. All 70 solved DistShift rows, K3 active-column witness
semantics and the 13 solved cGAN rows are guards.

## Universal transaction and promotion gate

Each cohort receives one default-off candidate and must pass, in order:

1. exact identity or explicitly sound abstraction tests plus reverse witness
   reconstruction;
2. preservation of continuous factors, binary phases, predicates, shared frame
   identity and fail-closed behavior;
3. strict reduction of the registered whole reachable physical state and
   registered total nnz, with work/transient/RSS ledgers;
4. target, zero-hit guards and multiple same-structure shadows;
5. full affected-family replay and four-concurrent
   `baseline_time/candidate_time >= 1.0`;
6. one source manifest, one configuration and one path over all 2,413 cases;
7. retention of all 1,063 CERT, all 807 validated ADV and every family solved
   count, invalid ADV zero, plus at least one new solved row.

Any failure closes that candidate version without changing thresholds or
rewriting old results. The ordered campaign is **A, then B, C, D, E, F**;
only the active structure changes production code at a time.
