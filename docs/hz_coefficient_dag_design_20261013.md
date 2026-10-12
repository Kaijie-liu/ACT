# Frozen coefficient-reference support component

2026-10-13; parent `0ed08922abe894fe94e189e630d5bf393004b75c`.
This is an opt-in given-HZ representation/support component, not a new real
experiment, source-complete proof, solver search or device attempt. The root is
the only writer. Existing Store/source schemas and limits remain unchanged.

## Decision and finite scope

The fixed tensor fixture's dense first layer has 786432 weight entries. Input,
router and that layer require at least 795648 stored nonzeros. Its following
ReLU repeats each dense vector in either a stable output or an unstable equality
(this fixture has no inactive rows), increasing the lower bound to 1582080,
above the old Store's 1048576 limit. Byte paging or a rowwise affine call alone
cannot fix this. Old 64-row pages can also exceed 1 MiB, and the old affine
checker performs at least 11796480 counted visits on that layer. None of these
limits is raised here.

Instead, test a coefficient-vector DAG with exact adjoint support. A node is
`local + sum(scale * earlier coefficient node)`. Row kind, name, domain and rhs
remain independent root records. In particular an unstable ReLU equality uses
the negative parent coefficient vector but rhs `parent_center - upper/2`:
scaling an entire affine row would be wrong.

This component represents one supplied support domain. It does not enumerate
MoE routes or discharge all output properties. Input/source lowering, complete
captured-source integration, convolution, request supervision and GPU remain
separate work. A large given-row control is not the earlier captured source.

## Mathematical contract

For topologically ordered coefficient nodes,

\[
C_v=L_v+\sum_{u<v}a_{vu}C_u.
\]

For objective roots inject their property weights; for equality/inequality
roots inject minus their supplied dual multipliers. Reverse topological
accumulation `alpha[u] += a[v,u] * alpha[v]` and
`r += alpha[v] * L[v]` gives exactly the expanded residual
`c - A.T*y - E.T*z`, by substitution in descending order. Each local vector is
consumed at most once in this accumulation; no expanded-vector cache is used.

The scalar constant is computed only from the original roots: objective
constant plus `b.T*y + h.T*z`. It is never propagated along coefficient edges.
For `y <= 0`, free `z`, and all continuous/relaxed binary factors in [-1,1],
`constant - sum(abs(r))` is a valid lower bound. Absolute values are taken only
after combining all contributions to each global factor. Different experts'
private factors cannot cancel by accidental local-column renumbering.

The claim is exact representation and same-candidate lower-bound equivalence,
not a tighter domain, optimizer improvement, novelty priority or faster full
verification. Constraints and roots cannot be deduplicated just because they
share a node. Binary factors are explicitly relaxed, not claimed exactly solved.

## Identity, scope and resources

Bind externally expected graph, request and original-given-row identities.
Graphs have one frame, immutable factor registry and explicit domain ancestry;
private owners must be visible, guards cannot be weakened, and node references
must target the same/ancestor domain. All coefficients, including zero-weight
subgraphs, are checked before accepting a candidate. Canonical nonzero sorted
local entries and references, earlier references, unique roots/nodes, complete
root/query/dual inventories, exact rhs, rational bit limits and deadlines are
mandatory. Hash binding alone does not prove correspondence to a native HZ.

For actual SparseHZono controls an independent rowwise bridge compares every
original output/constraint coefficient and rhs, factor map and frame. It expands
one row at a time, not a persistent expanded matrix. This bridge is charged and
is not a network-to-HZ proof. The exact support checker uses only the standard
library; native intake is a separate optional module.

Reuse the old primitive limits: 32768 factors, 16384 active factors, 128 domains,
64 reference/ancestry depth, 65536 nodes/roots, 1048576 stored local nonzeros,
4096/8192 stored/intermediate rational bits, cumulative 8000000 counted work and
one 300-second deadline. Add at most 65536 reference edges. Graph bytes are
bounded at 128 MiB; this finite component is not the old paged Store. No reset
per node, row or stage; no automatic EndpointWork 96M substitution.

## Fixed controls and acceptance

1. Small exact controls: nonzero-center ReLU equality; active alias; diamond
   reference; same node/different rhs roots; shared cancellation versus distinct
   private factors; all three unordered pairs of three synthetic experts; both
   continuous and binary columns; zero-dual nodes still checked. Compare every
   residual and constant to an independent expanded-row calculation.
2. Actual small SparseHZono: output/constraint shared-row references, explicit
   sparse correction, nonzero centers, private/continuous/binary factor mapping;
   full native bridge and same-candidate differential. No native solve.
3. Fixed dense shape: 256 vectors of width 3072 using the already registered
   tensor fixture's dyadic weight formula with salt 2; each is reused negatively
   by a second root with a different rhs. Consume all coefficients, compute an
   independently streamed expanded residual for a fixed nonzero candidate and
   a cancelling candidate, and record storage/work/cost separately. This is
   a given-row duplication control, not full source lowering or a positive model.
4. Refusal controls: changed identities/rhs/property/scale, omission/duplication,
   forward/cyclic refs, scope/frame/factor alias pollution, guard promotion,
   wrong dual sign/count, noncanonical numbers, bit/work limits and expiration.
   Preserve all failed attempts. Mutations with recomputed graph hash must still
   fail the original-row bridge if they change the represented native object.
5. Relocated `python -I -S` small package check without native imports. This checks
   given rows, not a portable full network certificate.

Each subprocess has a 300-second hard cap; no data/checkpoint/GPU use. The one
large-shape control runs once in a new directory. Whole stage archive cap is
64 MiB, no silent overwrite; unit failures can be repaired with recorded reruns,
not by changing these mathematical fixtures or limits. Record generation,
validation, exact accumulation, independent expanded reference, serialization
and outer elapsed cost; do not call a component timing end-to-end acceleration.
Stop after controls, review and archive. Source integration is the next decision,
not automatic real-model admission. All G1–G6 remain OPEN.
