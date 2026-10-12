# Scaled source and independently checked batched residual controls

Frozen before implementation. Parent: `0b6b30de56b40717f30b5c6f22a7959b786bffa1`.
This admits a joint representation/consumer implementation and finite correctness
controls, **not the dense source, a trained model, CUDA or a performance study**.
It follows the [scale-only refusal](hz_factor_scale_capacity_20261013.md).

## Complete capacity design

The unchanged fixed dense recipe has n=3072 inputs, h=256 first-hidden outputs,
u=99 unstable rows, m=260 first-affine rows, N=798720 first-bank weights and
K=512 later-affine weights. Analytic classification and source identity are in
the previous report; no new source computation was used to obtain them.

Conservative inventory: F=n+5u+4=3571 factors, V=n+m+4u+2=3730 nodes,
L=n+N+10u+4=802786 local entries, R=u+K=611 references, H=11 views,
P=6n+2h+6+3u=19247 root records and G<=V nonempty local scale groups.
The first-layer input centers are zero; arbitrary clipped/nonzero inputs do not
inherit this capacity admission argument. ReLU and final-affine compensation
remain included, not assumed zero.

Source construction plus its one independent check has the following bound:

| Work category | Formula | Counts |
|---|---|---:|
| Row visits | 2F+2V+2H+P+2h | 34,383 |
| Coefficient visits | 2N+2K+L+R | 2,401,861 |
| Explicit scale parsing | F | 3,571 |
| Exact calls | 2L+4G+2R+2n+2(m+2)+4K+4h+14u | 1,632,840 |
| Source subtotal | | **4,072,655** |

These are the existing Work call definitions, not a new way of counting FLOPs.
Tensor reading, hashing, JSON conversion, identity comparisons and frozen
handle creation retain deadline checks and are fully charged to wall time,
even where the existing policy has no corresponding Work increment.

The consumer design's bound is 3,890,655; the combined count is **7,963,310**,
leaving 36,690 below 8M. Its accounting separates the three large banks
(native 798720, objective 1465344, dual 1520640) from native preparation 11978,
gate/plan 28337, native roster 613, iterations 386, candidate export 598,
objective overhead 9622, dual overhead 11460, residual merge/weighted support
42882 and receipt/terminal metadata 75. These bounds constrain implementation;
they are not measured counts, timing guarantees, or dense execution approval.
Actual call sites must be checked against this ledger before a separate dense
execution freeze. All remaining failures retain the unchanged 300s/8M limits.

For the consumer, S=(157+2)n=488448 is the two endpoints' first-bank
objective total, T=un=304128 the dual bank, M=3u=297 constraint rows and
g<=2 the full-factor scale groups. Its supplementary formulas are:

| Consumer term | Formula / inventory | Counts |
|---|---|---:|
| Large banks | N+3S+5T | 3,784,704 |
| Native remainder | 4066+V+611+F | 11,978 |
| Gate / private plan | V+6n+2n+2+3+26 | 28,337 |
| Native roster / checked roots | fixed roster and scalar conversions | 613 |
| Evaluations / updates | 129+129+128 | 386 |
| Candidate export | 2M+4 | 598 |
| Objective remainder | 2V+3(512)+3(202)+20 | 9,622 |
| Dual remainder | V+5(792)+5(99)+6M+4M+M+8 | 11,460 |
| Merge / weighted support | 2+4F+4+4(2F+3g) | 42,882 |
| Receipt / terminal | 16+36+4+2+7+9+1 | 75 |

The last row covers four candidate field sets, two eighteen-field receipts,
four bound encodings, two issuances, complete coverage, fixed proof metadata
and final completion. It is not an unassigned buffer. Extra implementation
fields/passes must be counted. All four original dual rosters, including both
zero proposals, are parsed; full-F merges and norms do not assume residual
sparsity or early cancellation. Arbitrary-source cost uses actual calls, not
these fixed-shape bounds.

## Representation and independent source checking

Use an opt-in V3 schema with explicit nonnegative binary64 factor scales D.
Input centers and outward radii are independently reconstructed from clipped
intervals. Input nodes retain distinct unit-coordinate factors. The first
affine local bank stores exact captured W, with independent center calculation
and center-error compensation; no float W*D intermediate is trusted. Later
affine/reference and guard semantics follow V2. All auxiliary factors have D=1;
expert-private factor identity, error-before-lift ordering, rhs and every
tie-legal unordered pair remain checked. No old matrix/bound is relabelled V3.

For C=(I-R)^(-1)LD, weighted local norms group equal scales **after** absolute
values, without merging factor identities. ReLU uses the recursive triangle
radius. This is a new enclosure, not V1/V2 bound equivalence. Zero scales require
no division and do not remove nonzero input-center terms. Nonuniform clipped
radii and nonzero centers are correctness controls, not large-capacity claims.

Default V1/V2 parsing and their numerical contracts remain unchanged. Small
internal parsing/norm hooks may be factored to share validation; a V1 entry must
still reject V3 and all old regressions must pass. A checked V3 handle can only
be issued by successful source checking and stays bound to its cumulative Work
session. Pure source checking remains stdlib-only.

## Multi-target exact consumer

For each endpoint, independently compute the objective adjoint once. For each
bounded tile, propagate the proposed dual columns together; reuse a coefficient
visit, but charge every active column's exact multiply/add. Merge objective and
dual contributions at the same global factor before taking absolute values.
Then group by exact D: sum_g D_g sum_{j:D_j=D_g}|residual_j|. This preserves the
shared input relation and avoids repeated scale multiplication.

Both proposed and explicit zero candidates are completely parsed and checked.
The checker constructs the entire plan from the checked source and properties;
caller plans are not accepted. All pair/property/endpoint slots and constraint
rosters are mandatory. The deterministic selection is max(proposed, zero), with
proposed chosen on exact ties. No native lower bound authorizes acceptance.

One request-owned checker issues sealed receipts binding representation,
source/request, factor map, task, candidate bytes, both exact bounds, choice and
the same Work session. Terminal aggregation verifies the complete receipt
roster and deadline rather than reevaluating the identical selected candidate.
Serialized receipt claims are never trusted by the portable checker: it reloads
the source and all raw candidates and repeats independent checking. Partial,
missing, duplicate, stale, cross-session and fabricated receipts cannot close a
request. Threshold remains strict `lower > 1/10000000`.

The CPU candidate path uses the same 128 projected-subgradient updates and 129
evaluations as the old path, with one thread and float64. Its operator is the
scaled general-DAG forward/adjoint, not the V1 one-level scatter. It exports
candidates without doing its own exact selection; the independent checker
performs that selection once. No optimizer, fallback or obligation is dropped.
Native and reference buffers are bounded; at most one tile of exact objective
and dual residuals lives at once. No constraint matrix is expanded in production.

Splitting objective/dual work can cost more if they would cancel before a dense
ancestor. Controls must compare exact bounds for identical candidates and record
this case, not claim universal speedup. Representation and consumer changes
remain distinct factors in any later performance experiment.

## Finite controls and acceptance

Reuse mixed and deep captured recipes for tied/all pairs and general reference
depth. Add exactly two declared input variations of the deep graph: center
(0.875,-0.75), radius 1/4, clip [-1,1]; and the same center with radius zero.
They exercise distinct clipped scales, nonzero centers and zero D. Also retain
the one-dimensional half/guarded controls for complete endpoint aggregation.
Do not select a source based on its output bounds. No dense fixture is generated.

Required controls:

- source/scale/center/weight/factor/guard/expert tampering, wrong schema, suffix
  mutation, equal scales with distinct factors and untrusted handle rejection;
- full affine/ReLU enclosure, source suffix exhaustion and same-V3 eager matrix
  comparison for forward/adjoint and exact proposed/zero candidate bounds;
- multi-column vs scalar, objective/dual cancellation, partial private-factor
  sharing, nonzero/fixed/zero scales, reference depth and independent rhs;
- all ties, multiple pairs/classes, missing/extra properties/endpoints/duals,
  positive inequality duals, nonfinite/malformed candidates and exact selection;
- alias/receipt pollution, cross-request/cross-session reuse, duplicate tiles,
  invalid complete-plan construction and no positive terminal after deadline;
- cumulative work exhaustion, exception/partial candidate retention, portable
  checking with `-I -S` and no native modules or historical data;
- unchanged V1/V2 source/endpoint and archive-wrapper regressions.

Finite control calls each use the existing environment, one CPU thread, one
parent 300s hard deadline and cumulative 8M Work cap. Fixture construction,
source conversion, candidate generation, exact checking, serialization and
reception all count. Store control evidence in a new directory under
`runs/hz_scaled_batch_20261013_r1`, capped at 16 MiB; preserve attempts/failures.
Archive committed code before final controls; offline rechecking has separately
reported cost. No real-source correspondence, GPU acceleration, external
advantage or G1–G6 completion follows from passing these controls.

After controls, inspect actual call sites/counts against the full ledger. Only
then may a separate frozen execution admit the unchanged dense source once.
No old96M limit, cap increase, counter redefinition, sealed input, holdout or
physical GPU retry is authorized by this protocol.
