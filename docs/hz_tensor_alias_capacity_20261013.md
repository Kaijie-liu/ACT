# First-affine tensor aliases: standalone capacity refusal

2026-10-13. Read-only analysis at `7684307c18a5c8269358c86cb56e5e96d19747e3`.
**Decision: do not implement first-affine tensor aliases as a standalone next
mechanism.** With the current complete-query consumer, removing first-affine
materialization cannot establish capacity for the registered eight-expert
recipe. This is a bound on specified code paths, not a complexity lower bound
on MoE verification or an observation about a trained model's safety.

The [machine-readable calculation](hz_tensor_alias_capacity_20261013.json) binds
thirteen code/configuration/result files, including the inherited checker
implementations. Its
[stdlib analyzer](../scripts/analyze_tensor_alias_capacity.py) reads layer shapes
and code only. No checkpoint, input, source conversion, candidate optimization,
GPU call or new numerical experiment was executed. The previous
[dense synthetic query](hz_scaled_dense_20261013.md) remains closed and
UNKNOWN_NONPOSITIVE; its measured 7,351,435 Work is not a trained-model result.

## Complete-query lower bound, even with source work free

The registered seed-1/2 recipe is input 3072, router `3072→128→8`, and eight
experts `3072→256→128→10`. This is not a new topology check of seed 0. The
unchanged V3 plan covers 28 unordered pairs and nine properties per pair:
252 required properties. It does not remove pairs using an uncompleted
feasibility query.

Assume first-layer storage aliases do not change the factor roster, node
allocation, pair-local tiling or the full scans in
[scaled_endpoints.py](../scoped_source/scaled_endpoints.py), and all endpoint
candidates are present. There are at least:

- `F = 3072` factors, including zero-scale input factors;
- `N = 3072 + 3288 + 336 = 6696` nodes: input nodes, linear outputs, and
  `28×2×(8−2)` guard nodes. This omits ReLU and compensation nodes.

Let `p` be the number of pairs whose computed gate interval is not the
singleton `{1/2}`. The endpoint count is `E = 252 + 9p`. At the most favorable
allowed tile width eight, there are `T = 56 + p` pair-local tiles.

Every present endpoint performs three full-factor loops: residual merge,
proposed-candidate norm and zero-candidate norm. Each charges one coefficient
visit and one exact operation per factor. The consumer also traverses every
node for each of 28 gate pullbacks, each objective pullback and each dual tile,
including nodes with zero adjoints. Consequently its partial cost is at least

```
L(p) = 6 F E + N (28 + E + T)
     = 6,894,720 + 232,848 p.
```

| Represented router case | Endpoints | Tiles | Consumer Work lower bound |
|---|---:|---:|---:|
| All eight affine forms identical (`p=0`) | 252 | 56 | 6,894,720 |
| Smallest nonidentical case (`p=7`) | 315 | 63 | 8,524,656 |
| All pairs nondegenerate (`p=28`) | 504 | 84 | 13,414,464 |

These are lower bounds, not runtime estimates or complete upper bounds. They
omit source construction/checking, most coefficient arithmetic, native work,
metadata, serialization and finalization. With 504 endpoints, the factor loops
alone cost 9,289,728, already above the unchanged 8M limit.

Why is seven the smallest positive `p`? The gate is a singleton exactly when
the represented router centers and D-scaled global affine forms agree. Equality
partitions the eight forms into classes. If not all agree, the fewest cross-class
pairs arise from sizes `7+1`, giving seven. The analyzer enumerates all 22 integer
partitions to check the count; it does not enumerate observed model routes.
Equality on a stronger guarded or constrained feasible set is not what this
global-box gate calculation tests.

**The actual router case is unknown here.** No trained weights or input values
were loaded. Topology alone does not rule out `p=0`; that row's consumer lower
bound is below 8M. Thus the consumer-only refusal is conditional on at least
two distinct represented router forms. The report leaves actual `p` null.

## Source and later-layer costs cannot be hidden by an alias

The present producer and independent checker each visit every first-affine
weight position, even a zero weight:

```
N_first = 3072 (128 + 8×256) = 6,684,672
2 N_first = 13,369,344 > 8,000,000.
```

Even granting the entire producer and native preparation for free, retaining
one checker's weight visit plus the unchanged minimum consumer costs at least
`6,684,672 + 6,894,720 = 13,579,392`. This optimistic variant therefore fails
even in the all-identical-router case. A different source check could change
the premise; this is not permission to omit independent source checking.

Two further restrictions are **conditional**, not observations of trained
nonzero values:

| Unchanged operation | Conditional restriction |
|---|---|
| Producer/checker norm and center arithmetic | Let `Z` count nonzero first weights, and `K` count first positions with both nonzero weight and nonzero interval center. At least `2Z+2K` exact calls remain. Dense weights imply 13,369,344 calls even with zero centers, or 26,738,688 if all corresponding centers are nonzero. |
| Later-affine explicit references | There are 273,408 later-layer weight positions. If these weights are nonzero and parent node IDs remain distinct, their references exceed the unchanged 65,536 cap. Actual nonzero counts were not read. |

Inactive ReLUs create distinct empty nodes in the present representation;
active ReLUs alias their distinct parents. Neither alone establishes that the
later-layer reference roster becomes small. Tensor sharing may avoid duplicate
storage, but does not itself prove centers, compensation, ranges or residuals.

## Next research decision, not an execution freeze

Stop this standalone alias proposal. Do not add another source-only prefix,
repeat small controls, rerun the closed dense query, raise 8M/300s, relabel
per-scalar work as one operation, delete the zero candidate or retry physical
GPU admission. All G1–G6 remain OPEN, and sealed real inputs remain sealed.

A next proposal must jointly account for:

1. Source-bound tensor center/range computation and compensation, including
   nonzero centers; not merely storing weight aliases.
2. Complete-query residual work without repeating unrelated node/factor scans,
   preserving shared-factor summation before absolute values and both candidate
   checks.
3. Later-affine tensor references and all layer/factor identities.
4. Every pair/property/endpoint, raw evidence, portable reconstruction, fallback
   and final aggregation under the existing cumulative budget.

One candidate for analysis is **scoped expert-margin residual templates**. This
would cache checked affine residual vectors, not only previous positive scalar
bounds or propagated expert HZs. Let the homogeneous node vector be `h=Cζ`,
with `|ζ_j|≤D_j` and checked pullback `P=Cᵀ`. All output/constraint centers
are already included in `d_i`, `b_i` and `g_ab`. For the property
`d_i+c_iᵀh`, expert constraints `A_i h≤b_i` and legal dual `y_i`, form

```
R_i = P(c_i − A_iᵀ y_i),      κ_i = d_i + b_iᵀ y_i.
R   = t R_a + (1−t) R_b − P(G_abᵀ v),
κ   = t κ_a + (1−t) κ_b + g_abᵀ v,
L   = κ − Σ_j D_j |R_j|.
```

The factor vector retains the same global shared/private identity; auxiliary
binary factors are relaxed by their existing boxes, not removed. Both experts
use the identical property and offset, including `−margin`, so their convex
combination includes that offset once, not twice.

Here `G_ab h ≤ g_ab` denotes additional pair constraints in the same node
coordinates before pullback, and inequality duals have the existing nonpositive
sign convention. Equality duals, if present, remain unrestricted. Both expert
domains must contain the pair domain. A checker must verify row embeddings,
scope inclusion, property/source identity and multiplier signs. Shared rows
have their scaled multipliers added; expert-private factors must not be
identified. The shared residual is combined **before** taking absolute values.

For `t∈[0,1]` this constructs a valid candidate from legal template and pair
duals. It restricts the candidate family compared with arbitrary pair-specific
duals: neither equal bounds nor unchanged coverage follows. An unrestricted
fallback, if retained, must be fully charged. The 72 expert/property templates
could themselves repeatedly traverse the dense first bank; this algebra is
therefore **not a capacity admission**. A complete registered-recipe ledger
must precede implementation or another microbenchmark. No particular new
algorithm, threshold, source or real execution is frozen here.

## Checks and reproducibility

Six static unit tests pass: source counts/closed-result boundary, complete
consumer counts, the all-equal exception, all equivalence partitions,
pair-local tiling/denominator, and code-identity/missing-inventory rejection.
They check arithmetic and bindings; they do not prove the whole implementation.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S -m unittest scripts.test_tensor_alias_capacity -v
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -S scripts/analyze_tensor_alias_capacity.py --check-report docs/hz_tensor_alias_capacity_20261013.json
```

Only small code/report files are added. No raw archive, checkpoint, environment
or scientific evidence is created, deleted or replaced by this stage.
