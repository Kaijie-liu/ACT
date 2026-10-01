# Same source HybridZ representation results

The [precommitted protocol](hz_source_representation_design_20261001.md), frozen
at `ed453c2e1`, completed four original declared sources and two representation
arms. Thirteen control groups and 103 existing regressions pass. The independent
[saved-evidence audit](hz_source_representation_20261001_r1.json) checks all eight
packages and the new MC feasibility diagnostic without solving.

This is a direct finite mechanism result on **actual, source-checked HybridZ**,
not merely on the earlier direct-node LP. It is not a trained-model result,
GPU speedup, external-tool win or native-floating execution proof. All G1–G6
remain OPEN.

## Complete outcomes

The table counts positive pair/property obligations; all required obligations
must pass for a positive whole request. The two arms each reconstruct their
source, propagation and joint HZ. Their entire checked lowering is identical,
including input containment, affine compensation, ReLU constraints, factor maps,
all guards, gate intervals and output properties. Nothing is dropped or reused
from the previous positive records.

| Original declaration | Endpoint positive / required | MC positive / required | Endpoint full request | MC full request |
|---|---:|---:|---|---|
| weighted_sign | 3/3 | 2/3 | positive | UNKNOWN_NONPOSITIVE |
| tied_partial_reuse | 18/18 | 18/18 | positive | positive |
| unsafe_tied | 0/6 | 0/6 | UNKNOWN_NONPOSITIVE | UNKNOWN_NONPOSITIVE |
| unresolved_sign | 6/6 | 6/6 | positive | positive |

There are 33 duties per arm, 42 distinct endpoint targets and 33 MC targets.
The tied fixture does not receive free proof reuse. Empty guards remain duties;
a positive bound on an empty branch is not evidence that the route is reachable.
The existing source has distinct legal routes, but this comparison does not
claim a new empirical route-flip census.

## A checked representation gap

For weighted_sign, pair {0,1}, class1, the exact checked endpoint lower bound is

    194996818405069 / 18014398509481984 ≈ 0.0108245.

The predeclared MC diagnostic is freshly expressed in this HZ's continuous and
binary factor coordinates. Its gate is 1/4 and product variable is -1/8. After
independent source and MC-construction checking, every equality, inequality and
box bound is checked with exact arithmetic. Its feasible objective is

    -225179981368525 / 9007199254740992 ≈ -0.025.

Thus this same-domain range-only MC LP cannot prove that obligation positive,
whereas the endpoint representation does. This is stronger than simply observing
that its fixed 128-step candidate lower bound is negative: the feasible point
provides an upper bound on the MC optimum. It is **not** a real-network witness,
an LP optimality certificate or a claim about every possible MC strengthening.

The difference interval is recomputed from this new joint HZ, not transplanted
from the earlier analytic example. It is exactly

    [-13510798882111487/9007199254740992,
       13510798882111489/9007199254740992].

Its slight asymmetry comes from the declared stored coefficients. The old
convenient interval [-3/2,3/2] was not silently substituted. No extra range
optimization, tighter gate, merged private factors or altered shared domain was
given to the endpoint arm.

For fixed-gate controls, the two mathematical representations are equivalent;
their different floating candidate parameterizations need not return identical
lower bounds. A full [0,1] gate remains the expert-wise sufficient condition.
The experiment isolates the representation factor, not shared versus independent
inputs or a novel general convexity theorem.

## Implementation and numerical contract

The additive [producer](../scoped_source/hz_representation.py) uses the unchanged
SparseHZono propagation and endpoint support APIs. MC uses the existing rational
constructor, with the same projected-dual algorithm and 128 updates. Its rational
vectors are converted to finite floats only in the candidate copy; its original
LP and exact certificate identity are retained for acceptance.

The [independent bridge](../scoped_source/check_hz_representation.py) does not call
propagation, the MC builder or the candidate optimizer. It rechecks source
lowering and pair maps, independently derives the difference range, validates all
MC planes and objectives, and checks the exact residual-compensated lower bound.
Missing duties, stale candidates, wrong properties, inward ranges, factor aliases
and mutated planes are rejected or remain UNKNOWN according to their contract.

The final guarantee is for the declared real graph with actual binary64 parameter
values. Correspondence to an intended program and checker implementation remain
trusted; native floating-point execution remains outside the guarantee. The
production default verifier and all prior mathematical modules are unchanged.

## Costs and scope

| Declaration | Endpoint finite arm seconds | MC finite arm seconds |
|---|---:|---:|
| weighted_sign | 0.088 | 0.125 |
| tied_partial_reuse | 0.171 | 0.260 |
| unsafe_tied | 0.083 | 0.105 |
| unresolved_sign | 0.103 | 0.119 |

Each finite arm charges source creation, propagation/joint preparation,
representation construction, proposals, in-memory serialization and checking to
one cooperative 300-second deadline. Imports (0.808 seconds) and test/archive
orchestration are outside those arm columns and separately recorded; the complete
control suite took 2.173 seconds. These tiny, warm-process measurements are **not**
a production end-to-end speed comparison or a hard-supervision result.

No native optimization, CUDA, real/model-data loading or extra sample ran.
The single raw R1 directory is
`baseline_runs/hz_source_representation_20261001_r1` (1,053,351 bytes). It includes
code/config snapshots, eight normal packages, tests and costs. No failure or old
evidence was deleted. Both read-only code reviews found no blocker within this
finite comparison; they do not replace independent human review.

## Next decision

The finite same-source endpoint-versus-MC mechanism gap is now established. Do
not repeat this example, tune its 128 steps, add a supervision layer for a better
label, or expand a toy merely to make a larger table. Shared-input independence
remains a separate factor, not an explanation already isolated by these two arms.

The next necessary G1/G3 step is an explicit **real-object intake and capacity
assessment** using existing configuration and source metadata: supported operators,
expert/class counts, factor/constraint growth and exact coefficient round-trip
limits. Current finite limits do not admit arbitrary real networks. Select any
next representation/capacity change from that evidence and freeze it separately
before proposing a real run. Do not raise caps blindly, reopen sealed inputs,
retry a refused GPU automatically or treat this result as external superiority.
