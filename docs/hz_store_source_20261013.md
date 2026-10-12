# Row-consuming checked source operators on the HybridZ store

Completed 2026-10-13 under the separately
[frozen 2026-10-12 protocol](hz_store_source_design_20261012.md).
The [compact result](hz_store_source_20261013_r2.json) records four normal objects,
their identities, costs and scope. All G1–G6 remain OPEN.

## What changed

The [independent source checker](../scoped_source/check_hz_store_source.py) now
consumes immutable rows from the request-owned common/private store. It checks
input containment, exact affine residuals, ReLU graph construction, common-input
restoration and every unordered pair guard. It does not expand a view through
legacy whole-source parsers or their small dimension limits.

Only final binary64 rows are stored. Exact operator reference rows are temporary;
an enclosed target inherits the preceding checked state's constraints, not a
potentially nonrepresentable intermediate reference. The checker fixes the entire
operator-factor inventory before allocating output and equality compensation.
Shared input/router factors retain one identity; expert factors remain private.
This is inclusion with explicit compensation, not exact equality to floating
arithmetic or a new general abstract domain.

The [producer](../scoped_source/hz_store_source.py) obtains affine nominals through
actual `sparse_hz_linear`. ReLU uses an independently checked compressed graph,
then becomes an actual SparseHZono; this is not a claimed differential with
`sparse_hz_apply_relu_exact`. The [live bridge](../scoped_source/hz_store_sparse.py)
uses explicit global-to-continuous/binary maps and reads all nine sparse fields,
including zero columns, RHS vectors and the frame/enclosure flag. It independently
checks CSR pointer bounds and strictly increasing row indices rather than trusting
a possibly stale cached `has_canonical_format` flag.

For affine nominal polynomial p and exact declared polynomial f, a fresh factor
with coefficient R = |f₀−p₀| + Σ|fⱼ−pⱼ| covers the residual on factor boxes.
The checker reconstructs R from the original declared operator and parent rows.
If R or a new ReLU coefficient is not binary64, checked outer compensation covers
the additional rounding. Inequalities weaken their RHS by the coefficient error;
unchanged ancestor constraints are shared, not replaced or rounded again.

## Fixed controls and independent checks

| Saved object | Checked content | Result |
|---|---|---|
| Wide, page 64 | 3,072 input/output coordinates, diagonal affine, three router scores, three expert projections, all three pairs | Every affine row has independently derived residual 2⁻¹⁰⁴; 6,144 factors |
| Mixed, page 64 | Two affine/ReLU stages in router and each expert, input restoration and all guards | 906 factors; 65 then 2 unstable ReLUs per network |
| Same mixed object, page 1 | Complete row coefficients, factor identities, ownership, constraints and maps | Same mathematical signature as page 64 |
| Rational input/ties | [1/3,2/3], [0,1], [1,1]; three equal router scores | Input containment checked; all three tie-legal pairs retained |

Mixed rows 0 and 65 straddle a page boundary. In each of the four networks their
first ReLU needs output compensation 2⁻⁵⁴ and equality compensation 2⁻⁵³:
eight paired locations, **sixteen radii**. The second ReLU appends new constraints
after an existing 130-inequality prefix. One zero-radius input factor remains
explicit, so 905 of 906 mixed factors have nonzero coefficients; it is not lost
from the live coordinate map.

The normal execution records 69 complete per-view live CSR matches. Nineteen
source refusals cover inward/missing input, affine error, target-only terms,
missing enclosure, missing rows, wrong ReLU order, inherited-prefix replacement,
factor ownership, common restoration, guard orientation, source identity,
missing layer and partial-deadline checking. Fifteen live refusal controls cover
all nine fields, bad kind maps/frame/flag, cached duplicate indices, dangling data
and invalid pointers. Refusals require specific reasons, not arbitrary errors.

Two saved objects (wide and mixed) recheck after relocation under `python -B -I -S`,
without model/data, ACT numerical modules or a solver. The portable checker has
the store, declaration and standard-library source checker only. It checks a
declared source, not correspondence to an unprovided trained checkpoint.

The [separate-process audit](../scripts/audit_hz_store_source.py) reconstructs all
four source checks, analytic oracles and all nineteen source mutations from saved
objects. It verifies inventory, code snapshots, cost clocks and portable identities.
The final saved audit also binds the executing checker/recipe files to the
recorded versions. The initial audit and its exact code remain beside `audit_v2.json`;
this hardening re-reads the same objects without generating a new source.
It **does not independently replay archived native CSR objects**: those 69 matches
and 15 mutations remain numerical execution controls, not an invented extra
standard-library proof. Eleven new exact regressions and 57 prior regressions
pass (68 total); the observed command/result is recorded in the raw R2 archive.

Pre-execution review fixed two implementation defects: future compensation
columns could enter the reference basis, and duplicate CSR indices could hide
behind a cached flag. It also fixed a source-capture race: the external hash now
binds the actually captured bytes, not a second serialization of a mutable caller.
Regression controls cover all three. These were correctness repairs before R1,
not changed scientific outcomes or threshold adjustments.

## Attempts and cost boundaries

Both attempts are retained outside the repository:

```text
/data1/Kane/MOE/baseline_runs/hz_store_source_20261012_r1
/data1/Kane/MOE/baseline_runs/hz_store_source_20261012_r2
```

R1 passes the fixed mathematical controls but lacks a saved parent process clock.
R2 changes only the test harness to record startup/return and report reception.
All four normal manifests and declared-source identities are identical across
R1/R2. No third generation was run. The independent audit only reads saved objects.

R2 takes **5.035 seconds** from parent entry through the observed child return and
receipt, within its 300-second control envelope. Peak child RSS is 603,640 KiB,
including numerical imports and all controls; it is not an isolated wide-object
memory measurement. The retained stage is about **8.18 MB**, below the 64 MiB cap.
Tiny costs are not speedup evidence or real-model throughput.
The parent entry is after that Python script's own imports; this is not a
measurement of the external CLI's complete startup or a production request.

The wide case records 1.693 s generation (including imports), 0.162 s store load,
0.193 s exact source check and 0.189 s live instantiation/matching. Within generation,
0.092 s is native workspace/affine work and 0.018 s member serialization; these
subintervals must not be added to generation a second time. Declaration creation,
declaration serialization, analytic checking, signature calculation and receipt
publication are separately recorded. Offline audit cost is not generation cost.
The last receipt write follows the reported observation. This control harness is
not a newly validated production hard-budget/partial-evidence supervisor.

R2 report SHA-256:
`edbe82279fbcc4be4b03e0c45328a437e5497e54e08664b871d8323ed3e18b49`.
Independent audit SHA-256:
`aba6afc659c3d1477ed0fee48354644a8427284e7b6a11a26baa3c38b9801ada`
(`audit_v2.json`).

## Decision and remaining work

Stop this component's width/timing study. Next implement the complete property
inventory and bounded endpoint tiles directly on these views, without rebuilding
the old duplicated full-source dictionaries. Preserve every distinct endpoint
and check all duties; a readable source store alone proves no output property.

No output query, candidate optimization, native solver, real model, training or
CUDA ran here. The 252-duty capacity target is not closed. Full source capture,
tensor streaming, convolution/pooling, new-schema production supervision and
physical GPU admission remain separate open gates. A diagonal 3,072-row source
does not establish trained dense-network capacity. This completes a source/view
mechanism component, not G1, G2, G3 or a new external win.
