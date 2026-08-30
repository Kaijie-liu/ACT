# CIFAR100 / TinyImageNet failure buckets and C1 stop loss

Date: 2026-07-29

This note is a diagnostic research ledger, not proof authority.  Historical
labels and ground truth may select development cases and explain regressions;
they may never alter a bound, certificate, raw VNNLIB predicate, or verdict.
Gate-40 remains held out from candidate design until Gate-14 promotion.

## Historical failure buckets

The historical `sound6` totals were:

| family | VERIFIED | ADV | UNKNOWN | TIMEOUT |
|---|---:|---:|---:|---:|
| CIFAR100 | 37 | 26 | 2 | 135 |
| TinyImageNet | 156 | 36 | 7 | 1 |

This separates two different problems:

1. CIFAR100 is dominated by proof-cost explosion.  A previous
   three-certificate experiment changed many verified instances into
   timeouts, while a later persistent HiGHS prototype kept five verdicts
   fixed and reduced its Tier-3 loop by `7.8x` (`4.68x` end to end).
2. TinyImageNet is primarily a remaining tightness/coverage problem rather
   than a global timeout problem.

The fixed development ladder is the manifest's `6 -> 14 -> 40` split.  Each
stage contains both historical S and U cases across CIFAR100-medium,
CIFAR100-large, and TinyImageNet-medium.  No candidate may choose a verdict
from those labels.

## Closed or subordinated directions

- Global LP construction: about 10.5 million nonzeros and 22--24 seconds of
  build time before useful rival work; all 99 iid2 rivals timed out.
- Final two-neuron PairHull: exact Fraction proposals improved intercepts only
  at roundoff scale and retained `0/99`.
- Dyadic final-plane mixtures: only one tiny `0.000688685` improvement.
- Final ADD source planes: doubled useful source material without a measurable
  bound gain.
- Blind larger exact-ReLU budgets, global RLT, dense convolution expansion,
  and full-dataset reruns.
- Layerwise query dual is a tightness oracle, not the current implementation:
  on iid2 it reduced the diagnostic worst property upper from `69.590067` to
  `14.686187` and crossed zero on `85/99` rows, but independent replay missed
  the 12-second stop loss.

These results locate the missing information before the final tail: shared
correlations lost at materialized residual/DAG joins.

## C1: property-conditioned row-local ADD correlation shadow

For a
`ADD -> [FLATTEN] -> (DENSE|CONV2D) -> RELU` route, retain the ordinary
materialized ADD and all equality bands.  For property-selected ReLU rows
only, recompose the affine row over the pre-materialization ADD expression
and intersect its outward box with the ordinary cube.  Selection has no proof
authority.  Soundness follows from intersecting two independently outward
enclosures of the same stored-float graph.

Controlled evidence before any real run:

- cancellation toy: ordinary preactivation about `[-1.75, 2.25]`, shadow
  `0.25 +/- outward error`, changing one ReLU from relaxed to stable active;
- 96 deterministic dyadic Fraction cases with zero enclosure violations;
- point-box consistency and exact affine-Jacobian/fanout checks;
- duplicate, missing, non-ReLU, and out-of-range targets fail closed;
- the targeted operator/residual/preactivation/gate closure passes
  `144/144`.

### Single authorized iid2 build-only probe

Configuration:

- CIFAR100-medium iid2 only;
- `property_correlation_budget=32`;
- deterministic equal-depth quotas, at most 8 rows per targetable ReLU;
- selector time limit 1 second;
- materialized ADD on;
- exact ReLU, preactivation LP, residual normal form, property tail,
  PairHull, query dual, GPU dual, and verdict LP off.

The candidate is promoted to Gate-6 only if the receipt has:

- stable inputs/source and zero fallback/integrity error;
- at least four targetable residual depths and at least 24 prepared rows;
- at least eight strictly tightened rows across at least two depths;
- no property cube upper regression;
- at least 5% aggregate property-upper improvement and either 5% worst-upper
  improvement or four newly stabilized ReLUs;
- operator build at most 30 seconds and constraint nonzeros no larger than
  the materialized baseline `10,498,220`.

Zero strict tightening, a single-depth-only effect, any soundness conflict, or
missing the time/size gate closes C1 without a budget sweep or alternate iid.
Only a geometrically useful but under-budget C1 receipt may justify a
separately preregistered C2; it does not authorize Gate-6 by itself.

### C1 result and C2 fanout correction

C1 is closed.  Its one authorized receipt is
`gate1_property_correlation_k32_buildonly_20260729.summary.json`:

- build `22.1558` seconds, `10,498,220` nonzeros;
- only final ReLU 40 was admitted (`8` prepared rows);
- all eight rows changed only by outward-roundoff scale
  (`max improvement ~= 9.95e-14`);
- zero stabilized phases and unchanged property cube
  `[89.2641, 120.1344]`.

The failure exposed a static routing error in the candidate definition.  The
official residual ADDs fan out both to the next convolution and to a later
skip ADD, e.g. ADD 8 has successors 9 and 12.  Requiring a single ADD
successor excluded the intended layers 10, 14, and 22 and reduced C1 to the
already-closed final-tail case.

C2 is a distinct, preregistered mechanism, not a budget retry:

- preserve every original fanout and materialized equality;
- allow exactly one explicitly targeted affine/ReLU route from an ADD while
  other successors continue to later skip joins;
- reject more than one targeted nonlinear route as ambiguous;
- discover all such residual depths, discard depths with zero unstable rows,
  then allocate the same total budget round-robin.

On the fixed medium topology the static routes are
`10,14,18,22,27,31,35,40`; certified preactivation facts remove the already
stable `18,27,31,35`, so the frozen budget is
`{10:8,14:8,22:8,40:8}`.  The shared-fanout Fraction/Jacobian toy must pass
before launch.  C2 reuses the same iid2, 32-row, 1-second-selector, 30-second
build and nonzero stop losses.  It additionally requires strict improvements
at three or more depths.  Failure closes row-local correlation shadows; it
does not authorize 64 rows, another iid, or Gate-6.

### C2 result and C3 residual phase screen

C2 is closed.  It covered the intended schedule
`{10:8,14:8,22:8,40:8}` with 32/32 strictly intersected rows and no
soundness/integrity error.  Build time was `18.9039` seconds.  Layer 10 gained
as much as `0.153638` on each side and layer 14 gained `0.001701`, but no row
became phase-stable.  The worst property upper changed only
`120.134377 -> 119.836596` (about `0.25%`), so the candidate missed its
property and phase gates.

C3 changes the retained object rather than increasing the property budget.
For every supported residual depth it scans only the ordinary cube-unstable
rows in bounded chunks, recomposes their pre-ADD shadow, and commits a row
only when the outward shadow proves `upper <= 0` or `lower >= 0`.  Ambiguous
rows and their transient generators are discarded immediately.  The original
fanout, materialized variables, and equality bands remain unchanged.

This is motivated by the earlier full live-affine receipt, which found 141,
59, and 98 additional inactive rows at layers 10, 14, and 22 respectively.
C3 should recover those phase facts without retaining full-width affine
generators.

One iid2 build-only probe is authorized after the shared-fanout,
point/Jacobian, Fraction, crossing-zero, and blast-radius tests pass:

- `residual_phase_screen=true`;
- property correlation selector/budget off;
- every other candidate and verdict LP off;
- scan the fixed residual routes once, with no quota sweep.

Promotion requires zero integrity/soundness error, at least 1,200 unstable
rows scanned, at least 250 newly stable inactive rows across at least three
depths, build at most 27 seconds, no nnz increase, at least 5% aggregate
property-upper improvement, and at least 10% worst-upper improvement or a
newly certified property row.  Failure closes C3 without another iid or a
larger screen.

### C3 result and C4 recursive skip shadow

C3 is closed.  Its screen took only `0.1930` seconds inside a `17.7772` second
build, scanned all 1,234 expected unstable rows, stabilized 15 active and 141
inactive rows, and reduced nonzeros to `9,889,170`.  All 156 phase facts came
from layer 10.  Layers 14 and 22 retained zero rows, and the worst property
upper reached only `118.584104` (about `1.29%` better than baseline), missing
the preregistered multi-depth/property gates.

The reason is structural: C3 removed only the immediate ADD materialization.
At ADD 12 its skip operand still referenced the normalized variable created
at ADD 8, so correlation could not survive across residual depths.  The
historical unmaterialized full-width run did preserve that skip chain.

C4 adds a read-only recursive skip shadow:

- after each nonlinear main branch, ordinary materialized ReLU variables
  remain authoritative;
- when a later ADD consumes a previous ADD through its skip edge, its
  alternative source reuses the previous pre-materialization shadow;
- the shadow is never returned as the graph value and never changes another
  fanout;
- more than one targeted nonlinear route remains fail-closed.

The two-block Fraction toy proves that the second residual depth recovers
`ReLU(1/4)=1/4` only with recursive skip provenance, while the ordinary graph
remains relaxed.  C4 may run the same single iid2 phase-screen configuration
once because its source semantics and controlled oracle are distinct.  It
must stabilize at least 280 inactive rows across at least three depths,
retain zero crossing-row false phases, build within 27 seconds, not increase
nnz, and improve the worst property upper by at least 10% (or certify a row).
Failure closes recursive phase screening without a second instance.

### C4 result and C5 recursive residual bound screen

C4 recovered the historical multi-depth signal with less than half a second
of screen work:

- layer 10: 15 active + 141 inactive;
- layer 14: 59 inactive;
- layer 22: 10 active + 95 inactive;
- total: 25 active + 295 inactive across three depths;
- build `17.0496` seconds and `9,279,838` nonzeros.

It nevertheless missed the property gate: worst upper
`120.134377 -> 115.966334`, about `3.47%`.

C5 performs no additional row scan and keeps no transient generator matrix.
It changes only the commit rule: for each of the same 1,234 unstable rows,
retain any strict outward `l/u` intersection, even when the row remains
unstable.  Thus all C4 phase facts remain, while surviving triangle
slopes/intercepts can tighten.  A crossing-zero Fraction toy verifies that an
exact `[-1/2,1/2]` row stays relaxed but replaces the ordinary
`[-3/2,3/2]` bounds without excluding an exact point.

One iid2 build-only C5 probe is authorized with
`residual_bound_screen=true` and phase-only mode off.  It must scan the same
1,234 rows, preserve at least C4's 320 phase facts, strictly tighten at least
500 rows, build within 27 seconds, not increase nnz over C4, and improve the
worst property upper by at least 10% from the materialized baseline (or
certify a property row).  Failure closes the recursive scalar-bound family.

### C5 result and one verdict probe

C5 passes its build-only geometry gate:

- every currently unstable residual-route row was covered: `1,232/1,232`;
  the count is two below C4's initial 1,234 because preceding tightened
  layers had already removed two later unstable rows;
- all 1,232 rows had a strict outward intersection;
- 26 active and 296 inactive phases were proved;
- screen `0.4369` seconds, build `16.7751` seconds;
- `9,267,544` nonzeros;
- property cube range
  `[49.286389, 69.255736]`, versus baseline
  `[89.264134, 120.134377]`; worst-upper improvement is about `42.35%`.

This authorizes one C5 iid2 verdict probe under the official 100-second outer
budget and 60-second persistent-LP allocation.  Property tail, query dual,
GPU dual, exact ReLU, and every other candidate remain off.  A VERIFIED or
strict-replayed ADV result promotes C5 to the six-sentinel ladder.  An UNKNOWN
may justify a new proof-engine candidate only if at least 20 of 99 rival rows
receive independently checked LP certificates; fewer closes the unchanged
C5+LP combination.  No alternate iid or Gate-6 run is authorized by a mere
timeout.

The verdict probe returned UNKNOWN and closes unchanged C5+LP:

- build `16.5631` seconds;
- one persistent model, 99 completed fair-slice solves in `30.3563` seconds;
- only four candidate duals reached independent checking;
- `0/99` certified rows and no pruned rival.

### C6 composition with exact-audited final property planes

C5 changes the prefix geometry; the previously audited property-tail
Fraction planes change how that geometry is consumed.  One build-only
composition is therefore allowed, with C5 bound screening plus
`property_tail_upper`, 32 negative-alpha candidate steps, and a 1.5-second
candidate cap.  Baseline and candidate planes remain grouped per rival;
selection has no proof authority and every exported plane is reconstructed
by the exact endpoint oracle.

Promotion requires all 99 groups covered, no fallback/integrity error, at
least 20 negative group-best uppers or a worst group-best upper at most 20,
and at least 30% aggregate improvement over C5's raw property cube.  Missing
all three closes this final-tail composition without PairHull, mixture, or an
LP verdict run.

C6 is closed.  All 99 alpha alternatives improved their grouped baseline,
with total proxy improvement `374.7334` and maximum per-rival improvement
`5.5018`, but group-best uppers remained
`[37.140955, 57.415596]`; zero groups crossed zero.

### C7 original-frame batched CUDA dual

C7 converts C5's tighter original HZ formulation directly into property
certificates.  CUDA optimizes one multiplier tensor indexed by
`[rival, original_constraint_row]`; it cannot certify a row.  Each candidate
is projected to legal signs and recomputed by the independent long-double
original-CSR residual-box checker.

One iid2 probe is authorized with C5, eight CUDA steps, an eight-second hard
candidate cap, at most 2,048 retained dual rows per rival, and verdict LP off.
It must have zero candidate/checker errors and independently certify at least
20 rivals to stay open; all 99 are required for VERIFIED.  Fewer than 20
closes the unchanged original-frame CUDA-dual candidate without a learning
rate or top-k sweep.

C7 is closed.  The fixed receipt
`gate1_recursive_bound_gpu_dual_s8_k2048_20260729.summary.json` completed all
eight CUDA steps in `8.0247` seconds with zero checker errors, but did not
improve the minimum support (`49.495168`) and certified `0/99` rows.

### C12--C21 shared-suffix localization

C12 first exported an independently replayed affine property plane over the
shared prefix at ADD 33.  It was sound and cheap (`1.0230` seconds), but its
free-cube improvement was only about `1e-11`.  C13's adaptive infrastructure
did not change that geometry.  C14's attempt to move directly to the earliest
ADD exceeded the 100-second build deadline.

C15 replaced three endpoint replays by one frozen optimized-alpha replay.
At the earliest ADD 8 it improved every rival's free cube by a mean
`16.2125`, but still left the suffix upper as high as `40.9729`; build time
was `55.43` seconds.  C16 bound each suffix row to the exact constraint prefix
at its stop layer and cached repeated prefix hashes.  HiGHS still failed to
close the remaining row inside the official budget.

C17 fixed a materialization mismatch by composing the suffix plane with the
pre-ADD correlated source rather than the normalized ADD box.  It improved
all 99 free cubes by a mean `21.9248` and reduced the suffix worst upper to
`34.0978`, but build remained about `55.9` seconds and the prefix LP stayed
UNKNOWN.  C18's original-frame GPU multiplier candidate also remained
positive.  C20 crossed every residual block and replayed directly to the
input box: all rows improved, but the independently replayed constants stayed
in `[22.1128,31.5862]` with zero negative rows.  C21 increased alpha
optimization from 4 to 64 steps and changed the range only to
`[22.0906,31.5745]`; extra alpha work is therefore closed.

These probes establish two distinct facts:

- moving the suffix stop earlier recovers substantial lost correlation;
- even a full-input single affine plane remains far from zero, so one global
  frozen-alpha plane cannot represent the needed piecewise ReLU geometry.

### C22--C26 exact phase cover and property-dual candidates

C22 implemented stable-id exact binary substitution and exhaustive depth-1/2
cover semantics.  A child may authorize SAFE without repeating parent base
feasibility only through the private exact-cover capability; every selected
binary assignment is enumerated and every continuous child must be SAFE.
The focused C22c row was still positive: its two prefix LP uppers were
`27.0063` and `27.0092`.

C23 moved the phase target to an adaptive one-block prefix.  The exact split
did not change the suffix plane, and the focused support remained
`17.808995` in both phases.  C24 added a deterministic coordinate wavefront
which crosses zero-gain chains; the real support improved for the first time,
to `17.729023` with 82 independently checked nonzero multipliers.

C25 seeded a bounded constraint-generation LP from that wavefront.  Loading
6,144 selected rows reduced support to `16.838317` in about `1.49` seconds.
C25b increased the selected set to 20,562 rows / about 2.83 million loaded
nonzeros, but support reached only `16.696054`; the independently checked full
upper remained about `40.0882`.  The marginal gain does not justify further
row-cap expansion.  C26 increased suffix alpha from 4 to 32 steps and slightly
worsened the focused cube (`41.2012 -> 41.2100`), closing that parameter line.

### C27--C31 branch-conditioned suffix replay

The missing operation was not merely fixing a binary in the prefix.  C27
replays a fresh suffix affine plane after imposing the selected exact ReLU
phase, attaches it to the matching child only, and materializes every
roundoff allowance as an explicit generator.  A private guard token prevents
conditional rows from appearing on the parent or the wrong child.

Two implementation stop-loss findings were corrected before measuring
geometry:

- replaying all 99 rivals for both phases tripled the expensive suffix work;
  the final version replays only the selector-bound dominant rival and leaves
  every other ordinary sound row intact;
- adding one output-error column caused the same 9-million-nonzero prefix to
  be hashed once per output row.  Hashes are now cached by exact
  `(eq_rows,ub_rows)` prefix, preserving the binding while removing the
  accidental `99x` work.

The first valid real receipt is
`gate1_c30_focused_conditional_cachedhash_iid2_20260729.summary.json`.
For CIFAR100-medium iid2, final ReLU `(layer 40,row 8)` and rival 50:

- ordinary focused cube upper: `52.428355`;
- inactive conditional phase: `39.072386`;
- active conditional phase: `40.111295`;
- build: `28.1908` seconds; total: `35.241` seconds;
- zero fallback, missing child, or soundness error.

This is a genuine `12.32--13.36` (about 25%) geometric gain, but neither
phase crosses zero.  C31's GPU wavefront/constraint-generation candidate
reduced prefix supports from `19.4199/19.8037` to
`17.9805/18.4640`; the extra `1.37--1.44` is still insufficient.

### C32--C34 joint depth-2 conditional replay

C32 generalized the private conditional capability from one guard to a
complete Cartesian guard set.  Two exact ReLUs require all four assignments;
omitting any quadrant fails closed.  Each quadrant independently replays one
jointly conditioned suffix plane.  The initial facility selector spent its
second bit on rival 70, so rival 50's worst phase was unchanged from depth 1.

C33 introduced a phase-cover-only selection policy: the first target is still
the multi-rival facility winner, while the second target maximizes the same
bottleneck rival's remaining ReLU contribution.  Ordinary residual selection
retains its multi-rival policy.  It selected final-ReLU rows 8 and 49, both
for rival 50, and produced four cube uppers:

| phase `(row8,row49)` | conditional cube upper |
|---|---:|
| `(-,-)` | 35.651387 |
| `(+,-)` | 36.559808 |
| `(-,+)` | 37.244085 |
| `(+,+)` | 38.328920 |

The worst phase improves another `1.7824` over depth 1 and `14.0994` over the
ordinary row.  C34 nevertheless closes this depth-2 line: after GPU
wavefront/constraint generation, the four candidate supports were
`17.8624`, `19.7372`, `19.7372`, and `19.7430`.  The worst support regressed
relative to depth 1, all four children remained UNKNOWN, and no Gate-6 run is
authorized.

The joint implementation remains as a sound research primitive.  Its
controlled audit includes a Fraction exact-phase network whose output is
identically zero, complete four-quadrant enumeration, rival/alpha binding,
wrong-child isolation, explicit roundoff factors, and incomplete-cover
rejection.  The current property/operator blast radius passes `98/98`.

### Next localization gate

The decisive residual is now the independently checked prefix support, not
the final cube: successful final-phase geometry can move a cube by fourteen
points while the worst constrained support remains around 18--20.  The next
candidate must therefore report a layer/block attribution of the support
term before changing another relaxation.  It may proceed only if a toy and
the single iid2 diagnostic agree on which earlier residual join or ReLU group
dominates the remaining support.  Blind depth-3 phase enumeration, larger
constraint-generation caps, more alpha steps, and a six-sentinel run are
closed.

## C35--C48 evidence ledger

### Status vocabulary

For this ledger, the historical manifest labels have one unambiguous
property-verification meaning:

- `S = SAT / FALSIFIED`: a valid input satisfying the VNNLIB input region
  violates the requested output property;
- `U = UNSAT / VERIFIED`: every valid input satisfies the requested output
  property.

A manifest label is useful for diagnosis, but is never itself proof.  An `S`
result requires an independently replayed counterexample; a `U` result
requires a sound proof receipt.  `UNKNOWN`, timeout, and worker error are
neither `S` nor `U`.

### C35--C41: localize the support before changing more relaxations

- **C35/C35b -- support attribution.**  C35 first exposed a telemetry bug:
  every generator column was attributed to synthetic layer `-1`, so its
  ranking could not guide an operator change.  C35b repaired column
  ownership.  On the focused depth-1 rows, the two candidate supports were
  `19.74299` and `19.20446`; layer 20 contributed `13.35696`/`10.73886`,
  followed by layer 22 at `5.66640`/`5.41626`.  Earlier layers were much
  smaller.  The verdict remained `UNKNOWN` (`18.055` seconds build,
  `28.180` seconds total), but the receipt justified moving the experimental
  stop to ADD20 instead of guessing another late ReLU.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c35_support_attribution_depth1_gpu40_iid2_20260729.summary.json`
  and
  `artifacts/hybridz_largecls_gates/gate1_c35b_support_attribution_depth1_gpu40_iid2_20260729.summary.json`.

- **C36 -- ADD20 build-only localization.**  Stopping at ADD20 retained 198
  survivor rows and produced cube uppers from `25.48853` to `60.67361`.
  It was still `UNKNOWN`; operator build alone cost `29.443` seconds
  (`30.057` seconds total).  This established a controlled prefix boundary,
  not a proof improvement.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c36_attributed_stop_add20_build_iid2_20260729.summary.json`.

- **C37/C38 -- GPU scheduling versus geometry.**  C37 processed 99 of 198
  eligible rows on GPU in `7.065` seconds, certified zero rows, improved zero
  supports, and never reached the focused rival because candidate order
  exhausted the budget.  C38 repaired that scheduling defect and did process
  objective row 149.  Its support was still positive at `8.76172`, dominated
  by layers 10 (`4.13311`), 5 (`2.53378`), and 14 (`1.85617`); again zero
  rows were certified and zero supports improved.  Thus the scheduler fix
  was retained, while “more rows with the same geometry” was closed.
  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c37_attributed_stop_add20_gpu40_iid2_20260729.summary.json`
  and
  `artifacts/hybridz_largecls_gates/gate1_c38_scheduled_stop_add20_gpu40_iid2_20260729.summary.json`.

- **C39/C40 -- exact phase at the localized boundary.**  C39 split one exact
  phase after ADD20 for rival 50.  Both children remained `UNKNOWN`, with
  focused cube uppers `29.65812` and `30.02543`; build cost `29.716` seconds.
  Moving the boundary to ReLU22 in C40 cost more (`45.853` seconds build),
  selected rival 40, and gave identical phase uppers of `32.54473`, so that
  boundary was closed.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c39_stop_add20_phase1_build_iid2_20260729.summary.json`
  and
  `artifacts/hybridz_largecls_gates/gate1_c40_boundary_relu22_phase1_build_iid2_20260729.summary.json`.

- **C41 -- bounded query feedback.**  A four-step, 20-second transaction
  targeted layers `[10,14,22,40]`, but hit its deadline and restored the
  baseline.  Its status is `FAIL_ERROR`
  (`deadline_fallback_baseline`/`QueryDualPipelineTimeout`), not a proof
  verdict and not an ordinary `UNKNOWN`.  No candidate was promoted.
  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c41_query_feedback_s4_t20_build_iid2_20260729.summary.json`.

The reusable lesson from C35--C41 is that attribution, target scheduling,
and geometric effectiveness are separate gates.  Correcting either of the
first two does not imply tighter bounds; a candidate must report the
targeted row's support before and after the change.

### C42--C47: branch policy and child-refinement stop-losses

- **C42 -- joint-gain groups.**  Four groups and 32 probe nodes found a best
  worst-child lower-bound gain of only `0.01648`.  After 225 nodes the run
  remained `UNKNOWN` with a 204-node pool (`75.736` seconds total).
  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c42_joint_gain_groups4_iid2_20260729.json`.

- **C43/C43b -- worst-rival branching.**  Focusing one worst property row
  reached 207 nodes with a 154-node pool and remained `UNKNOWN` in
  `72.119` seconds.  Extending the internal budget to 75 seconds reached 361
  nodes but was still `UNKNOWN`, and its `106.593`-second wall time exceeded
  the official 100-second limit.  C43b is therefore diagnostic only and
  cannot be used as benchmark evidence.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c43_worstrival_branch_iid2_20260729.json`
  and
  `artifacts/hybridz_largecls_gates/gate1_c43b_worstrival_branch_t75_over100wall_iid2_20260729.json`.

- **C44 -- contraction threshold.**  A `0.9` contraction target reproduced
  C43 exactly: 207 nodes, 154-node pool, and `UNKNOWN`.  Its split-depth
  histogram (`176` at depth 1, `1` at depth 3) showed that this policy did not
  alter the effective search.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c44_worstrival_contraction09_iid2_20260729.json`.

- **C45 -- fixed-tail child refinement.**  The 12-second micro gate made one
  refinement call and changed one layer, with 12 strict lower and 12 strict
  upper improvements.  Nevertheless it pruned no child and ended at the
  same 9 processed nodes/16-node pool, so fixed-tail refinement was closed.
  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c45_childrefine_tail8_micro_iid2_20260729.json`.

- **C46 -- immediate successors.**  Successor selection queried no rows and
  changed nothing because the chosen split was a terminal ReLU with no
  unstable successor.  This converted a silent no-op into an explicit
  eligibility failure, but did not improve the `UNKNOWN` result.
  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c46_childrefine_successors8_micro_iid2_20260729.json`.

- **C47 -- require a nonterminal split.**  The filter was applied twice with
  no fallback and exposed eligible layers `[3,5,10,14,22]`.  Refinement then
  selected downstream layers `[14,40]` and queried 256 objective rows in
  `0.163` seconds, yet produced zero strict lower/upper changes and again
  ended at 9 nodes with a 16-node pool.  The plumbing worked; its measured
  benefit was zero, so the line was stopped.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c47_nonterminal_successor_refine_micro_iid2_20260729.json`.

These runs rule out node count, branch focus, and refinement-call activity as
standalone success metrics.  The minimal promotion metric is a changed child
bound that either prunes a child or measurably reduces the live pool under
the same budget.

### C48: property-separable BaB finds one strict `S`, but is not promoted

C48 replaced one monolithic multi-property root with eight independent
property lanes.  On **CIFAR100-medium iid2 only**, the 12-second micro gate
found a candidate in lane 6/property row 67 after eight nodes.  The first
receipt did not export a counterexample and was therefore insufficient:
`artifacts/hybridz_largecls_gates/gate1_c48_property_forest8_micro_iid2_20260729.json`.
C48b added an internal replay and export:
`artifacts/hybridz_largecls_gates/gate1_c48b_property_forest8_strictce_iid2_20260729.json`.

The authoritative check is C48c.  It replayed the exported input with
ONNX Runtime on CPU, parsed and evaluated the raw VNNLIB assertions, and
required zero tolerance.  The receipt reports:

- true class 27, violating rival 68, raw property row 67;
- internal violation margin `0.06654183` and ORT margin approximately
  `0.06654263`;
- `replay_completed=true`, `ort_executed=true`,
  `raw_spec_evaluated=true`, `property_holds=true`,
  `zero_tolerance_holds=true`, and `valid_counterexample=true`;
- authority `onnxruntime_cpu_raw_vnnlib_zero_tolerance`, tolerance `0.0`,
  with no replay error.

Evidence:
`artifacts/hybridz_largecls_gates/gate1_c48c_property_forest8_ort_rawvnnlib_iid2_20260729.json`
and the strict authority receipt
`artifacts/hybridz_largecls_gates/gate1_c48c_property_forest8_ort_rawvnnlib_iid2_20260729.strict_replay.json`.

Therefore C48 has exactly one strict `S = SAT / FALSIFIED` result for the
CIFAR100-medium iid2 development instance.  It has **not** formally passed
the promotion gate or established benchmark-wide improvement.  It provides
no current evidence for TinyImageNet, for other CIFAR100 instances, or for
any `U = UNSAT / VERIFIED` case.  The historical `S` label agrees with the
receipt, but only the ORT plus raw-VNNLIB zero-tolerance replay establishes
this counterexample.  Any future promotion must preserve this strict replay
and demonstrate controlled coverage without turning one development
instance into a full-dataset claim.

### C48 strict-audit and bounded migration addendum

The post-C48 audit found two generic proof-boundary defects that had to be
closed before counting any SAFE-side diagnostic:

- an exact zero slack could be pruned even though TOP1/MARGIN semantics count
  an output tie as a violation, and a NaN could be dropped because both
  ordered comparisons were false;
- concrete counterexample replay ignored `LIN_POLY` and unknown input-spec
  kinds.

The repaired rule is fail-closed: an ALL-row proof is usable only when its
slack is finite and strictly above the float64 accumulated-rounding tolerance.
The same rule now governs ordinary dual evaluation, per-node BaB status, and
root row pruning.  Concrete replay checks BOX, L-infinity, LIN_POLY, and
LP-embedding constraints; incomplete and unknown kinds cannot pass.  Exact
zero, a `5e-12` near-boundary slack, NaN, infinity, missing constraints, and
an out-of-polytope candidate are controlled negative tests.  The expanded
12-module blast radius passes `171/171`.

C48 was then rerun with the audited code:

- iid2 again produced the identical candidate SHA256
  `3d76beec28042eaa4d0a8e79df829b8022b4353eccf45e922a30a2c145a67f45`
  from eight retained rival rows.  CPU ONNX Runtime plus raw VNNLIB at
  tolerance zero again accepted it; total wall time was `19.390` seconds.
  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c48k_strictfinite_tol_property_forest8_ort_rawvnnlib_iid2_20260729.json`.
- held-out CIFAR100-medium iid64 produced a second independently replayed
  `S`, margin `0.00459422`, in `18.283` seconds.  Root presolve retained only
  one row, so this validates the counterexample chain but is not evidence that
  multi-rival separation caused the solve.  Evidence:
  `artifacts/hybridz_largecls_gates/gate40_c48g_property_forest_sat_cifar100_medium_iid64_20260729.json`.
- CIFAR100-medium iid11 retained one row and remained `UNKNOWN` after 17
  processed nodes with 24 nodes live (`29.933` seconds wall).  Evidence:
  `artifacts/hybridz_largecls_gates/gate14_c48f_property_forest_sat_cifar100_medium_iid11_20260729.json`.
- CIFAR100-medium iid29 was discharged by root presolve under the repaired
  finite/strict/tolerance rule in `12.386` seconds.  This probe deliberately
  carries top-level `proof_authority=false`, so it is a SAFE-side diagnostic,
  not yet an official `U` receipt.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c48j_strictfinite_tol_property_forest_unsat_cifar100_medium_iid29_20260729.json`.

The CIFAR100-large iid118 and TinyImageNet-medium iid6 attempts reached the
float64 affine root propagation but could not allocate their next temporary
while another user process held about 54 GiB of the 95 GiB GPU.  They produced
OOM diagnostics, not verification verdicts, and are not counted against C48.
Repeating them under the same memory pressure is closed; the next attempt must
use an uncontended window or a toy-audited low-memory float64 root path.

Current decision: retain C48 as a promising experimental primitive, but do not
claim Gate-6 promotion.  Its causal evidence is one multi-rival development
solve (iid2); iid64 and iid29 exercise paths that presolve reduced to one or
zero rows.  Safe-side tightness work and the large-model memory gate therefore
remain separate required lines.

### C49: signed micro-RLT mathematical gate

C49 is an attribution-guided, factor-space degree-1 RLT candidate, not the
previously closed global RLT direction.  For a selected signed binary factor
`s in {-1,+1}` and one stored upper row `r <= 0`, it retains the original row
and appends both `(1+s)r <= 0` and `(1-s)r <= 0`.  Shared product factors use
the four-row convex hull of `v=s*q`; all generated coefficients are first
formed as exact `Fraction` values, then any binary64 coefficient-storage error
is added in L1 to an outward-rounded right-hand side.

The decisive duplicate-ReLU toy has two exact phase factors for two copies of
the same input.  Its ordinary independent Big-M LP relaxation permits
`max(y2-y1)=0.5`, although the exact graph has `y1=y2`.  The complete signed
lift changes that relaxed optimum to exactly `0.0`.  Removing the single
necessary row `rlt[1,0].plus` reopens the optimum to `0.5`, so the improvement
is caused by the intended two-sided geometry rather than solver noise.  All
four exact phase assignments preserve the base projection, 33 grid points
plus the four phase choices at zero satisfy every generated row, receipt
tampering fails, and malformed/capped/non-finite requests fail closed.  The
focused mathematical gate passes `11/11`.  It also covers commutative
binary-product sharing, global continuous/binary stable-ID collisions,
non-dyadic and `1e200` coefficients, a half-ULP coefficient, and half of the
least positive binary64 subnormal.  The receipt explicitly binds the
`s in {-1,+1}` factor convention to the solver transformation `s=2z-1`.

The first audit correctly rejected the initial prototype as production
evidence: it was not called by the verifier, used local ID allocation, did
not bind every matrix/RHS/stable-ID digest, and would lose process-local
Operator-HZ capabilities if applied post hoc.  Those defects were then closed
in the primitive itself: fresh IDs come from the global allocator, the live
validator re-hashes `c/Gc/Gb/Ac/Ab/b/Auc/Aub/ub` and both ID sets, the
ordinary matrix prefix must remain unchanged with zero new-column
coefficients, and a self-hashed receipt alone has no authority.

One boundary remains fundamental rather than an implementation defect:
after a selected signed bit is fixed, each product hull forces `v=s*q` and
the two RLT sides reduce to `2r<=0` and `0<=0`.  C49 can therefore improve
only the **parent binary relaxation before phase enumeration**.  It cannot
claim tighter fixed-phase children or any C38 early-row-prefix improvement.
Evidence:
`act/back_end/hybridz_tf/property_micro_rlt.py` and
`act/back_end/hybridz_tf/test_property_micro_rlt.py`.

### C50: DAG last-use root propagation unlocks the large families

C50 addresses the separate large-model memory blocker without changing any
bound arithmetic.  `DualTF.compute_forward_bounds` now counts unique DAG
consumer nodes and releases only its internal `lin_state`, `box_state`, and
`frame_dict` entries after the final consumer has completed every read.
Public per-layer `bounds_dict` entries required by the reverse dual pass are
never released.  Chain, residual fanout, duplicate predecessor, FLATTEN alias,
ASSERT, and alpha-path checks are bitwise identical to a retain-all hook in
float64.  The focused liveness/refine/property gate passes `15/15`; the
expanded blast radius passes `177/177`.

The same fixed C48 settings were then used once on each previously blocked
Gate-6 large-family pair while another user's process continued to hold about
54 GiB of the 95 GiB GPU:

- CIFAR100-large iid118 is `FALSIFIED` after one retained property row.
  Its independently replayed zero-tolerance margin is `0.55508981`.
  Peak allocation/reservation fell to `15.055/15.223 GiB`; process wall time
  was `47.06` seconds.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c50a_lastuse_c48_sat_cifar100_large_iid118_20260729.json`.
- TinyImageNet-medium iid6 is `FALSIFIED` from two retained rows.  Its
  independently replayed zero-tolerance margin is `0.83538676`.
  Peak allocation/reservation was `26.205/31.121 GiB`; process wall time was
  `86.21` seconds, including the comparatively expensive independent CPU
  replay.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c50b_lastuse_c48_sat_tinyimagenet_medium_iid6_20260729.json`.
- CIFAR100-large iid113 is root-presolve `CERTIFIED`: all 99 rows were removed
  under the strict finite-positive-slack rule.  Peak allocation/reservation
  was `18.880/33.018 GiB`, with `42.26` seconds process wall time.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c50c_lastuse_c48_unsatdiag_cifar100_large_iid113_20260729.json`.
- TinyImageNet-medium iid17 is `CERTIFIED` after the sole retained row 126 was
  discharged in nine nodes.  Peak allocation/reservation was
  `26.205/31.131 GiB`, with `69.36` seconds process wall time.  Evidence:
  `artifacts/hybridz_largecls_gates/gate1_c50d_lastuse_c48_unsatdiag_tinyimagenet_medium_iid17_20260729.json`.

Every probe records `ground_truth_loaded=false`.  Together with the audited
medium iid2/iid29 pair, the six fixed Gate-6 diagnostics now all agree with
their historical labels.  The three `S` cases have independent CPU ONNX
Runtime plus raw-VNNLIB receipts at tolerance zero.  The probe's top-level
`proof_authority` remains deliberately false, so the three `U` diagnostics
must not be presented as formal competition receipts and Gate-14 is not
automatically authorized.  C50 is nevertheless a real capability gain:
large/Tiny root propagation changed from repeatable OOM with no verdict to
four completed, correct fixed-sentinel outcomes under the same external
memory pressure.

### C51: bounded parent micro-RLT transaction and SAFE-only prefilter

C51 connects C49 at its only useful proof boundary without changing the
default verifier.  The Operator-HZ builder records the actual global rows of
exactly two property-selected exact ReLUs which share one explicit focused
rival.  For each bit it selects its own stored
`lower/x-branch/zero-branch` Big-M rows plus the other bit's lower row.  The
live transaction checks row tags, row uniqueness, stable binary IDs, and the
positive/negative Big-M coefficient signs before applying the lift.

The transaction runs after the complete ordinary `SparseHZono` is assembled
and before provenance, conditional-row capabilities, early-prefix receipts,
input replay data, and the constructive-nonempty token are attached.  Every
candidate matrix, count, digest, generated tag, ID, and provenance update is
staged before the builder state is committed.  A product-cap miss is a
complete no-op; malformed receipts, ID tampering, and sign tampering fail
closed.  With the feature omitted or explicitly set to cap zero, the complete
HZ and metadata are bit-identical.  The Operator-HZ focused gate passes
`6/6`.

The solver-side bridge uses the exact signed-to-relaxed coordinate change:

```
s = 2z - 1
c + Gc*xi + Gb*s
  = (c - Gb*1) + [Gc, 2Gb] * [xi, z].
```

The rounded center shift is enclosed by an independent long-double outward
guard.  SAFE and witness authorities are separate: a binary-relaxed LP may
contribute only an independently recomputed negative Lagrangian upper bound,
while its primal point can never become an `UNSAFE` witness.  The genuine
product-RLT negative-control toy changes the checked upper from `+0.4` to
`-0.1`; deleting the critical RLT row restores `+0.4`, and the fractional
`z=0.5` optimum remains explicitly non-witness.  The focused solver gate
passes `6/6`.

Finally, two default-zero configuration fields bound the parent experiment:
`property_micro_rlt_product_cap` and
`property_micro_rlt_parent_prefilter_seconds` (at most 10 seconds).  They can
be enabled only together for the depth-2 property-tail phase mode.  Before
creating any of the four exact-phase children, the verifier makes one
SAFE-only parent call with GPU candidates and mixture generation disabled.
SAFE promotion additionally requires no witness and complete rival-group
coverage.  `UNKNOWN`, exception, deadline exhaustion, or contract mismatch
falls back to the unchanged complete phase cover.

Two independent duplicate-ReLU controls now exercise the whole chain:

- direct Operator-HZ plus production LP: cap zero gives checked upper
  `+0.400000000000011` and `UNKNOWN`; cap 64 gives
  `-0.099999999999963`, one certified row, and `SAFE`;
- real `verify_once`: the parent has two binaries, certifies in one call, and
  creates zero phase children.  A mocked parent `UNKNOWN` or exception still
  creates and solves all four children.

The parent-prefilter module passes `9/9`; the expanded CPU blast radius passes
`249/249` in `8.541` seconds with no legacy verdict regression.  This is
controlled toy and integration evidence, not CIFAR100/TinyImageNet evidence.
No real iid2 or Gate-6 run is authorized until a cap-only eligibility receipt
reports the required product count and the same-instance stop-loss accepts
its build time, memory, and independently checked parent upper.

### C52: outward phase-cover and private SAFE-capability firewall

The first independent C51 audit found three proof-boundary defects before any
real model was run.  First, fixing multiple binary columns used a binary64 CSR
matvec and then directly updated `c`, equality `b`, and upper `ub`.  Exact
dyadic sums such as `1 + 3*2^-54` need not be representable; an inward-rounded
child could therefore exclude a genuine fixed-phase point.  Second, the
verifier checked that every returned child was `SAFE`, but did not first bind
the returned collection to every unique sign assignment or require a
solver-private SAFE proof object.  Third, conditional suffix planes were kept
as mutable parent dictionaries protected only by a token, so a joint
parent/child mutation could preserve the token while changing the proof
plane.

All three defects are now fail-closed:

- fixed binary shifts are accumulated as exact `Fraction` values; a
  non-representable output center or equality RHS gets its own outward-rounded
  continuous error factor, while an upper RHS is rounded only toward
  `+infinity`;
- the phase verifier requires exactly `2^d` unique canonical assignments,
  checks assignment/child/ID receipts, and calls
  `hz_verify_sparse_binary_phase_child` to reconstruct every live child
  matrix from the parent and exact assignment;
- conditional planes can be attached only by the private Operator-HZ producer.
  Guards, center, generator CSR, error, rival map, and source receipt are
  deep-copied, frozen, and bound by an immutable parent seal plus a separate
  child live hash;
- grouped `SAFE` is promotable only with `hz_objbound_safe_v1`, issued inside
  the solver after its final local-deadline check.  The private capability
  binds all live HZ arrays/matrices/IDs, the exact `C/t/group` call, tolerance,
  base discharge, proof stage, and complete group coverage.  Entry clears any
  stale capability.  Parent promotion requires checked base feasibility;
  phase children require the private sound-cover membership capability;
- only the `persistent_lp_lagrangian` stage with the independently checked
  binary-relaxation eligibility, factor-width, coverage, and proof-authority
  fields is attributed as `parent_binary_relaxation_safe`.  Exact-point or
  cube SAFE remains sound but is labelled by its real stage.

The phase relation is deliberately stated as an outward cover, not an exact
floating-point union:

```
each exact parent sign slice is a subset of its audited outward child.
```

That implication is sufficient for `all children SAFE => parent SAFE`.
Permanent negative controls cover cancellation, RHS shrink, short/duplicate/
swapped covers, missing child capability, forged SAFE stats, live HZ or
`C/t/group` mutation, stale capabilities, conditional parent/child mutation,
micro-RLT row/ID/hash mutation, and worker-future exceptions.  The two
phase-rounding examples use `-2^30 + 2^30 + 2^-24` and an exact remaining
upper RHS of `2^-54`; both failed under the old arithmetic and are contained
by the new children.

The expanded CPU blast radius now passes `277/277` in `10.732` seconds
(`12.50` seconds process wall), including Gate orchestration.  Gate receipts
also record `ground_truth_loaded=false`,
`reference_diagnostic_label_present=true`, and
`reference_label_used_for_verdict_or_pass=false`.  No real benchmark was run
during C52.  The next authorized action remains exactly one CIFAR100-medium
iid2 `cap=1` eligibility probe; iid2 has an independently replayed strict
counterexample, so any parent `SAFE` or final `CERTIFIED` is a P0 conflict,
not an improvement.

## C53--C63 directed parent packet and first real tightness signal

All C53--C63 real probes remain restricted to CIFAR100-medium iid2.  They are
parent-only diagnostics with `ground_truth_loaded=false`,
`diagnostic_only=true`, and `promotion_eligible=false`.  The known strict
counterexample is therefore used only as a conflict sentinel: none of the
following `UNKNOWN` results is converted into an `S` or `U` label.

### C53--C55: measure the real lift, then packetize it

C53 correctly performed no lift at `cap=1`, but exposed that the initial
eligibility receipt stopped first on a selected-source-nonzero cap:
`required=12,300`, `cap=4,096`.  C54 made requirement counting complete before
the no-op and recorded the actual full two-bit demand:

- `4,100` product factors;
- `12,300` selected source-row nonzeros;
- `16,416` generated upper rows if the complete two-bit lift were applied.

This scale was too large for an undirected first real experiment.  The
replacement is a **complete directed packet**, not a partial row sample.  The
`first` packet lifts the first selected bit against its own
lower/x-branch/zero-branch rows and the other bit's lower row; `second` is the
symmetric construction for the second bit.  Each direction requires exactly
`2,050` products, `6,150` selected source nonzeros, and `8,208` generated
rows.  A direction is useful only on one side of the duplicate-ReLU toy:
`first` closes the toy's lower gap and `second` closes its upper gap, while
the complete two-direction lift closes both.  Every fixed binary assignment
has the same relaxed range as the unlifted base, so this directionality is a
tightness property rather than an invalid phase restriction.  A single
direction remains non-promotable in this experiment; the symmetric direction
must also be measured.

C55a confirmed the `2,050` requirement with another `cap=1` transactional
no-op.  C55b and C55c then failed closed in the Gate receipt boundary,
respectively on the parent-only/operator binding and on
`receipt_sha256_content`.  Neither error run supplies geometric evidence.
After repairing the live binding and JSON-normalized long-list content hash,
C55d became the first valid applied receipt:

- `n_cont=54,507`, `n_bin=2`, and `n_ub=106,584`;
- `8,208` generated rows and `9,316,750` total constraint nonzeros;
- `24.1023` seconds of operator build and `28.8715` seconds wall;
- parent verdict `UNKNOWN`, with no certified property row.

C55e increased only the bounded parent allocation from one to five seconds;
it remained `UNKNOWN` (`25.6505` seconds build, `34.5385` seconds wall).
C55f then localized the cost before considering a matrix cache.  The operator
build took `24.3035` seconds.  Within the five-second parent transaction, the
cube took `0.0527` seconds, base-matrix materialization `0.2263`, HiGHS model
construction `0.1419`, basis warm-up `0.9400`, and the persistent LP
`4.8714` seconds for 19 completed rows and zero certificates.  Rebuilding the
matrix was not the dominant parent cost, so matrix-cache work was
deprioritized rather than pursued by guesswork.

Evidence:
`artifacts/hybridz_largecls_gates/gate1_c53_micro_rlt_parentonly_cap1_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c54_micro_rlt_requirement_count_cap1_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c55a_micro_rlt_first_packet_cap1_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c55b_micro_rlt_first_packet_cap2050_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c55c_micro_rlt_first_packet_cap2050_receiptfix_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c55d_micro_rlt_first_packet_cap2050_jsonsafe_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c55e_micro_rlt_first_packet_cap2050_lp5_cifar100_medium_iid2_20260729.summary.json`,
and
`artifacts/hybridz_largecls_gates/gate1_c55f_micro_rlt_first_packet_cap2050_timing_cifar100_medium_iid2_20260729.summary.json`.

### C56--C58: two CUDA stop losses and the CPU timeout root cause

C56 restricted candidate generation to the packet core and one scheduled
objective, but the CUDA attempt hit the `50.3013`-second outer hard timeout
without a usable worker receipt.  The generic full constraint-generation
follow-up was then disabled for this restricted scope; C56b still hit the
`40.1572`-second hard timeout.  Under the preregistered stop loss there was no
third CUDA retry, learning-rate sweep, or wider objective batch.

The bounded fallback uses a sparse CPU coordinate wavefront only on the
packet core; accepted multipliers still go through the independent full-frame
checker.  Its exact-scale synthetic kernel was fast, but C57 nevertheless
hit another `40.1619`-second outer timeout.  The cause was outside the
candidate kernel and outside proof arithmetic:
`_hz_candidate_support_attribution` built a full 106,584-row string mask once
for every one of the 8,208 unique generated tags.  That diagnostic-only path
performed roughly `8.75e8` string comparisons after the candidate had
finished and did not obey the candidate deadline.

The repair aggregates tags once in row order, making the receipt `O(rows)`;
it also gives attribution an explicit deadline and runs the independent
checker before proof-neutral attribution.  On the full synthetic
`106,584 x 54,509`, `9,316,750`-nonzero frame, attribution takes about
`0.05--0.09` seconds and the independent checker about `0.07--0.08` seconds.
C58 then completed in `29.4836` seconds wall (`20.2139` seconds build).
Candidate work took `0.1456` seconds on `cpu_packet_core`, but made zero
updates: restricted support stayed `21.2998401213`, the full checked upper
was `48.3197538255`, and the verdict remained `UNKNOWN`.  C58 proves the
timeout repair; it does not prove useful packet geometry.

Evidence:
`artifacts/hybridz_largecls_gates/gate1_c56_micro_rlt_first_packet_gpu_core_top1_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c56b_micro_rlt_first_packet_gpu_core_top1_nocg_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c57_micro_rlt_first_packet_cpu_wavefront_cifar100_medium_iid2_20260729.summary.json`,
and
`artifacts/hybridz_largecls_gates/gate1_c58_micro_rlt_first_packet_cpu_wavefront_linear_attr_cifar100_medium_iid2_20260729.summary.json`.

### C59--C61: causal source rows and objective/plane alignment

The generated-only core had a causal initialization problem.  In the
controlled chain toy, the generated row `w <= 0` has zero coordinate
violation at the cube maximizer and cannot seed a move.  Adding the recorded
source row `x-w <= 0` lets the wavefront traverse the two-row chain and
tightens the objective upper from `+0.9` to `-0.1`; the final vector is
accepted only after full independent checking.  This justified adding only
the live source rows named by the micro-RLT receipt, with strict schema,
prefix, tag, and count caps.

C59 tested the first packet's 8,208 generated plus four source rows; C60
tested the symmetric second packet with the same `8,212`-row cap.  Both
completed in about `0.15` seconds of candidate time and about 28 seconds wall,
but both made zero updates and left the same checked upper
`48.3197538255`.  Source seeding was causally necessary in the toy but not
sufficient under the generic-hardness schedule then in use.  That receipt did
not export the actual scheduled row, so C59/C60 cannot establish whether the
objective was causally aligned with the packet.

The selector had explicitly chosen final-ReLU layer-40 rows 8 and 49 for
focused rival 50.  Exported property group 50 is `[50,149]`: row 50 is the
direct baseline property plane, while row 149 is the cube-tighter
`query_dual_shared_suffix_add_projection` alternative stopped at ADD33.
The earlier generic-hardness schedule was not bound to this focus.  C61 did
bind the group to rival 50, but then chose row 149 by within-group cube
tightness.  Because that shared-suffix alternative can bypass the selected
layer-40 neurons, it again produced zero updates: restricted support stayed
`19.8037084016`, full checked upper was `42.3981990297`, and the result was
`UNKNOWN`.

This is an objective/plane mismatch, not an unsoundness event and not
evidence that the packet has no effect.  The corrected scheduler validates
the live micro-RLT status, baseline-plane count, declared group map, and
focused-group membership, then selects the direct baseline row `50`.
Selection remains candidate-only; all 197 other rows are deferred rather than
discarded.

Evidence:
`artifacts/hybridz_largecls_gates/gate1_c59_micro_rlt_first_packet_cpu_wavefront_source_rows_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c60_micro_rlt_second_packet_cpu_wavefront_source_rows_cifar100_medium_iid2_20260729.summary.json`,
and
`artifacts/hybridz_largecls_gates/gate1_c61_micro_rlt_first_packet_focus50_cpu_wavefront_cifar100_medium_iid2_20260729.summary.json`.

### C62/C63: symmetric nonzero signal, still no solved instance

With the same one-objective, 8,212-row candidate cap, baseline row 50 is the
first real micro-RLT objective to show a nonzero checked tightening signal:

| probe | packet | wall / build (s) | restricted support | improvement | checked dual attribution | full checked upper | result |
|---|---|---:|---:|---:|---|---:|---|
| C62 | first | `27.7289 / 18.7391` | `20.5587830756 -> 20.5347541447` | `0.0240289308563` | 8 nonzeros; category counters were not yet exported | `52.4043261836` | `UNKNOWN` |
| C63 | second | `27.7862 / 19.0036` | `20.5587830756 -> 20.5347541447` | `0.0240289308563` | 6 generated + 2 source + 0 other | `52.4043261836` | `UNKNOWN` |

Both directions used eight wavefront updates and eight selected constraints,
had zero candidate/checker error, retained complete 198-row coverage, and
certified zero rows.  C63's explicit `6 + 2` attribution is important: the
gain is not explained by reusing only the four ordinary source rows; six
generated micro-RLT rows carry nonzero independently checked multipliers.
C62 predates the category counters, so its exact generated/source split must
not be reconstructed after the fact.

The distinction between evidence levels is decisive.  The
`0.0240289308563` value is a **candidate tightness signal** on one
focus-aligned row, and the accepted multiplier has been replayed against the
full matrix.  The resulting sound upper is nevertheless about `52.4043`,
far above zero.  Therefore C62/C63 provide no SAFE proof, no new strict
counterexample, no solved instance, no validation-rate gain, and no evidence
yet for TinyImageNet.  They do not authorize a Gate-6 or full-dataset rerun.

The current focused CPU regression after the attribution, source-row,
objective-map, baseline-plane, deadline, and dual-category changes passes
`71/71`.  It includes the duplicate-ReLU exact/fixed-phase oracles, the
source-seeded chain, wrong-focus fail-closed cases, full independent
long-double checking, binary-relaxation SAFE-only/witness firewalls, and
coverage/capability checks.  These tests support sound continuation; they do
not turn the positive C62/C63 signal into a proof result.

Evidence:
`artifacts/hybridz_largecls_gates/gate1_c62_micro_rlt_first_packet_focus50_baseline50_cifar100_medium_iid2_20260729.summary.json`
and
`artifacts/hybridz_largecls_gates/gate1_c63_micro_rlt_second_packet_focus50_baseline50_cifar100_medium_iid2_20260729.summary.json`.

Decision at the C63 boundary: keep the sound packet primitive and the CPU
candidate/checker path, but do not broaden objectives or datasets.  At most
one preregistered same-row update-budget scaling may test whether the
`0.0240` signal grows materially.  If it plateaus, the next distinct
candidate must add a toy-audited, bounded residual/materialization bridge
between the property objective and packet rows, again with full-frame
checking; blind CUDA retries, cap sweeps, and full benchmark reruns remain
closed.

### C64--C66: direct-packet plateau and constraint-cone bridge stop loss

C64 performed the single authorized update-budget scaling on the same first
packet, focused baseline row 50, and complete independent checker.  Increasing
the coordinate budget from 8 to 32 updates increased checked dual support
from 8 to 18 nonzeros (`16` generated plus `2` source), but the restricted
support and improvement were bit-for-bit unchanged:

```
20.55878307557851 -> 20.53475414472224
improvement = 0.024028930856271558
```

The full checked upper remained `52.40432618361822` and the result remained
`UNKNOWN`.  Direct packet iteration was therefore closed.

The next candidate selected only already-live ordinary rows from the exact
base prefix: all ReLU rows at the packet's final layer and the two nearest
complete `add_materialize` forward/reverse block pairs.  It introduced no new
constraint or HZ set.  A property-conditioned coordinate wavefront jointly
optimized this bounded constraint cone and the packet, followed by a bounded
packet refinement; every multiplier still required the full original-matrix
long-double checker.

The exact causal toy has `y,w in [-1,1]`, the materialization band `y=w`, the
packet row `w<=0`, and property `y-1/10`.  Packet-only has zero gain and the
bridge-only legal multiplier remains at `+0.9`; together they reach the exact
Fraction/LP optimum `-0.1`.  Removing either side retains `UNKNOWN`.  The
focused regression after implementation passed `73/73`.  A production-size
synthetic frame with shape `106,584 x 54,509` and `9,316,750` nonzeros
selected `8,792` bridge rows and `17,000` total candidate rows, completed
`64+32` updates in `0.406` seconds, reached the 96-dual hard cap, retained
full coverage, and did not hit its deadline.

C65 did not measure bridge geometry.  The selector deliberately failed
closed because its schema check named the private primitive receipt rather
than the live Operator-HZ metadata schema.  It selected zero bridge rows,
ran the old packet-only path, and reproduced the same `0.0240289308563`
improvement.  The live-schema regression was corrected before another real
probe; reconstructing the exact live tag blocks from C65 then selected
`8,394` rows: `8,192` ADD-band rows and `202` final-ReLU rows.

C66 is the single valid real bridge probe.  It selected `8,390` ordinary
bridge rows after excluding the four source-row duplicates, for `16,602`
candidate rows total.  Candidate work completed in `0.7001` seconds without
deadline or checker error.  Its result was:

- `10` bridge plus `2` source nonzero duals;
- `0` generated micro-RLT nonzero duals;
- `64+32` bounded updates;
- restricted improvement `0.0240289308561`;
- full checked upper `52.40432618361835`;
- zero certified rows and final `UNKNOWN`.

The preregistered continuation gate required improvement above
`0.240289` (ten times the packet plateau) and simultaneous nonzero bridge and
generated duals.  C66 met neither.  The ordinary bridge and micro-RLT packet
therefore reach the same local relaxation face on this objective; they are
substitutes, not a useful synergy.  This closes further packet directions,
update budgets, bridge-depth expansion, and Gate-6 runs for this line.

Evidence:
`artifacts/hybridz_largecls_gates/gate1_c64_micro_rlt_first_packet_focus50_baseline50_steps8_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c65_constraint_cone_bridge_first_focus50_cifar100_medium_iid2_20260729.summary.json`,
and
`artifacts/hybridz_largecls_gates/gate1_c66_constraint_cone_bridge_live_schema_first_focus50_cifar100_medium_iid2_20260729.summary.json`.

### C67: shared-ID causal-family discriminator and PC-CBDE entry gate

The C66 dual is not evidence that the RLT rows are globally redundant.  Its
checked improvement differs from C62/C63 by only `1.28e-13`, and the first
joint stage already reaches that plateau with `10` ordinary bridge plus `2`
source multipliers.  The packet refinement records 32 coordinate updates but
finishes with zero generated multiplier and no additional support gain.  The
coordinate search accepts zero-gain moves and has no flat-face cycle
detection; its second stage also starts from a residual objective with a
fresh packet dual, so it cannot reduce or exchange the ordinary multipliers
chosen by the first stage.  Nonzero family counts are therefore replaced by
full-frame family ablation as the causal criterion.

C67 adds the first exact discriminator before changing the optimizer.  For
both signs of the property objective, a materialized output is connected to a
packet coordinate either through the same stable generator ID or through a
fresh wrong-copy ID.  Fraction enumeration and an independent HiGHS LP agree
on every family ablation:

- the complete shared-ID chain has exact upper `-1/10`;
- deleting the complete materialization family, deleting the packet family,
  or deleting both gives `9/10`;
- the same shapes, intervals, and row-family tags with a fresh copied
  generator also give `9/10` and cannot certify.

The complete shared case is independently checked on the full matrix and
uses nonzero materialization and generated multipliers in both objective
directions.  The wrong-copy case retains the property row and zero generated
dual.  The focused test passes in `0.025` seconds, and the expanded focused
suite passes `74/74`.

This freezes the next candidate as **property-conditioned causal cone
block-dual exchange (PC-CBDE)**.  It must select a minimal path by actual
row/column incidence rather than layer-number proximity, close complete
semantic blocks, warm-start the current full dual, permit both increases and
decreases of existing multipliers, bound zero-gain exchanges with cycle
detection, and retain the best independently checked state.  Selection and
optimization remain proof-neutral; only the full-matrix long-double checker
may authorize an upper bound.

The C66 static selector remains available for its controlled tests but is no
longer auto-enabled in the live micro-RLT path.  No further real probe is
allowed until PC-CBDE passes wrong-copy, family-ablation, row-permutation,
deadline/cap, seeded-rational, and production-shape synthetic gates.  The
single authorized iid2 probe then requires either at least `0.24` checked
improvement or a full checked-upper decrease of at least `1.0`, strict
improvement over both `without_generated` and `without_bridge`, candidate
time at most two seconds, and total wall time at most 35 seconds.  Failure
closes the direction without increasing caps, objectives, instances, or
datasets.

Evidence:
`act/back_end/hybridz_tf/test_gpu_dual_candidates.py` and
`act/back_end/solver/solver_hz.py`.

### C68: PC-CBDE incidence and block-exchange controlled gates

C68 implements the two proof-neutral primitives required by C67, without
connecting either one to a verification verdict.

The incidence selector searches the actual stored CSR row/column bipartite
graph from packet columns toward the focused property columns.  It never
traverses a generated `property_micro_rlt` row as an ordinary shortcut.
ADD tags are not treated as layer-nearest hints: when an ADD layer is first
reached, its complete forward and negated-reverse stored-float signature
multisets and allowed-row closure are audited.  Search then expands only the
causal equality-coordinate atom, normally one forward/reverse pair.  Thus a
`2048 + 2048` ADD tag block selects two causal rows rather than all 4096; a
two-ADD chain selects four.  A wrong copy in any coordinate of an accessed
block, a partial block, malformed tags, row/nnz/depth caps, or a deadline
returns the empty candidate.

The production-shape selector gate uses `106,584 x 54,509` with exactly
`9,300,000` nonzeros and a real `2048 + 2048` ADD audit.  A structure-only
int8 CSC avoids copying the float64 coefficients.  Independent reruns took
about `0.066` seconds with about `90 MiB` additional RSS, below the
preregistered `0.75` second / `128 MiB` caps, and returned only the two
causal ADD rows plus the source path.  The independent selector suite passes
`9/9`.

The block-dual primitive consumes explicit selector-closed semantic blocks
and a full-dual warm start.  Version one supports only upper rows and keeps
all multipliers nonnegative.  It allows an existing multiplier to decrease,
performs exact one-dimensional piecewise-linear line searches along sparse
semantic-union directions, bounds flat-face travel with stable-key cycle
detection, and always retains the best full-frame float64 candidate.
Deadlines and nnz/update caps return that retained state; the result always
has `proof_authority=false`.

Its exact coordinate-stall LP is

```
maximize x1
x1 + x2/2 <= 0       (materialization)
x1 - x2/2 <= 0       (generated)
x1,x2 in [-1,1].
```

The old coordinate wavefront stops at dual `(1,0)` with support `1/2`;
fresh packet refinement makes zero updates.  The block exchange simultaneously
reduces the first multiplier and increases the second, reaching
`(1/2,1/2)` and exact support zero.  Full/materialization-only/generated-only/
neither optima are respectively `0, 1/2, 1/2, 1`, agreeing with Fraction and
HiGHS.  A stable-ID wrong copy remains at `1/2`.  Row permutation, negative
warm projection, flat-cycle retention, deadline/cap, and existing-multiplier
decrease gates pass.  The dedicated suite passes `7/7`, the adjacent
candidate suite passes `19/19`, and 100 random upper-only frames never return
a candidate worse than the better of warm and zero.

The candidate-only integration composes incidence selection, a local
original-frame residual, bridge/generated/source semantic families, four
family ablations, and expansion back to the unmodified full row order.  On
the genuinely separated property/packet toy, every expanded candidate is
accepted by the existing full-matrix long-double checker.  Fraction, HiGHS,
and the checker agree on `full / without-generated / without-bridge /
without-both = 0 / 1/2 / 1 / 1`.  Wrong-copy and deleted or modified causal
paths fail closed.  The initial integration suite passes `5/5`.  The
integration remains outside `solver_hz` dispatch, so these results establish
causal tightening and composition correctness but do not yet claim a solved
benchmark instance.

Evidence:
`act/back_end/hybridz_tf/gpu_dual_candidates.py`,
`act/back_end/hybridz_tf/property_causal_block_dual.py`,
`act/back_end/hybridz_tf/property_causal_block_integration.py`,
`act/back_end/hybridz_tf/test_property_conditioned_incidence_cone.py`,
`act/back_end/hybridz_tf/test_property_causal_block_dual.py`, and
`act/back_end/hybridz_tf/test_property_causal_block_integration.py`.

### C69: property-forest SAFE promotion prerequisites

The separate C48/C50 audit found no property-separation algorithmic
soundness defect inside the existing trusted `DualSolver + BaB` boundary.
The diagnostic probe's top-level `proof_authority=false` is hard-coded; it
must not simply be flipped.  The missing work is a controlled multi-tree
gate and a tamper-evident runtime receipt.

A two-dimensional duplicate-ReLU toy now exercises the complete
`verify_bab_batched` path.  Root presolve strictly certifies four of six
rows and retains non-contiguous original row IDs `[2,5]`.  The two property
trees select different input axes, drain after six processed nodes, and end
`CERTIFIED` with two certified leaves per row and no frontier/depth drop.
A new terminal omission firewall requires every retained root row to have at
least one processed and certified node.  Deleting a complete tree therefore
returns `UNKNOWN` with `property_forest_incomplete_coverage`; swapping a row
ID is rejected before pool insertion.  Frontier eviction, maximum depth,
zero slack, and NaN also remain non-certifying.

C50 last-use release was compared with a retain-all no-op on a fanout/ADD
property forest.  Both complete runs are `CERTIFIED`; retained rows,
coverage, branch decisions, batches, and all three float64 dual-slack tensors
are bitwise identical.  Likewise, under `dual_alpha_eta`, `optimize=True`,
and eight Adam iterations, the joint `B=1,M=2` state for rows `[2,5]` is
bitwise equal to the two forest `B=2,M=1` lanes for slack, strict masks,
alpha, eta, and split signs.  Zero/NaN controls are also bitwise equal and
fail closed.  The property module passes `16/16`; its combined forward
liveness gate passes `20/20`.

The omission firewall is necessary but not sufficient for an authoritative
receipt: after one leaf certifies, another descendant of the same row could
still be silently lost while the simple coverage counters remain positive.
The planned receipt therefore adds per-row node conservation

```
processed = certified + branched
roots + children_minted = processed
```

plus the corresponding global sums, exact root-certified/forest-row
partition, encoded-row/input-domain/network/config/source/dtype digests,
zero pool and integer drop counts, canonical JSON with `allow_nan=false`,
and an independent validator.  Until those controlled tamper gates pass,
the probe remains diagnostic and no existing SAFE artifact is promoted.

Evidence:
`act/back_end/bab/bab.py`,
`act/back_end/bab/test_property_separable_bab.py`, and
`act/back_end/dual_tf/test_forward_liveness.py`.

## C81--C84: real Operator K4 stop loss and localized E2 replacement

### C81--C83: full-parent K4 reaches the real parent, then closes on pair cost

The Operator exact-ReLU path was first completed end to end on controlled K4
geometry: raw TOP1 property parsing, focused-rival binding, exact Operator
literal selection, persistent pair checks, exact Fraction replay, fresh cut
materialization, and a private one-use solver handoff all passed their toy and
tamper gates.  This removed the earlier ambiguity about whether the K4 idea
could reach the live Operator parent at all.

The preregistered real diagnostic remained exactly one
`cifar100_medium`, iid 2 build-only probe with no verifier verdict and no
reference label.  C82 built the parent in `23.6406` seconds with shape

```
output=100, continuous=52,657, binary=4,
upper_rows=98,974, equality_rows=0,
constraint_nnz=10,498,232.
```

The candidate-first pipeline used `9.7260` seconds wall
(`8.0754` seconds internal) but reached `downstream_timeout` in
`exact_k4_candidate`; it emitted no materialized cut.  Total probe time was
`35.3750` seconds.  C83 changed only the already-preregistered phase window
to its maximum 40 seconds.  The build took `20.5818` seconds and the phase
pipeline `17.0212` seconds (`15.2240` internal), but the identical stage still
timed out with no cut; total time was `39.3951` seconds.

The zero edge/pair counters in these fallback receipts are initialized
fallback values, not evidence that all six K4 pairs were checked and found
compatible.  What the two probes establish is narrower and sufficient for a
stop loss: repeatedly building/solving the complete 10.5M-nnz parent for a
six-pair closure does not fit the candidate allocation.  Increasing K,
deadline, iid, or running the six sentinels cannot answer that bottleneck and
is closed.

Both receipts have `ground_truth_loaded=false`,
`reference_label_used=false`, `hz_objbound_decide_called=false`, and
`proof_authority=false`.  Evidence:

- `artifacts/hybridz_largecls_gates/gate1_c82_candidate_first_k4_buildprobe_cifar100_medium_iid2_20260801.json`
  (file SHA-256
  `a6946152fd47b129ff91dffb4b0dc52e016952364c5c1e80e9cb0601601a4f3c`);
- `artifacts/hybridz_largecls_gates/gate1_c83_candidate_first_k4_maxwindow_buildprobe_cifar100_medium_iid2_20260801.json`
  (file SHA-256
  `d829234b3c2c48dffc5a5257609d6b06b7800851cfab78cc022d2c43b71a60cf`).

### C84: one localized top-2 pair with mandatory full-parent exact replay

C84 replaces the numerical proposal LP, not the proof.  Starting from the
two selected binary columns, four pattern-only CSR-to-CSC incidence blocks
construct cumulative `64 -> 256 -> 1024 -> 4096` row prefixes.  Every local
model retains all parent variables and original `[-1,1]` bounds; it removes
only rows.  A local infeasibility ray is zero-padded back into the complete
`upper_then_equality` frame, and an edge exists only if the existing sparse
`Fraction` checker replays that ray against the untouched full parent.

The first implementation exposed a real ABA defect: solving against mutable
caller arrays allowed a transient mutation restored before the terminal seal.
The corrected implementation first creates a complete no-alias, read-only
private semantic snapshot whose digest must equal the caller seal.  Incidence,
HiGHS, ray reconstruction, and both exact replays read only that snapshot;
the live parent is checked again at return.  Original ABA, nested ABA,
persistent mutation, mixed signs, equality positive/negative orientations,
non-symmetric upper/equality row order, a third unfixed binary, corrupt CSR,
high-degree postings, resource caps, full-width feasible counterexamples,
binary enumeration, SciPy MILP, and exact cut tightness are covered.  The
localized direct suites pass `28/28`.

The full-shaped positive discriminator embeds only

```
x + s <= 0
-x + t <= 0
```

inside a synthetic parent with the C82 dimensions and `10,498,234` total
constraint nonzeros.  For `(s,t)=(+1,+1)`, the candidate completed in
`2.1278` seconds, used a `0.4966` second private snapshot, selected exactly
two H0 rows/four source nonzeros, built the local LP in `1.40` ms, solved it
in `7.20` ms, and accepted the edge after full-parent exact replay.  Peak RSS
was `1,273,892 KiB`.  This proves that the localized path can produce, not
merely reject, an exact edge at the real parent scale.  It is still synthetic
and says nothing yet about a real CIFAR100/TinyImageNet edge or solved case.
The receipt is
`artifacts/hybridz_largecls_gates/localized_real_shape_embedded_edge_probe_20260801.json`
(file SHA-256
`91c538a2c59db22c768fa5e6c88e6f87be25c922459a67e5fbf641a17133dde9`).

The default-off Operator E2 adapter ranks exactly two nonzero phases by exact
cross-rival score, binds selection/subset/property/row tags/caps/source frame,
and invokes exactly one localized pair.  Independent review found two defects
before a real probe:

1. the disabled adapter still copied the full parent and rederived selection;
2. `require_materializer_source=false` proved a statement about the current
   mutable HZ without proving that it was still the Operator builder's
   original network-level parent.

Both are now closed.  Disabled returns a static checksummed result before
reading any caller object.  Enabled requires the owner-bound constructive
nonempty producer seal, a matching private-parent digest, and the exact closed
source-mode vector; stale `ub`, active prefix frames, full-input replay,
micro-RLT receipts, and query-dual metadata all fail before the localized
oracle.  The combined adapter, localized, adversarial, and Operator snapshot
suite passes `68/68`.  Frozen adapter SHA-256 is
`487e1ce45372120b6c1ae6c8b1d10063a7a647ae673f4c77c9143b6ba87f625b`;
its test SHA-256 is
`778f8f4c51916504b960bd869e2ff2433f28ada41735abb4f62888ba39bc9ebd`.

C84 remains candidate-only and `proof_authority=false`.  It does not yet have
an owner-bound atomic consume/materializer capability, and the current
Operator plus localized layering makes two private snapshots.  Therefore it
cannot enter production pruning.  The next and only authorized real action
is a verdict-free iid2 build-only measurement with the following fixed gate:

1. exactly one deterministic top-2 pair and an exact full-parent certificate;
2. a fresh diagnostic pair cut, never applied to the live parent;
3. independent long-double checked LP uppers before and after the cut;
4. positive baseline margin and at least `5%` relative upper reduction;
5. parent/tags/inputs unchanged, wall below 60 seconds, peak RSS at most
   2.5 GiB, and CUDA allocation at most 8 GiB; and
6. no ground truth, verifier verdict, timeout retry, iid change, or second
   real run.

No exact edge or less than `5%` tightening closes localized E2 immediately.
The next algorithmic fallback is then RBS-to-Adaptive-K4: apply the already
sound recursive residual-bound screen before exact-bit allocation and use a
same-layer deterministic reservoir only if a selected phase stabilizes.  The
same iid2 C5 evidence supporting that fallback is `11.7%` fewer constraint
nonzeros, about `29%` faster build, and property worst upper
`120.134 -> 69.256`; it is not permission to rerun the already closed C5+LP
verdict path.

### C77: phase-forest solver numbers receive live rival-bound receipts

The adaptive forest cannot reuse a numerically valid LP upper bound under a
different node, rival, ASSERT row, or wave position.  The toy solver adapter
now independently rechecks every LP result with the solver layer's
long-double Lagrangian upper checker, then binds the exact live result object,
HZ semantic digest, ordered property digest, raw ASSERT digest, node lineage,
complete solver configuration, deadline, and batch position to a
process-local single-use capability.  A copied result or receipt, relabelled
rival, reordered batch, changed number, stale token, duplicate consume,
partial batch, or expired deadline fails the complete wave closed.

This closes the C76 result-provenance gap but does not promote the adaptive
forest: serialized receipts remain diagnostic, `proof_authority=false`, and
the module is disconnected from verifier and BaB dispatch.  The dedicated
receipt suite passes `6/6`; the combined receipt, original PC-PCC, persistent
PC-PCC, and signed-support grouping set passes `56/56`.

Evidence:
`act/back_end/hybridz_tf/phase_forest_solver_receipt.py` and
`act/back_end/hybridz_tf/test_phase_forest_solver_receipt.py`.

### C78: exact phase-conflict clique is the first non-C49 discriminator

PC-PCC checks whether two property-worsening signed binary phases are jointly
infeasible in the complete parent HZ.  A cut is emitted only after every edge
of a proposed clique has an exact dyadic/Fraction Farkas replay against the
original CSR rows and variable bounds.  The resulting inequality is

```
sum_i p_i s_i <= 2 - k,
```

which is exactly the stable-set statement that at most one of the `k`
mutually conflicting signed phases can hold.  Stable binary IDs, polarity,
parent/property digests, source rows, caps, deadline, and a live single-use
capability are all bound.  Missing one edge, swapping an edge, using a
same-count replacement, changing a source row, reordering CSR storage, or
omitting a binary from an alleged full clique cannot authorize a cut.

On the controlled K4 discriminator, the lifted C49 relaxation has LP upper
`4/3`, while exact Fraction enumeration and MILP both give `1`.  PC-PCC
replays all six pair certificates and tightens the LP to `1`, crossing the
fixed `9/8` SAFE threshold without any child split.  On K7 it tightens
`7/3 -> 1`; deleting one edge gives exact/MILP upper `2` and correctly emits
no K4 clique.  This is genuine geometric gain beyond the C49 parent, but it
is still a candidate-only toy result and not real CIFAR/TinyImageNet gain.

Evidence:
`act/back_end/hybridz_tf/property_phase_conflict_clique.py` and
`act/back_end/hybridz_tf/test_property_phase_conflict_clique.py`.

### C79: persistent exact pair oracle passes the preregistered speed gate

The exact conflict check now builds one HiGHS 1.14 model, changes only the two
literal bounds for each query, reuses the basis, restores both bounds, and
then treats the numerical dual ray only as a proposal.  Every accepted edge
is replayed from canonical source CSR rows with exact `Fraction` arithmetic.
The source frame is sealed once, semantic CSR digests are independently
recomputed, numerical dust that rationalizes to zero is dropped, and
deadline/cap checks cover source traversal and exact arithmetic.  The
candidate remains disconnected from verdict paths.

The fixed K7 gate used one warm-up and five alternating-order pairs on CPU
cores `8-11`, `HZ_MILP_THREADS=1`.  Legacy median was `0.174744864` seconds
and persistent median was `0.041010164` seconds.  Paired speedups were
`[5.140054, 4.260977, 4.267898, 4.247510, 4.284607]`: median
`4.267898x`, paired-bootstrap 95% lower `4.247510x`, with zero result
fallbacks.  This passes the preregistered `2.00x / 1.80x / zero-fallback`
performance gate while preserving the K7 `7/3 -> 1` tightness result.

Evidence:
`act/back_end/hybridz_tf/persistent_phase_conflict_oracle.py`,
`act/back_end/hybridz_tf/probe_persistent_phase_conflict_oracle.py`,
`act/back_end/hybridz_tf/test_persistent_phase_conflict_oracle.py`, and
`artifacts/hybridz_largecls_gates/pc_pcc_persistent_pair_gate_k7_20260730.json`
(artifact SHA-256
`5509d815637e40de82ef3802c6156e8b80e811c4bbb347ded56eed48803998d1`).

### C80: raw properties and signed-support subsets are exact candidates only

The raw VNNLIB bridge no longer derives rivals through the production
floating parser.  It uses an independent bounded S-expression parser and
exact sparse affine `Fraction` algebra, requires canonical Real declarations
and strict ASCII V1/V2 indices, binds the source descriptor metadata and
SHA-256 before/after the bounded read, and compares the exact raw semantics
with live float32/float64 ASSERT tensors including dtype, shape, signed-zero,
and raw bytes.  Candidate and consumed-batch identities, receipt and rivals
objects, TTL, revoke, and single-use consumption are bound in a capped live
registry.  Replacing a batch or receipt and recomputing public digests,
class-swapping, Unicode digits, leading-zero indices, nonlinear terms,
underflowed constants, stale files, and mid-read mutation all fail closed.

The signed-support stage accumulates every stored binary64 coefficient as an
exact dyadic value.  Rivals are grouped only when their complete nonzero
signed signatures agree; exact-zero binaries are recorded as omissions and
never silently treated as fixed.  For each eligible subset, fresh
parent/property/group-bound literals are created and all `k(k-1)/2` pairs
are checked in the complete original parent with one persistent model.
Only a complete exact closure emits a cut.  The verifier reconstructs trusted
literals itself, requires exact built-in primitive fields at every nested
level, enforces pair/certificate uniqueness under the subset binding, seals
the source frame once, replays every certificate exactly, and reconstructs
the emitted HZ.

The raw-to-group-to-cut toy has four selected binaries plus one zero-effect
omission: six exact certificates tighten the LP to `1`, the omitted column is
zero in the cut, and removing one edge produces no cut.  A K7 21-certificate
verification takes about `0.0101` seconds median after removing repeated
parent/frame scans.  Raw, subset, and Operator-selector focused tests pass
`58/58`.  A read-only real property parse was also consistent, but no real
model solve was run and there is no solved-instance claim.

Evidence:
`act/back_end/hybridz_tf/raw_vnnlib_rival_adapter.py`,
`act/back_end/hybridz_tf/property_phase_literal_groups.py`,
`act/back_end/hybridz_tf/property_phase_subset_clique.py`, and their focused
tests.

### C74: property-forest live SAFE authority boundary completed

C71's serialized node receipt remains `proof_authority=false`: counts and a
self-hash cannot prove UNSAT.  Formal SAFE authority now comes only from the
original in-process `VerifyResult.CERTIFIED`, whose root rows were strictly
certified by the live dual solve and whose retained forest leaves all ended
in live `SolveStatus.UNSAT` with a drained pool.  Before accepting children,
the verifier additionally checks their actual proof domains.  Input children
must be one exact contiguous axis partition; ReLU children must enumerate the
complete newly fixed `{-1,+1}^k` phase cube while preserving inherited phase
state.  Deleting, duplicating, row-swapping, or replacing one child by a
same-count sibling therefore ends `UNKNOWN`, not SAFE.

An authoritative run snapshots the exact ACT graph, encoded ASSERT rows,
input domain, complete BaB configuration, each parameter dtype and shape,
PyTorch default dtype, actual root-bound dtype/device, solver identity, batch
request, and time budget.  Exact ONNX and VNNLIB SHA-256 values are computed
before the first source read, checked again after model synthesis and ACT
conversion, sealed into the verifier context, and checked a third time at
the terminal validator.  The verifier issues an opaque process-local
capability only after constructing the original terminal CERTIFIED result.
The registry binds that exact result object identity and consumes the
capability once.  A copied or hand-built result, changed status or
counterexample, stale token/network/config/source/dtype, edited receipt,
wrong-source attribution, second validation, or missing capability fails
closed.  Arbitrary code execution inside the trusted Python verifier process
is outside this boundary; the serialized seal is explicitly not a portable
signature and cannot re-authorize a saved JSON.

`act.hybridz_joint_gain_probe.v2` no longer hard-codes every top-level result
to non-authoritative.  It sets top-level authority only after either:

1. the live property-forest SAFE capability validates; or
2. a FALSIFIED candidate passes the existing ONNX Runtime plus raw-VNNLIB
   zero-tolerance replay.

The SAFE receipt says `proof_authority=true` only for that trusted live run;
the conservation and scheduling receipts remain false.  The exact
duplicate-ReLU multi-rival forest, root-presolve-only SAFE, child-partition,
forged/stale/tamper, strict-zero, NaN, frontier, depth, and C50 last-use
equivalence gates pass.  The property suite passes `22/22`; combined
forward-liveness plus property tests pass `26/26`, and the expanded adjacent
BaB/liveness set passes `35/35`.  No real benchmark was run while
implementing C74.

The three earlier SAFE JSON files for medium iid 29, large iid 113, and
TinyImageNet iid 17 cannot be upgraded retrospectively: their processes are
gone, their top-level authority was false, and no opaque live capability or
sealed source context survives.  After independent review and the main
focused regression pass, the only authorized production check is a replay of
the exact fixed Gate-6 set, with no iid, model, property, dtype, iteration,
batch, or timeout substitution:

- CIFAR100 medium iid 2 and iid 29;
- CIFAR100 large iid 118 and iid 113; and
- TinyImageNet medium iid 6 and iid 17.

Gate-14 is authorized only if all six new files have top-level
`proof_authority=true`, `ground_truth_loaded=false`, zero verdict/ground-truth
conflicts, no promotion error, OOM, or missing receipt, and total time below
100 seconds for every instance.  SAFE rows must carry a validated live SAFE
receipt; unsafe rows must carry the strict replay receipt.  Any failure stops
the ladder without retrying another iid or changing a budget.  Only a `6/6`
pass may advance to one frozen held-out Gate-14 manifest; the existing
`14 -> 40 -> 400` stop-loss rules otherwise remain unchanged.

Evidence:
`act/back_end/bab/property_forest_authority.py`,
`act/back_end/bab/bab.py`,
`act/back_end/bab/test_property_separable_bab.py`,
`act/pipeline/verification/hybridz_joint_gain_probe.py`, and
`act/back_end/dual_tf/test_forward_liveness.py`.

### C75: fixed Gate-6 stopped at 2/6 on inapplicable SAFE-promotion errors

The exact C74 Gate-6 manifest was started without changing the model, iid,
property, dtype, iteration count, batch size, or 12-second verifier budget.
The first case, CIFAR100 medium iid 29, completed `CERTIFIED` in
`12.413352` artifact seconds.  Its live property-forest capability validated,
the SAFE receipt has `proof_authority=true`, top-level
`proof_authority=true`, `ground_truth_loaded=false`, and the promotion-error
list is empty.  This is the first authoritative live SAFE outcome for this
fixed gate.

The second case, CIFAR100 medium iid 2, completed `FALSIFIED` in
`19.605805` artifact seconds.  Its zero-tolerance ONNX Runtime/raw-VNNLIB
replay validates the counterexample, so both counterexample and top-level
`proof_authority` are true and no ground truth was loaded.  That conclusion
is authoritative.  However, the pre-fix probe also called the SAFE validator
on this FALSIFIED result.  The resulting artifact consequently contains 12
nonempty, inapplicable `property_forest_safe_promotion_errors`, including
`result_not_certified`.  This violates C74's fixed no-promotion-error
criterion even though it neither disputes the strict counterexample nor
reveals proof unsoundness.  It is an artifact-orchestration/status-routing
defect.  Gate-6 therefore failed and stopped exactly at `2/6`; the remaining
four fixed cases were not launched.

The probe now routes SAFE promotion by terminal status.  Only
`CERTIFIED` plus `property_separable_bab` invokes the live SAFE validator.
`FALSIFIED` emits
`property_forest_safe_promotion_status="not_applicable_falsified"` and an
empty promotion-error list, discards any accidental live SAFE capability,
and derives authority only from strict counterexample replay.  `UNKNOWN`
similarly emits `"not_applicable_unknown"`, an empty error list, no
authority, and no capability.  Pure helper tests cover all three terminal
statuses, mismatched receipt controls, and capability removal; they pass
`3/3`.  This local schema/routing repair did not run a benchmark and does not
retroactively rewrite either artifact.

Per the preregistered stop-loss, iid 2 is not rerun after this repair, the
remaining `4/6` cases are not run, Gate-14 is not opened, and no timeout,
budget, iid, model, or manifest substitution is authorized.  Thus C75
preserves the iid 29 live SAFE evidence and iid 2 strict-CE evidence while
closing this production line at the first gate-policy failure.

Evidence:
`artifacts/hybridz_largecls_gates/gate6_c75a_live_authority_cifar100_medium_iid29_20260729.json`
(SHA-256
`ef0996404383b56be13ead98fd6784ba1cd15609f01bb57c2c935c4e608ca3c2`),
`artifacts/hybridz_largecls_gates/gate6_c75a_live_authority_cifar100_medium_iid29_20260729.time.txt`
(SHA-256
`d95643f6bb2b9b32e42bf9fe771cc1b69b09d5bd8c397e989469fefcf4229516`),
`artifacts/hybridz_largecls_gates/gate6_c75b_live_authority_cifar100_medium_iid2_20260729.json`
(SHA-256
`32c8ee09613d32df1c14a5a13edeb87fc90f4bef99ad401ac68582239c1de8df`),
`artifacts/hybridz_largecls_gates/gate6_c75b_live_authority_cifar100_medium_iid2_20260729.time.txt`
(SHA-256
`5307c17cbf2999a96d6e9ca2c3272d2ad9a3101e7737eb3cc4c54be690e13193`),
`artifacts/hybridz_largecls_gates/gate6_c75b_live_authority_cifar100_medium_iid2_20260729.strict_replay.json`
(SHA-256
`85a1921db72de0c8d38c186ea528479837e0e117c46be1a3ffb043a07430f3ef`),
`artifacts/hybridz_largecls_gates/gate6_c75b_live_authority_cifar100_medium_iid2_20260729.counterexample.npy`
(SHA-256
`3d76beec28042eaa4d0a8e79df829b8022b4353eccf45e922a30a2c145a67f45`),
`act/pipeline/verification/hybridz_joint_gain_probe.py`, and
`act/pipeline/verification/test_hybridz_joint_gain_probe.py`.

### C76: adaptive phase-forest binding audit passes candidate-only

C73's adaptive exact-binary forest has completed its promotion audit without
being connected to a live verdict path.  The first audit reproduced a
critical positional bypass: for thresholds `(50, 200)`, a callback could
reverse a naked upper vector from `(100, 0)` to `(0, 100)`, copy the separate
batch binding unchanged, and make both children appear SAFE.  The interface
now has no naked `rival_upper` field.  Every numeric value is carried in a
frozen `RivalUpperBound` with its stable rival ID and binding digest, and the
validator checks the exact ordered identity before comparing thresholds.
The binding digest covers the stable ID, exact float64 objective bytes,
exact float64 threshold bytes, and raw-ASSERT SHA-256; a separate property
digest binds the ordered rival batch.  Reversing the two identified bound
objects now fails closed with `bound_rival_id_mismatch`.  Objective,
threshold, ASSERT, and rival-order swaps also fail closed.

The callback/selector firewall now binds the complete live `SparseHZono`
semantics rather than only dimensions and remaining factor IDs.  Its digest
covers `c`, `b`, optional `ub`; shape, `indptr`, `indices`, and data for
`Gc`, `Gb`, `Ac`, `Ab`, optional `Auc`, and optional `Aub`; `col_ids` and
`bcol_ids`; and every dynamically present conditional metadata field,
including the actual private parent rows/seal/receipt and child-applied
metadata.  Stable IDs must be present, exact `int64`, nonnegative, and
unique, so unsigned wraparound and duplicate-ID controls fail closed.
Dense-center, CSR-data, actual-conditional-metadata, selector, and callback
mutations are detected against before/after semantic snapshots.

The depth comparison is now measured rather than inferred.  The test calls
the real fixed-depth enumerator for all 16 depth-4 children, validates every
child against its live parent, and solves every leaf with HiGHS: five leaves
are feasible with upper `1`, and eleven are exactly infeasible.  The
adaptive candidate performs eight child-bound attempts over four sequential
waves and terminates with five SAFE leaves.  A separate non-first-ID control
selects `bcol_ids[2]`, verifies that exactly this stable ID is removed from
both children, and records it in both lineages.

The dedicated candidate suite passes `11/11`; the main independent expanded
adjacent run passes `122/122`, and a focused adaptive/phase-cover/C49/binary
phase run passes `52/52`.  Independent review reports
`BLOCKER=0, HIGH=0, MEDIUM=0, LOW=0` for the candidate-only scope.
`py_compile` and whitespace/diff checks also pass.  No real CIFAR100 or
TinyImageNet benchmark was run, and the module remains disconnected from
solver, verifier, and BaB dispatch with `proof_authority=false`.

This does not promote the result to a production certificate.  An arbitrary
or buggy callback can still relabel a wrong numeric value with the correct
rival ID and binding; a live path therefore remains blocked on a trusted
solver capability, a receipt binding objective/threshold/ASSERT inputs to
each solver result, and independent numeric replay or equivalent validation.
The `8` versus `16` result is only a bound-attempt count, not a wall-time
speedup: the adaptive path has four dependency-ordered waves, while the
fixed leaves can run concurrently.  The three toy rivals are scaled copies,
full semantic hashing is `O(nnz)` and has no production cost measurement,
and there is still no evidence of independent multi-rival geometry, GPU
memory safety, or real large-class gain.  These are hard prerequisites for
any future live proposal, not reasons to broaden the present experiment.

Evidence:
`act/back_end/hybridz_tf/adaptive_phase_forest.py` and
`act/back_end/hybridz_tf/test_adaptive_phase_forest.py`.

### C73: adaptive exact-binary phase forest passes a non-C49 toy gate only

C73 reopens **only** the reuse question left by C72, with a discriminator
that is not the duplicate-ReLU geometry already solved by C49.  The current
verifier chooses one or two exact factors up front, enumerates every
assignment of that fixed depth, and solves all leaves.  C73 instead fixes one
stable `bcol_id` at a time, validates the two live children, prunes a child
immediately when all rivals are SAFE, and recurses only on UNKNOWN children.
It is therefore an adaptive exact-factor forest, not another blind depth-3
cover or a new relaxation.

The interface audit found reusable proof primitives but no existing live
adapter:

- `hz_fix_sparse_binary_assignment` can remove any selected subset with exact
  dyadic substitution and explicit outward roundoff;
- `hz_enumerate_sparse_binary_phase_cover(..., positions=(p,))` makes the two
  complementary children while retaining every other binary factor;
- `hz_verify_sparse_binary_phase_child` independently binds each child to its
  live parent and assignment;
- ordinary BaB already recomputes input- and ReLU-split nodes in tensor
  batches, but through the dual backend, not Operator-HZ; and
- `verify_once` still returns `hybridz_batched_not_supported` for `B>1`.
  `hz_objbound_decide` batches all property rows of one HZ, but there is no
  production cross-HZ batch.

The new exact discriminator is a seven-factor complete-graph stable-set HZ.
With `z_i=(1+s_i)/2`, `s_i in {-1,+1}`, it stores all 21 inequalities
`z_i+z_j<=1` and maximizes `sum_i z_i`.  Exact Fraction enumeration gives
integer maximum `1`.  Crucially, the **complete existing C49 lift** is applied
to every one of the seven factors and all 21 source rows, and its live result
validator passes.  The lifted parent LP upper nevertheless remains exactly
`7/3`; with threshold `9/8` it is UNKNOWN.  Along the all-inactive path, the
independent HiGHS/Fraction RLT uppers are

```
7/3 -> 2 -> 5/3 -> 4/3 -> 1.
```

Every active sibling has upper `1` and is immediately SAFE.  Thus the final
K4 UNKNOWN node has two individually SAFE children, and the complete
adaptive tree terminates with five SAFE leaves.  It evaluates eight child
bounds over four sequential waves, compared with sixteen leaf bounds for a
matched fixed depth-4 cover: a `2x` bound-count reduction and `68.75%` fewer
terminal leaves.  Node conservation is exact:

```
roots=1, children_expected=children_minted=8,
processed=9, certified=5, branched=4,
unresolved=active=0.
```

Three rival lanes are carried in every `(2 children) x (3 rivals)` wave.
They are positive scalings of the same clique objective.  This tests shared
rival binding and batching only; it is **not** evidence of independent
multi-rival geometry or real CIFAR/TinyImageNet gain.

Missing a child, duplicating one sign (a duplicated subtree plus omitted
complement), selecting the wrong position, swapping assignment/child copies,
omitting or reordering bound results, NaN, expired deadline, and depth/node
caps all fail closed.  The dedicated gate passes `6/6` in about `0.12`
seconds.  The combined adaptive, phase-cover, C49 micro-RLT, and binary-phase
suite passes `47/47` in about `1.57` seconds.  The module is disconnected from
solver/verifier/BaB dispatch and permanently returns
`proof_authority=false`.

The cost model prevents interpreting the `2x` count as a wall-time result.
For wave width `n_w`, `P` parallel child solvers, and `M` rivals,

```
T_adaptive ~= sum_w [
    n_w * (T_fix + T_live_audit)
    + ceil(n_w/P) * T_HZ(M)
].
```

The toy has four dependency-ordered waves, while a fixed cover can schedule
all sixteen leaves at once.  Full-matrix child projection and live audit may
also scan millions of nonzeros.  A future candidate would start with two CPU
workers, at most five HiGHS threads per worker, one breadth-first wave in
flight, immediate release of SAFE children, and the complete `M=99/199`
rival batch inside each child.  CPU construction/audit may run for the two
siblings in parallel; GPU objective work must initially remain serialized
and row-chunked because duplicating large sparse HZ state has not passed a
memory gate.  CUDA streams or `B>1` HybridZ dispatch are not assumed.

No real probe is authorized.  If a production adapter is ever proposed, its
six-instance stop-loss is preregistered as follows:

1. use exactly six fixed diagnostics (three known-safe and three
   known-unsafe, with ground truth used only for evaluation), identical root
   configuration, factor selector, caps, and deadline;
2. require full child live audits and node conservation on all six, no NaN,
   timeout masking, missing child, or SAFE/unsafe-ground-truth conflict;
3. require at least two safe instances whose complete C49 parent remains
   UNKNOWN but an adaptive child is strictly SAFE, and at least one additional
   certified safe instance versus the unchanged root baseline;
4. on each promoted instance, use at most half the child-bound count of the
   matched complete cover; aggregate wall time must improve by at least
   `1.25x`, peak RSS/GPU allocation must be at most `1.25x` the root path,
   and every run must stay inside the official instance budget; and
5. any failure closes the line without changing depth, factor order, iid,
   dataset, timeout, or rerunning the six.  Only a passing six-instance gate
   may advance to the existing `14 -> 40 -> 400` ladder.

Evidence:
`act/back_end/hybridz_tf/adaptive_phase_forest.py` and
`act/back_end/hybridz_tf/test_adaptive_phase_forest.py`.

### C72: pure joint multi-rival suffix is dominated; shared phase is reuse only

An exact negative discriminator closes the proposal that a joint
multi-rival suffix or predicate-disjunction hull could tighten the result
solely by coupling rival selectors.  For one fixed network relaxation
`P` and rival affine forms `f_r`,

```
max over conv(union_r {(e_r, z, f_r(z)) : z in P}) of t
    = max_r max_{z in P} f_r(z).
```

The maximum of the linear epigraph coordinate is attained at an extreme
rival branch.  A simplex mixture therefore cannot create cross-rival
generator cancellation, and a cover that only restricts several rivals
from violating simultaneously is invalid for the required unsafe `OR`.
Any stricter result must instead tighten the network relaxation inside each
branch, for example with an exact ReLU phase cover or RLT constraints.

The controlled duplicate-ReLU toy has `x in [-1,1]`,
`a=b=ReLU(x)`, and rival margins `a-b-1/10` and `b-a-1/10`.
Independent triangle copies give exact Fraction and HiGHS upper bounds
`(+2/5,+2/5)`.  The exact predicate hull remains `+2/5`, proving no joint
predicate gain.  A complete shared semantic phase split gives
`(-1/10,-1/10)`, matching exact MILP, but the same phase cover applied
separately per rival gives the identical bound.  Sharing reduces network
tree nodes from six to three; it does not add geometric tightness.  Fixing
only one copy yields `(-1/10,+2/5)`, while a fresh wrong-copy MILP has true
upper `+9/10`, so stable semantic-ID mismatches must fail closed.

This toy deliberately reuses the C49 duplicate-ReLU geometry: C49 micro-RLT
already tightens its parent from `+2/5` to `-1/10`.  It is therefore a
negative differential audit, not a new solved geometry.  Together with the
C22--C34 closure of blind deeper phase covers and the C42--C47 no-pruning
branch experiments, it closes pure joint suffix, predicate hull, and
common-across-rival residual elimination as benchmark directions.  Shared
phase trees may only be reconsidered as a fixed-budget reuse optimization
after a non-repeated toy shows an adaptive child SAFE prune and at least
twofold node reuse; no real probe is authorized from the present evidence.

The dedicated Fraction/HiGHS/MILP suite passes `6/6` in approximately
`0.02` seconds.  The candidate remains disconnected from solver, verifier,
and BaB verdict paths and has `proof_authority=false`.

Evidence:
`act/back_end/hybridz_tf/multi_rival_shared_phase_audit.py` and
`act/back_end/hybridz_tf/test_multi_rival_shared_phase_audit.py`.

### C70: PC-CBDE live gate and single real-probe stop loss

PC-CBDE was connected only to the already-explicit micro-RLT packet path.
Ordinary HybridZ remains unchanged because both micro-RLT and GPU-dual
budgets default to zero.  The live gate requires one scheduled objective, an
existing `cpu_packet_core` warm candidate, no C66 static bridge, `row_topk=0`,
valid full-frame tags, and at least `0.30` seconds reserved for the ordinary
outer checker.  Its own deadline is at most `1.75` seconds.

The live candidate uses all micro-RLT rows for CSR incidence but at most the
64 largest nonzero warm generated multipliers for optimization.  Any
unselected generated warm multiplier is set to zero before all four
ablations, so `without_generated` really removes the complete generated
family rather than silently retaining a truncated tail.  Full,
without-generated, without-bridge, and without-both candidates are each
expanded to the unmodified original row frame and checked by the independent
long-double upper checker.  Family causality and replacement use
scale-aware `512 * float64-epsilon` tolerances; approximately `1e-13`
same-face noise can no longer be called a strict improvement.  Even a
replacement is checked again by the ordinary outer certificate loop.
Errors, deadlines, `row_topk`, malformed tags, and missing paths return the
old packet candidate unchanged.  PC-CBDE itself permanently reports
`proof_authority=false`.

The live exact filter toy moves the old packet candidate from checked upper
`+0.9` to `-0.1`.  All three deleted-family controls remain at `+0.9`;
the outer checker then certifies the full candidate.  A 66-row generated
warm test proves that the two rows beyond the 64-row optimization cap do not
survive the generated-family ablation.  Bridge multipliers are attributed as
bridge rather than `other`.  Parent and phase-child stats use explicit
allowlists, and unlisted poison fields are dropped.

An independent seeded gate evaluates 256 fixed dyadic shared-ID/wrong-copy
pairs under random row permutations.  It checks 1,024 expanded candidates
with the full-frame long-double checker and compares 160 sampled cases
directly with HiGHS; exact Fraction oracles cover every seed.  All 256
wrong-copy cases fail closed with no candidate.  The combined PC live,
selector, optimizer, integration, seeded, and stats gate passes `67/67`;
the broader HybridZ focused blast-radius passes `100/100`.  The production
incidence gate remains below its caps at approximately `0.20` seconds and
`90 MiB`.

Only then was the single preregistered real probe launched on
`cifar100_medium`, iid 2, first packet, focused baseline row 50.  Its
continuation criteria were unchanged: at least `0.24` checked support
improvement or at least `1.0` decrease in the full checked upper; full
strictly better than both deleted generated and deleted bridge; all four
ablations verified; PC work at most two seconds; total wall at most 35
seconds.

The worker was hard-killed by the outer 40-second deadline during ONNX
conversion/model synthesis, before verifier entry.  Parent wall was
`40.1570` seconds and the enclosing run took `40.4502` seconds.  Consequently
there is no PC candidate, no ablation receipt, no checked upper, no verdict,
and no solved-instance gain.  Historical C66 completed the same instance in
`27.7327` parent seconds, so this is consistent with frontend/cold-start
variance, but the preregistered one-probe and 35-second rules are not waived.
No retry, cap increase, objective change, iid change, or dataset expansion is
authorized.  PC-CBDE remains a default-off, sound candidate and controlled
research asset; its real expansion line is closed.

Evidence:
`act/back_end/solver/solver_hz.py`,
`act/back_end/verifier.py`,
`act/back_end/hybridz_tf/test_gpu_dual_candidates.py`,
`act/back_end/hybridz_tf/test_property_causal_block_seeded_soundness.py`,
`artifacts/hybridz_largecls_gates/gate1_c70_pc_cbde_first_focus50_cifar100_medium_iid2_20260729.summary.json`,
`artifacts/hybridz_largecls_gates/gate1_c70_pc_cbde_first_focus50_cifar100_medium_iid2_20260729.time.txt`, and
`artifacts/hybridz_largecls_gates/logs/20260729T122850Z-0d7b534101aa_cifar100_medium_iid002.log`.

### C71: property-forest node-conservation receipt completed

The C69 omission firewall is now strengthened into an independently
validated `act.property_forest_node_conservation.v1` receipt.  Each retained
property row records roots, expected and minted children, processed,
certified, branched, active-pool count, frontier/max-depth drops, terminal
reasons, and integrity errors.  A complete SAFE forest requires

```
processed = certified + branched
roots + children_minted = processed
roots = 1
children_expected = children_minted
active_pool = 0
drops = 0.
```

The validator also recomputes global sums, compares the receipt with the
actual processed-node and remaining-pool counts, and rejects non-integer,
negative, non-finite, duplicated-root, missing-child, duplicated-child, and
row-ID-swapped states.  Expected fanout is bound when a branch is created, so
deleting one child cannot be hidden by otherwise self-consistent aggregate
counts.  Eight independent receipt-tamper controls fail closed.  Existing
frontier, max-depth, zero, and NaN controls remain `UNKNOWN`.

The receipt is necessary for coverage but remains
`proof_authority=false`; formal SAFE still has to arise from the live UNSAT
forest rather than from receipt metadata.  The property suite passes
`18/18`, combined forward-liveness plus property tests pass `22/22`, and the
expanded adjacent BaB/liveness set passes `31/31`.

Evidence:
`act/back_end/bab/bab.py`,
`act/back_end/bab/test_property_separable_bab.py`, and
`act/back_end/dual_tf/test_forward_liveness.py`.
