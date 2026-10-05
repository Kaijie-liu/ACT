# Neural-HZ Structure-by-Structure Goal Charter

Locked on 2026-08-31 for branch `redu-hz`, whose starting commit is
`f1bc0f16612bd3f2112f2970174ec2f7cc3bf5ac`. This charter governs the current
active research goal. It does not promote a candidate or change a score.

This is the governing specification for the goal; `README.md` explains the
workspace and `TRIAL_LOG.md` records evidence, but neither may silently weaken
this charter. A change to the objective, baseline, score semantics, promotion
gate, or prohibition requires explicit user authorization, a dated revision,
and a new sidecar SHA-256.

## Objective

Design a PLDI-level, genuinely nonconvex Neural Hybrid-Zonotope representation
for neural-network verification, and use a sequence of reusable structural
innovations to improve the frozen 13-family result beyond **1,870/2,413**.
The work proceeds one repeated network/HZ structure at a time: define the
mathematical identity, implement one state-based rule, prove it on focused
tests, demonstrate it across instances and families with the same structure,
and only then replay all affected families and the complete 2,413-instance
universe.

CIFAR100 and TinyImageNet are the first large-CNN research targets. A material
gain on both is a primary capability objective, but their results have an
independent score ledger and are never added to the 1,870 formal score.
The formal endpoint is 2,413/2,413 on one promoted path. Until that endpoint is
reached, a local gain, a crossed layer, lower memory, a faster component, or an
external-family gain is a milestone and cannot mark the overall goal complete.
No unproved intermediate result is reported as progress toward the score.

## Frozen baseline and score semantics

The sole formal baseline is:

- 2,413 total cases in 13 families;
- 1,063 CERT;
- 807 concretely validated ADV;
- **1,870 solved**;
- 269 UNKNOWN and 274 TIMEOUT.

The authoritative per-family vector, source hashes, witness provenance, and
historical exclusions are fixed in `BASELINE_LOCK.md`. The following rules are
invariants:

1. Every old CERT and every old validated ADV must remain solved.
2. The solved count of each of the 13 families must not decrease, even if the
   aggregate count would remain unchanged.
3. A new ADV counts only after replay through the concrete network and property
   with zero invalid witnesses. CERT must remain sound under the frozen
   semantics.
4. UNKNOWN, TIMEOUT, ERROR, intentional layer stops, partial replays, and
   disconnected capability runs count as zero formal gain.
5. The displayed formal score remains exactly 1,870 until one candidate hash
   completes the full 2,413 replay with per-family zero regression and at least
   one new CERT or validated ADV.
6. A formal score vector must come from one candidate source manifest, one
   configuration and budget, and one execution path. Results from incompatible
   arms cannot be unioned, even if every member was individually sound.

## Representation invariants

Every accepted innovation must preserve a genuine Hybrid Zonotope:

- continuous latent factors;
- binary, nonconvex phase factors;
- equality and inequality predicates;
- shared latent identity across graph branches; and
- a reversible path to concrete witness reconstruction.

It is out of scope to replace or silently relax the state into a Zonotope,
Constrained Zonotope, interval box, or another convex domain. Binary factors
cannot be pivoted away merely for compactness. Any capacity, numerical, model,
or proof precondition that is not established must fail closed to UNKNOWN.

The representation gain must come from the Neural-HZ mathematics or its exact
physical realization. Attack/PGD, branch-and-bound, input or phase splitting,
backward refinement, and dual-rescue paths cannot be credited as a Neural-HZ
breakthrough.

## Uniformity and relevance rules

Each candidate is one rule selected only from observable mathematical state,
such as operator geometry, phase stability, predicate support, liveness,
sharing, or a proved storage/work bound. It must never branch on family name,
model/file identity, instance id, a public label, terminal margin, historical
verdict, or a hand-built per-instance menu. Evaluation ids may select
pre-registered measurement points, but no such identity is visible to the
representation rule. Structural thresholds and budgets are frozen before the
first target result for a candidate version is inspected.

An LP or MILP may be the ordinary terminal decision procedure and may provide
diagnostic evidence after a run. LP status, primal/dual marginals, or an
infeasibility ray cannot select or repair the HZ representation in this goal;
that would be a solver rescue rather than a Neural-HZ representation result.

We target repeated structures with enough representatives to support a family
claim. Singleton numerical pathologies and extreme corner cases are not
optimization targets. They still remain soundness guards: the implementation
must either preserve them or fail closed, never trade soundness for headline
coverage.

## One-structure transaction

Only one structural breakthrough is the primary implementation target at a
time. Every structure receives a pre-registered card containing:

1. the repeated blocker and the structural cohort in which it occurs;
2. an exact set identity or sound abstraction theorem;
3. a uniform state-based trigger and explicit fail-closed rejection gates;
4. preservation arguments for latent identity, predicates, phase binaries, and
   witness reconstruction;
5. logical work, unique resident bytes, live-state size, wall time, and verdict
   metrics, with allocator omissions stated explicitly;
6. a target set, at least one same-structure shadow set, and zero-hit guards;
7. a uniquely named output, branch/commit/config/source hashes, and an immutable
   result record; and
8. success, rollback, and next-structure conditions fixed before the run.

The promotion ladder is strictly:

`math/equivalence tests -> target -> same-structure shadows -> family replay -> full 2,413 replay`.

Candidates are default-off and explicit opt-in through all shadow stages. A
component can be kept as a documented capability or performance result without
being enabled or changing the score. A negative result is closed and recorded;
it is not rewritten or hidden.

Expansion to the next layer, target, family, or structure is allowed only when
the current level passes its exactness/witness tests, physical representation
gate, resource gate, target condition, and registered guards. Soundness or
witness failure, any old-result regression, identity leakage into a trigger,
ERROR, a hard resource violation, or failure to obtain the required strict
representation reduction closes that candidate version. Thresholds are not
edited after observing the failure. The overall goal remains active; the next
hypothesis must be derived from the repeated measured blocker in the same
structural cohort or from the next registered cohort.

### Capability and speed gates

Capability promotion and pure speed promotion are separate, as authorized by
the previously selected F semantics. A capability candidate must preserve the
frozen solved vector, remain fail-closed, and pass the pre-registered
four-concurrent no-regression gate (`baseline_time / candidate_time >= 1.0`) on
the gate workload. The former `1.5x` single-request, `2.0x` four-concurrent, and
`1.8x` bootstrap requirements apply only to a pure-speed claim. Hardware,
concurrency, bootstrap method, timeout, and aggregation statistic are frozen
with the candidate record; favorable one-off timing cannot substitute for the
gate.

### Global representation gate

Calling an expanded object lazy or storing it behind a reference is not an HZ
simplification. The ledger separately reports logical work/expanded nnz and
unique reachable physical storage, including HZ value maps, equality and
inequality predicates, operators, biases, phase/bounds metadata, retained
caches, and measured peak memory. Python/allocator/workspace quantities that
cannot be exactly charged are stated as omissions and bounded or measured
separately. A simplification claim requires a strict reduction in the
pre-registered reachable physical metric, not merely a smaller local core or a
single faster timing.

## Ordered structural campaign

### S0 -- residual large-CNN nonlinear frontier (active)

Target the repeated Conv/BN/Add/ReLU structure in TinyImageNet and CIFAR100.
The rule separates stable-negative, stable-positive, and genuinely unstable
ReLU rows, keeps the stable-positive affine path exact, materializes only the
nonlinear frontier, and represents Conv as an exact implicit
kernel/geometry/row-mask operator. Shared residual ancestors and global latent
slots remain intact.

The fixed progression is:

1. TinyImageNet iid143 at ReLU36, then ReLU63, ReLU71, and the terminal
   property;
2. CIFAR100-large high-evidence targets iid166 and iid153, followed by the
   registered S6 guards;
3. same-rule shadows on SRI-ResNet-A, SRI-ResNet-B, CIFAR2020, and Collins RUL;
4. the formal Conv-ReLU cohort of the 13-family baseline.

The iid list selects measurement points; it does not enter the rule. Advancement
requires exact equivalence, physical reachable-state reduction, bounded work,
and no loss of an old solved verdict. The current known blocker is exact lazy
Conv composition work, so the next implementation decision must be based on a
uniform support/work bound rather than on iid143.

### S1 -- TLL symmetric ReLU graphs

Close the already measured signed-sharing/dead-graph candidate through its
remaining cross-family and full-replay gates. Its candidate `+12` on TLL is not
formal gain until the same candidate hash preserves the complete 1,870
baseline.

### S2 -- plain FC-ReLU liveness and exact factor simplification

Target Cora and ACAS Xu, with fully solved SAT-ReLU and Malware families as
zero-regression anchors. Rules must exploit repeated affine/predicate liveness
or exact shared expressions, not solver-specific instance repairs.

### S3 -- shared residual/Add/Concat identities

Target LinearizeNN and Cersyve first, then transfer only proven shared-latent
rules into residual ViT blocks. This stage must preserve branch ancestry and
must not duplicate a shared factor into independent convex copies.

### S4 -- exact discrete and pooling structures

After the Conv foundation is stable, target Max/AveragePool and quantized Sign
with explicit binary semantics on YOLO, VGG, and Traffic Signs cohorts.

### S5 -- smooth, attention, and gated-product HZ relations

Treat Sigmoid/Tanh/GELU/Sin, attention QK/Softmax, and recurrent gate products
as separate mathematical structures. Begin only after their independent
baselines and witness protocols are frozen. They cannot be approximated by
pretending that an ordinary ReLU or affine rule applies.

Later stages may be reordered only when the active structure is formally
closed by its pre-registered success or rollback condition and measurements
identify a more frequent blocker. Difficulty on one iid is not sufficient.

## External-family ledger boundary

Every non-13 family has a separate manifest hash, model/spec hash set, baseline
verdict vector, witness protocol, and zero-regression gate. External results
are reported by family and may establish generality, but they are never summed
with 1,870/2,413.

Three score namespaces are strictly disjoint:

1. the formal 13-family `1,870/2,413` ledger;
2. the historical large-classification `59/400` aggregate; and
3. the current VNNCOMP2025-root CIFAR100/TinyImageNet universe and all other
   external-family ledgers.

They cannot be joined by arithmetic union or by iid. Disconnected results and
intentional intermediate-layer stops have formal gain zero in every namespace.
Before any external family is promoted, its ledger must freeze the denominator,
per-instance baseline verdict vector, complete instance/model/spec file
manifest, timeouts, hardware, source/configuration hashes, and concrete ADV
validation against the original model and property. Without this tuple it
remains capability evidence, regardless of a historical aggregate.

The current VNNCOMP2025-root manifests are:

- CIFAR100: `aa656d7a73529ba7c41b5618440f543ba4677418bb44115d384b644cc034f9ee`;
- TinyImageNet: `188058624df1122f32295f99d83380485a7d736212555a5e8214204459c22b7e`.

They are incompatible with the older `/data1/Kane/ACT/data/vnnlib` copies.
Per-iid claims, normalized rows, or the historical 59/400 aggregate cannot be
transferred between those universes. Until a full 400-row vector is frozen,
the current isolated subset is a capability ledger only.

## Data and experiment hygiene

`/data1/Kane/HyZor` is read-only historical evidence. No model, log, table,
witness, or verdict there may be overwritten or repurposed as a new result.
All new outputs live below `experiments/neural_hz_20260831/`; result filenames
are unique and use exclusive-create semantics. Every run records enough source,
configuration, benchmark, and environment provenance to identify its exact
candidate.

An unfinished authorized run may continue in the background and retain its own
result automatically. While such a run is active, files covered by its source
hash are frozen. Its output is interpreted only after process exit and checksum
capture; a long or failed run is recorded honestly rather than silently killed
or replaced.

The current dirty/untracked worktree is permitted for isolated research only.
Before any promotion claim, the candidate must receive a complete immutable
source/dependency/environment manifest; the starting Git commit alone does not
identify the modified implementation. Logs, result, exit status, checkpoint
where applicable, and hashes are retained by exclusive or atomic creation.

## Definition of progress

A structural step is meaningful when it gives at least one of:

- a new CERT or concretely validated ADV in its frozen ledger with zero
  regression;
- a proved exact representation reduction that crosses a previously measured
  capacity boundary on multiple same-structure instances; or
- a reusable theorem/implementation primitive whose equivalence, resource
  bound, and cross-structure utility are demonstrated.

Only the first item can ultimately raise a score. Pure speed, one disconnected
counterexample, or a smaller local object without a smaller reachable physical
state is evidence, not a promotion. All positive and negative evidence remains
in `TRIAL_LOG.md` and the corresponding immutable result files.
