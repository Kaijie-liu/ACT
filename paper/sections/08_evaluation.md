# Evaluation

The evaluation asks whether route conditioning expands the verifiable domain,
which component provides that expansion, and whether the same analysis layer
survives changes in router geometry, gate semantics, and expert scale. We keep
confirmatory endpoints immutable, separate development from holdout cohorts,
and label every later closure or engineering rerun as such.

## Research questions

**RQ1: Candidate precision.** Does correlation-preserving router analysis
strictly reduce the candidate upper set relative to IBP and an ordinary
zonotope when more than one route can be feasible?

**RQ2: Structural decomposition.** On route-unstable regions, how much does
per-route conditioning reduce the maximum simultaneous binary width relative
to a monolithic formulation?

**RQ3: Certificate yield.** How often can the staged verifier prove an output
property when route invariance is false or cannot be established? What fraction
comes from gate elimination and what incremental fraction requires F0?

**RQ4: Retained path information.** Does constraint-aware guarded support
eliminate binaries and change paired solver coverage? Does coordinate boxing or
property compilation preserve the same benefit?

**RQ5: Backend and scale transfer.** Can route specialization turn routed
models rejected by a general verifier frontend into static programs accepted
by the same verifier, and does that produce certificates on official-scale
RT-ER and AdvMoE checkpoints?

**RQ6: Scalability and cost.** How do lazy route-set enumeration, exact-support
big-M tightening, and MIP-start submission affect solved sets, binaries, model
builds, solve time, and completeness as (E) increases?

**RQ7: Artifact applicability.** Are published theorem assumptions,
preprocessing semantics, represented perturbation sets, and stateful-layer
evaluation modes established by the released artifacts? How sensitive is the
route-invariance applicability set to radius and initialization?

## Subjects and data

The verification-scale subject is an eight-expert selected-softmax top-2 MLP
trained on official torchvision CIFAR-10. The balanced coefficient-0.10
checkpoint is the frozen structural pilot; the coefficient-0.05 model is an
accuracy-balance control and the collapsed coefficient-0.01 model is used only
as a negative load example. Checkpoint selection uses the deterministic
validation split, not the test set.

RT-ER is an official-code, paper-configuration reproduction on CIFAR-10. The
exact dependency pin is retained as a failed contemporary-hardware probe; the
executed model is separately labelled Blackwell-compatible dependency
reproduction. Its four-way affine router remains byte-identical across
checkpoints under the released training script. TinyImageNet is used only for
the frozen official-construction router census and preprocessing-semantics
audit, never for end-to-end expert certification.

AdvMoE is a third-party learned-router target with two shared deep paths. Its
repository lacks a license and a complete dependency specification; no source
is copied into ACT and checkpoint redistribution is not assumed. The official
ordered CIFAR-10 test set is used for the K=20 initialization route-share audit
under both eval/default-running-statistics and registered train/current-batch-
statistics semantics. Training and final-checkpoint verification remain
pending.

Every dataset archive, split, checkpoint, preprocessing product, runtime, and
result file is identified by hash in the artifact manifest. Raw binary
artifacts are stored outside Git under `/data1/Kane/MOE`; tracked manifests
bind them to the corresponding code and configuration.

## Baselines

The route-invariance baseline first proves that one route is the only legal
route over the perturbation set and then invokes the same downstream expert
backend and budget as Route A. It is a MetaMoE-style reimplementation, not a
claim to execute unavailable author code.

Candidate baselines are IBP and an ordinary zonotope upper set. The exact-
router HZ label is used only for an unrelaxed reachable router. Structural
comparison uses a monolithic HZ/MILP encoding with the same support-derived
bounds. Guard representation is compared across retained HZ constraints,
guarded coordinate hull followed by CROWN, the original box followed by CROWN,
and the sound positive-margin eta property reduction. Published RT-ER theorem
formulas are instantiated through explicit constant providers; empirical
gradient constants are diagnostic and never labelled certified.

Alpha-beta-CROWN is a commodity expert backend rather than a method competitor.
We record whether the dynamic model frontend is rejected and whether each
route-specialized static model is accepted. All open-source baselines are pinned
to commits. Unavailable or non-executable systems appear only in the artifact
case series and are not assigned synthetic performance numbers.

## Cohorts and radii

The development cohort consists of the first 100 deterministic clean-correct
CIFAR-10 inputs. Confirmatory ranks 100--199 were not used to design F0,
guarded support, or the solver taxonomy. Fixed-radius census rows use

\[
  \epsilon\in\{0.25,0.5,1,2\}/255.
\]

The end-to-end confirmatory endpoint uses one preregistered route-boundary
radius per sample. The exact router feasibility oracle returns a strict
bracket and the primary radius is 1.05 times its upper endpoint. The original
overall denominator is all 100 inputs, including the 24 for which no boundary
was found through the frozen 4/255 cap. Experiment 1D and all later solver
reruns remain separately labelled closures; they never overwrite the 56/100
confirmatory solved rate.

Official-scale RT-ER rows use the frozen grid

\[
  \epsilon\in\{0.5,1,2,4,8\}/255.
\]

AdvMoE uses the same five radii for the final two-path table. Initialization
attacks and numerical probes are diagnostics, not prevalence cohorts.

## Verdicts, numerical requirements, and statistics

`SAFE` requires every legal route branch and every property row to have a
validated positive lower bound under the registered tolerance and outward-
rounding policy. `UNSAFE` requires a concrete input within the represented box
that replays through the complete routed model and violates the property.
Solver limits are `TIMEOUT`; semantic or relaxation incompleteness is
`UNKNOWN`. Weighted per-expert violations cannot become `UNSAFE` without a
full-model witness.

Candidate reduction and unique certificate proportions use clean samples as
clusters. We report Wilson intervals for binomial proportions and sample-
cluster bootstrap intervals where several radii share one input. Binary-width
ratios use median, IQR, and 90th percentile. Guard support uses a paired 2-by-2
transition table and exact McNemar/binomial test; runtime effects are reported
separately from solved coverage. Every reported unsafe result is independently
replayed.

The certificate identity contains both the requested real radius and the
represented tensor set. A positive bound over a float32 singleton produced by
an ULP-scale request is not a real-ball certificate. Runtime, preprocessing,
dtype, solver feasibility and integrality tolerances, positive-margin
tolerance, and outward-rounding policy are frozen manifest fields.

## Completed verification-scale results

The confirmatory candidate and width results answer RQ1 and RQ2. Among 86
route-unstable fixed-radius rows, exact-router HZ is smaller than IBP on 83
(96.5%) and smaller than the ordinary zonotope on 75 (87.2%). Conditional on
multiple feasible candidates, the route-conditioned-to-monolithic structural
width ratio has median 0.430 and 90th percentile 0.530.

For RQ3, the immutable confirmatory endpoint solves 56/100 inputs and fails its
preregistered 60% overall solved-rate gate. Boundary applicability is 76/100;
conditional coverage is 56/76. Thirty-six inputs are route-changing SAFE:
five by gate elimination and 31 by F0. F0 resolves 43/60 base semantic-
incompleteness cases, including 12 full-model-replayed unsafe inputs. The
explicit route-invariance baseline solves 12/100 under the same downstream
backend and budget, while Route A plus the labelled closure solves 68/100; all
36 route-changing certificates are unique to Route A. This is a coverage-cost
comparison, not a speedup claim.

For RQ4, retained guarded support reduces binaries from 10,076 to 8,466 in the
confirmatory accounting and closes the LP, MILP, and structural elimination
identity. Its paired coverage improvement is reported with the frozen McNemar
test. On the adapter cohort, guarded coordinate boxing and the original box
produce the same certified set, while eta reduction certifies none. The result
supports the narrower statement that guard value depends on a representation
that retains the path constraint.

## Table 1: official RT-ER comparison [pending]

The official-code compatibility target is now concrete. Seed 0 completes all
130 epochs but lands at 34.22% ordered-test SA and 32.70% independently replayed
PGD-50 RA, missing the frozen paper-reference intervals by 43.59 and 36.39
percentage points. The endpoint audit has zero issues. The full trajectory and
paper/source configuration audit are reported separately because neither low
accuracy nor a configuration ambiguity is an output certificate. Seed 1 is the
registered follow-up required before pipeline-level reproduction wording; B3
retains seed 0 as the released-pipeline product regardless of that outcome.

The main official-scale table has one row per registered radius and the
following columns. A row is complete only when every count has an independent
audit, all unsafe outputs replay, and the common expert backend and budget are
identical between the first two method columns.

| Radius | Exact route-stable fraction | Route invariance + CROWN: certified / time | Route A + CROWN: certified / time | HZ reference subset | Theorem 5.4 sound / empirical provider state | PGD-50 upper bound |
|---:|---:|---:|---:|---:|---:|---:|
| 0.5/255 | pending | pending | pending | pending | pending | pending |
| 1/255 | pending | pending | pending | pending | pending | pending |
| 2/255 | pending | pending | pending | pending | pending | pending |
| 4/255 | pending | pending | pending | pending | pending | pending |
| 8/255 | pending | pending | pending | pending | pending | pending |

The table must additionally report candidate-count distributions and the cost
of verifying all feasible experts. The 8/255 row is an applicability extreme,
not the sole headline; the non-degenerate comparison is expected at smaller
radii. Interim B1 accuracy and router census values cannot populate this table.

## AdvMoE two-path evaluation [pending]

The final AdvMoE checkpoint evaluation records clean accuracy, route share,
signed router-score offset, selected-margin distribution, and load entropy.
Intermediate checkpoints use the same fields under both eval/current-running-
statistics and train/ordered-test-current-batch-statistics semantics to test
whether supervised router training breaks the initialization collapse. The
train-mode row is a co-batch diagnostic, not a replay of the literal augmented
training stream. First-order diagnostics
cover the full test set; strong PGD uses a frozen deterministic subset; CROWN,
alpha-CROWN, or beta-CROWN closure is restricted to the final checkpoint and a
registered subset.

The end-to-end table verifies route-stable inputs through one static path and
route-uncertain inputs through both static paths. A router-independent two-path
row is always included because (E=2). Each radius reports route-invariance
coverage, two-path Route A coverage, full-model attack witnesses, per-path
runtime, and the guarded-cell ablation. The table is complete only after the
literal dynamic model rejects or accepts under the recorded frontend, both
specialized paths are accepted, all concrete equivalence tests pass, and every
unsafe result replays through the original routed model.

## Scaling and solver engineering [pending]

Lazy route-set enumeration is evaluated at (E\in\{4,8,16,32,64\}) under
frozen top-(k), router geometry, and budgets. Each row records feasible sets,
completeness, model builds, solves, no-good cuts, MIP-start submissions,
selector binaries after exact-support tightening, wall time, and peak memory.
Paired start/no-start runs measure observed effect without claiming that HiGHS
internally used an accepted start. All 30 registered conditions completed and
passed independent audit. The all-tied no-start family scales from 6 sets in
0.0053 seconds at E=4 to 2,016 sets in 50.23 seconds at E=64, while the E=64
one-set control takes 0.0616 seconds. Partial MIP-start submission has median
paired ratio 1.128 and therefore provides no observed speedup.

The exact-support big-M study and monolithic baseline use the frozen 20-row
cohort. Big-M is evaluated only where it is consumed: router membership
feasibility. The true monolithic baseline uses one bounded-homogenized
disjunctive MILP per property over all feasible guarded F0 pair branches; it
shares Route A's gate-range/McCormick semantics and changes only decomposition.
Exact support reduces membership selector width from 657 to 631 on the frozen
20-row cohort, but its median total-time ratio is 7.142 and it produces no node
reduction; the fast sound bound therefore remains the default.
The earlier incremental property-MILP rerun remains a negative engineering result:
model reuse accelerated LP hull construction by 15.03 times but did not speed
search-dominated property MILPs.

The true monolithic run solves 8/20 rows (6 SAFE, 2 replay-validated UNSAFE),
whereas the frozen staged Route A reference solves 12/20 (10 SAFE, 2 replay-
validated UNSAFE). Five rows are solved only by Route A and one only by the
monolithic formulation (exact paired binomial p=0.21875). We therefore report
the four-row coverage difference descriptively, not as statistically
significant or set-inclusion dominance. The 11,007.57-second monolithic and
5,151.10-second Route A totals are also descriptive because the runs were not
interleaved.

## Artifact applicability [partly complete]

The RT-ER artifact case study audits theorem instantiation, model semantics,
training semantics, and verifier consumption. The exact affine oracle gives an
official-construction applicability curve over 20 seeds; at 8/255 the route-
invariance set is empty for 18 seeds and contains at most two of 10,000 inputs
for the other two. TinyImageNet supplies an independently labelled census and
preprocessing-semantics audit, not an expert result.

AdvMoE initialization provides a distinct artifact finding. At the default
seed, every official test image selects expert 0. Across K=20 initializations,
13 eval/default-stat routers and 8 train-batch-stat routers are exactly
collapsed; median maximum load remains 100% and 99.305%. Thus local boundary-
scale ratios are confounded, and the identity of an init function must include
BatchNorm mode and statistics.
Sparse CROWN reduces IBP's dimensionless relaxation inflation by 5.17--5.36
times but leaves a residual near (10^{11}). No-flip attacks through epsilon 1
are non-proof diagnostics. These results motivate checkpoint route-share
telemetry and the router-independent two-path table; they do not substitute for
the trained result.

## Completion criteria

The evaluation is paper-complete only when the official RT-ER table, AdvMoE
trained two-path table, monolithic comparison, and (E)-scaling study have
zero-issue independent audits; all unsafe rows replay; no pending cell is
silently removed; and the immutable confirmatory failure remains visible. A
negative or null result satisfies completion if the registered experiment ran
to its endpoint and its scope is reported accurately.
