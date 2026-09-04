# Paper-complete roadmap and red-team closure

Decision date: 2026-08-30. This document replaces deadline-driven expansion
with a finite, evidence-driven completion definition. FSE 2027 is explicitly
skipped. A later ICSE or FSE cycle will be selected only after its official CFP
is published and this project reaches `PAPER_COMPLETE`.

## Completion definition

`PAPER_COMPLETE` means that all five conditions below are simultaneously true:

1. The official-scale B3 table is complete and independently audited across the
   frozen radius grid, including route applicability, identical-backend
   route-invariance, Route A, theorem-instantiation status, and attack ceiling.
2. The official third-party AdvMoE target has a completed official-code
   reproduction, frozen sampled input-space route census, and hidden-computation
   Route A results. A separately labelled learned-router RT-ER variant is a
   fallback only if this execution proves infeasible.
3. Expert-count scaling is evaluated at `E in {4, 8, 16, 32, 64}` using the
   frozen lazy route-set enumeration/no-good-cut design and tightened big-M
   bounds.
4. Every red-team item in this document is either closed by evidence or retained
   as an explicit limitation with paper wording that does not exceed the
   evidence.
5. The manuscript, artifact manifest, witness replay, and anonymous
   reproduction exercise are frozen and pass their independent audits.

December 2026 is a planning target, not an endpoint that may relax any item
above. No submission date can redefine `PAPER_COMPLETE`.

## Finite work registry

### Tier 1: required

| Work item | Evidence bought | Current status |
|---|---|---|
| Official RT-ER B1 and B3 | Official-scale end-to-end comparison | B1 seeds 0 and 1 landed outside the frozen reference intervals; B3 r5 completed as audited numerical conformance, with the formal endpoint still gated on outward-rounded CROWN bounds |
| Router-gradient audits of RT-ER, robust-moe-cnn, and V-MoE | Multiple-pipeline external-validity map | Source audit complete, independently audited |
| AdvMoE official third-party learned deep-path target | Learned-router and non-output-layer external validity | Architecture/dependency audits plus strong-PGD/sparse-CROWN init pilot complete; K=20 dual-BN init census complete and init line sealed; alpha/beta closure, training, and Route A wait for B1/B3 |
| Lazy route-set enumeration plus no-good cuts; exact-support big-M | `E` scalability and search-relaxation defense | Implemented; E=4--64 scaling and exact-support timed rerun completed with zero-issue audits |
| Dimension-law simulation grid | Defense against two-point-fit criticism | Executed and audited: both point slopes near -1/2, but one of two frozen bootstrap-interval rules misses by 8.6e-5; composite endpoint retained as failed |
| Source-native survey retrieval | Recall-qualified survey if retrieval succeeds | Pending institutional retrieval |
| Monolithic solver baseline | Direct decomposition comparison | Completed on the frozen 20-row cohort; Route A 12/20 versus monolithic 8/20, descriptive paired p=0.21875 |

The dimension simulation grid is frozen before execution as
`d in {1000, 3000, 12000, 50000, 150000, 500000}`. Router initialization must
follow the documented PyTorch default scaling. Synthetic input second moments
are frozen to the independently audited CIFAR normalized moment
`1.554975707473284` and Tiny-224 literal-centre moment `0.9276910751175297`;
every synthetic sample uses an exactly matching Rademacher construction. The
primary check is the fitted log-log slope against `-1/2`, with uncertainty and
constants reported rather than tuned. The point-slope tolerance is `0.08`, and
the 20-seed cluster-bootstrap 95% interval must include `-1/2`. Sixteen samples
per seed and moment, 2,000 bootstrap replicates, and all random-seed
derivations are frozen before execution.

The frozen run contains 3,840 observations. Point slopes are `-0.5286` and
`-0.4854`, both within the registered `0.08` error. The Tiny-moment bootstrap
interval contains `-0.5`; the CIFAR-moment interval is
`[-0.555536,-0.500086]` and therefore does not. The composite preregistered
endpoint is recorded as failed without rounding, threshold changes, or a
replacement run. The grid is supportive order-law evidence, not an
unqualified confirmation.

### Tier 1.5: absorbed by the AdvMoE target after B3

The official AdvMoE CIFAR-10 architecture replaces the proposed home-built
hidden-layer toy. Its router reads the image before the ResNet stem, so it is
not a hidden-state router and there is no prefix-HZ domain to reuse. Instead,
one learned global hard top-1 route specializes 16 hidden MoE convolutions
throughout the ResNet. The bounded task is to propagate the nonlinear router
from pixels and verify each feasible full deep pathway. This closes the
"output-layer only" attack without mislabelling the architecture.

The learned-router RT-ER modification is optional fallback work, not a
completion requirement, because an official third-party learned router carries
stronger external-validity evidence.

The AdvMoE line is capped at one official seed-0 reproduction, full-test
first-order router telemetry at init/final, deterministic intermediate
telemetry and strong-PGD subsets, trained-checkpoint-only CROWN/alpha/beta
closure, the five-radius two-path staged-verifier table, and one
guard-representation ablation. No 10,000-input init CROWN census, second ratio,
or AdvMoE expert-count sweep is permitted. The literal router's
auto_LiRPA rejection and the default-CROWN memory bottleneck must be reported.
The resource-gated sparse-CROWN pilot tightens IBP by 5.20x--5.37x in median
bound magnitude but remains negative on every row; its 0/100 strong-PGD flips
and 100/100 undecided band cannot be renamed formal route stability. The init
engineering pilot has independently audited orchestration, all attack
endpoints, BN identity, and CROWN accounting, but it does not satisfy the
required init/final census or staged-verifier table. Alpha-CROWN and
beta-CROWN/BaB remain closure tiers rather than silently substituted results.
The existing pilot's two closely agreeing local estimates (`67.85/255`
first-order and `70.64/255` PGD-slope) are now explicitly confounded: the
official initialization routes all 10,000 test images to expert 0, with signed
score-difference `abs(mean)/standard_deviation=9.1406`. The approximately 130x
architecture interpretation is retired. Strong PGD finds no flip on the 20
inputs through epsilon 1, but this remains attack non-discovery. The five-radius
layered relaxation-inflation curve shows that CROWN improves IBP by
`5.17x--5.36x` while leaving `1.07e11--1.66e11` CROWN medians; it has no
approximation-ratio claim. Trained telemetry must report route share and signed
offset under both eval/current-running-statistics and registered
train/ordered-test-batch semantics at every checkpoint. The K=20 closure finds
exact collapse for 13/20 eval-default-stat seeds and 8/20 train-batch-stat
seeds, with median maximum loads of 100% and 99.305%. BatchNorm semantics
changes the degree but does not remove the initialization phenomenon; no
further init measurement is allowed.
The first-five-sample numerical-reach bisection is ULP-limited in all five rows:
its median sign transition is `1.856e-9--1.868e-9`, versus a `1.609e-12`
linear extrapolation. It is retained as a negative numerical-semantics result,
not a sound reach or paper-ready certificate-gap axis. The two-path staged
table does not depend on certifying router stability.

### Tier 2: optional and non-blocking

An upstream issue or patch may document dynamic `GatherElements` and input
polytope limitations in a current general-purpose verifier. It cannot delay B3
or manuscript closure.

## Explicit prohibitions

- No third router-census dataset.
- No TinyImageNet ViT-224 end-to-end expert certification.
- No F1 segmented fallback unless new evidence identifies a material
  range-relaxation bottleneck.
- No prevalence statement before source-native retrieval closes.
- No new experiment unless it names a red-team attack below and specifies the
  evidence needed to close it.
- No result may overwrite a preregistered endpoint or remove a negative result.

## Red-team registry

| Attack | Required defense | Status |
|---|---|---|
| “The 49%-accuracy model is a toy.” | Official-scale B3 plus an official third-party learned-router model if executable. | B3 numerical conformance complete on all 318 branches with a zero-issue audit, but formal SAFE is deliberately zero pending outward-rounded bounds; AdvMoE seed-0 r1 is preserved as an excluded supervisor-race failure after 27 snapshots, and the unchanged r2 reproduction restarts from epoch zero because upstream checkpoints omit RNG states |
| “Enumeration cannot scale beyond E=8.” | Lazy enumeration/no-good cuts and `E={4,8,16,32,64}` data. | Pending |
| “The method only covers output-layer MoE.” | Official AdvMoE deep route-specialized pathway or explicit limitation. | Both static paths specialize exactly in tests; init router bracket remains unresolved and property verification remains pending after B3 |
| “The route-invariance baseline is self-serving.” | Exact definition-level applicability plus identical downstream backend and public implementation. | Verification-scale formal comparison and official-scale numerical-conformance comparison complete; official formal CROWN endpoint remains open |
| “The cohort was selected after outcomes were known.” | Full-test applicability decomposition, frozen ranks, and immutable endpoints. | Closed |
| “The dimension law fits two points.” | Frozen synthetic dimension grid, slope and constant analysis. | Closed with a mixed preregistered result: point estimates support the order, composite gate fails 1/2 interval checks |
| “The paper targets one prior work.” | Multi-pipeline gradient audit, case series, neutral artifact wording, and responsible disclosure. | Audit complete; contact managed by PI under the frozen one-reminder/one-issue protocol |
| “Guard value survives any abstraction.” | Retained-guard positive result plus box-hull, eta-reduction, and tie-soundness negative controls. | Closed at verification scale |
| “Incremental solving gives a universal speedup.” | Separate build-dominated and search-dominated measurements, retaining the negative end-to-end result. | Closed |
| “UNSAFE comes from a relaxation witness.” | Full-model replay for every weighted unsafe result. | Closed for completed cohorts; remains a standing gate |
| “Preprocessing, frontend set representation, or BN mode is outside the artifact identity.” | Runtime/preprocessing identity, requested/represented-set schema, ULP point-collapse regression, dual-BN initialization census, and fail-closed replay. | Closed in implementation and exercised by the accepted B3 r5 identity/audit chain |

## Change control

A new work item may enter this registry only if it closes an identified attack
that no current item can close. It must state its bounded cost, success and
failure interpretations, and which existing item it displaces. Interesting
measurements without a claim-level gap are out of scope.

Administrative external actions are tracked separately. Author contact is
`CONTACT_MANAGED_BY_PI`; the repository retains only the one-reminder,
one-neutral-public-issue, and standard-paper-wording protocol. The agent does no
countdown or follow-up. No Zenodo connector, account, or deposit token is
available, so no deposit, DOI, or publication may be claimed.

## Post-B1 resource schedule

B1 landing triggers two resource-separated lanes. The GPU lane builds the
isolated AdvMoE reproduction environment and starts official seed-0 training.
The CPU lane runs the paired exact-support-big-M, lazy-enumeration/MIP-start,
and monolithic-MILP studies with bounded workers and low process priority.
These jobs may overlap only while monitoring shows that the AdvMoE data loader
and CPU timing do not contend. B3 CROWN evaluation begins after both lanes have
closed and the GPU has a quiet window; it is not launched concurrently with
training. This schedule changes resource occupancy, not any frozen scientific
endpoint.
