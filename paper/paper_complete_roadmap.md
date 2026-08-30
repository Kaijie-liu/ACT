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
| Official RT-ER B1 and B3 | Official-scale end-to-end comparison | B1 running; B3 pending |
| Router-gradient audits of RT-ER, robust-moe-cnn, and V-MoE | Multiple-pipeline external-validity map | Source audit complete, independently audited |
| AdvMoE official third-party learned deep-path target | Learned-router and non-output-layer external validity | Architecture/dependency audits plus strong-PGD/sparse-CROWN init pilot complete; 100/100 remains undecided; alpha/beta closure, training, and Route A wait for B1/B3 |
| Lazy route-set enumeration plus no-good cuts; exact-support big-M | `E` scalability and search-relaxation defense | Pending |
| Dimension-law simulation grid | Defense against two-point-fit criticism | Pending; grid must be frozen before execution |
| Source-native survey retrieval | Recall-qualified survey if retrieval succeeds | Pending institutional retrieval |
| Monolithic solver baseline | Direct decomposition comparison | Pending after B1 |

The dimension simulation grid is frozen before execution as
`d in {1000, 3000, 12000, 50000, 150000, 500000}`. Router initialization must
follow the documented PyTorch default scaling. Synthetic input second moments
must be fixed before results are viewed. The primary check is the fitted
log-log slope against `-1/2`, with uncertainty and constants reported rather
than tuned.

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
The existing pilot additionally supports two closely agreeing empirical
boundary-scale estimates (`67.85/255` first-order and `70.64/255` PGD-slope),
approximately `130x` the exact RT-ER K=20 aggregate median. This closes an
architecture-regime diagnostic only; it neither closes the nonlinear-router
bracket nor licenses a causal architecture claim.
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
| “The 49%-accuracy model is a toy.” | Official-scale B3 plus an official third-party learned-router model if executable. | B3 pending; AdvMoE strong-PGD/sparse-CROWN init bracket is audited but fully undecided; alpha/beta closure and training pending |
| “Enumeration cannot scale beyond E=8.” | Lazy enumeration/no-good cuts and `E={4,8,16,32,64}` data. | Pending |
| “The method only covers output-layer MoE.” | Official AdvMoE deep route-specialized pathway or explicit limitation. | Both static paths specialize exactly in tests; init router bracket remains unresolved and property verification remains pending after B3 |
| “The route-invariance baseline is self-serving.” | Exact definition-level applicability plus identical downstream backend and public implementation. | Verification-scale comparison complete; official scale pending B3 |
| “The cohort was selected after outcomes were known.” | Full-test applicability decomposition, frozen ranks, and immutable endpoints. | Closed |
| “The dimension law fits two points.” | Frozen synthetic dimension grid, slope and constant analysis. | Pending |
| “The paper targets one prior work.” | Multi-pipeline gradient audit, case series, neutral artifact wording, and responsible disclosure. | Audit complete; contact managed by PI under the frozen one-reminder/one-issue protocol |
| “Guard value survives any abstraction.” | Retained-guard positive result plus box-hull, eta-reduction, and tie-soundness negative controls. | Closed at verification scale |
| “Incremental solving gives a universal speedup.” | Separate build-dominated and search-dominated measurements, retaining the negative end-to-end result. | Closed |
| “UNSAFE comes from a relaxation witness.” | Full-model replay for every weighted unsafe result. | Closed for completed cohorts; remains a standing gate |
| “Floating-point preprocessing is outside the artifact identity.” | Runtime/preprocessing identity schema and fail-closed replay. | Closed in implementation; B3 instance pending |

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
