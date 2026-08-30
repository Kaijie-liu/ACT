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
2. The separately labelled learned-router RT-ER variant has three completed
   seeds, immutable checkpoints, router-geometry trajectories, and certificate
   decomposition results.
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
| Learned-router RT-ER variant, three seeds | Controlled static-versus-learned routing comparison | Approved, must wait for B1 |
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

### Tier 1.5: strongly recommended after B3

A minimal hidden-layer MoE instance places dispatch in the second layer of a
small CNN. The prefix HZ becomes the shared router/expert input domain; route
conditioning and staged verification must reuse the existing semantics. This
is one bounded model and experiment, not a new architecture sweep.

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
| “The 49%-accuracy model is a toy.” | Official-scale B3 plus an official third-party learned-router model if executable. | B3 pending; robust-moe-cnn source path identified |
| “Enumeration cannot scale beyond E=8.” | Lazy enumeration/no-good cuts and `E={4,8,16,32,64}` data. | Pending |
| “The method only covers output-layer MoE.” | Bounded hidden-layer instance or explicit limitation. | Pending Tier 1.5 |
| “The route-invariance baseline is self-serving.” | Exact definition-level applicability plus identical downstream backend and public implementation. | Verification-scale comparison complete; official scale pending B3 |
| “The cohort was selected after outcomes were known.” | Full-test applicability decomposition, frozen ranks, and immutable endpoints. | Closed |
| “The dimension law fits two points.” | Frozen synthetic dimension grid, slope and constant analysis. | Pending |
| “The paper targets one prior work.” | Multi-pipeline gradient audit, case series, neutral artifact wording, and responsible disclosure. | Audit complete; disclosure delivery pending external channel |
| “Guard value survives any abstraction.” | Retained-guard positive result plus box-hull, eta-reduction, and tie-soundness negative controls. | Closed at verification scale |
| “Incremental solving gives a universal speedup.” | Separate build-dominated and search-dominated measurements, retaining the negative end-to-end result. | Closed |
| “UNSAFE comes from a relaxation witness.” | Full-model replay for every weighted unsafe result. | Closed for completed cohorts; remains a standing gate |
| “Floating-point preprocessing is outside the artifact identity.” | Runtime/preprocessing identity schema and fail-closed replay. | Closed in implementation; B3 instance pending |

## Change control

A new work item may enter this registry only if it closes an identified attack
that no current item can close. It must state its bounded cost, success and
failure interpretations, and which existing item it displaces. Interesting
measurements without a claim-level gap are out of scope.

Administrative external actions are tracked separately. The user authorized
responsible disclosure and a versioned Zenodo v1 on 2026-08-30. Delivery and
deposit remain `AUTHORIZED_PENDING_EXTERNAL_CHANNEL`. The corresponding-author
address was resolved from the official paper and a send was attempted, but the
connected Gmail account lacks the required search/send OAuth scopes (HTTP 403),
so no message was delivered. No Zenodo connector, account, or deposit token is
available. Neither action may be represented as completed.
