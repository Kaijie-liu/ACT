# Experiment 1 confirmatory protocol

This protocol is frozen before inspecting clean-correct CIFAR-10 ranks 100--199.
Those ranks were not used to design the candidate analysis, guarded support,
solver schedule, or F0 fallback.

## Cohort and artifacts

- checkpoint: `cifar10_top2_e8_seed0_bal010.pt`;
- deterministic cohort: clean-correct ranks 100--199 under the same checkpoint;
- official TorchVision CIFAR-10 test set;
- tracked rerun config: `configs/experiment1_confirmatory_bal010_r1.json`;
- raw output: a new `experiment1_confirmatory_bal010_r1` directory;
- development and failed F0 directories are read-only and never overwritten.

The saved sample-index prefix must reproduce development ranks 0--99 before the
runner accepts ranks 100--199.

The first launch directory `experiment1_confirmatory_bal010` is permanently
excluded. Its runner recorded the 300-second limit but did not terminate a row:
rank 155 took 302.3 seconds and rank 171 returned after 382.5 seconds. The task
was stopped after 72 complete boundary rows. Its complete 400-row census and
partial boundary outputs are preserved only as failed engineering artifacts and
none of their verdicts enters confirmatory endpoints. The `_r1` rerun changes no
scientific radius, cohort, solver budget, fallback trigger, or GO threshold; it
only hard-enforces the already preregistered wall deadline.

## A. Fixed-radius router/width census

For every sample and `epsilon in {0.25,0.5,1,2}/255`, record IBP, ordinary
zonotope, and exact-router-HZ candidate sets; exact feasible unordered top-2
sets; route stability; structural monolithic and route-conditioned widths; and
guard-aware binary accounting. No output-property solve is run in this stage.

Each branch must close

```text
binaries_before - binaries_after
  = lp_support_eliminated
  + milp_support_eliminated
  + structural_or_propagation_eliminated.
```

`fast_unstable` counts only the preactivations entering direct support
statistics. `binaries_before` counts actual expert ReLU variables over the whole
branch, so the two fields are not interchangeable.

## B. Route-boundary end-to-end certification

The router-only exact feasibility search finds a strict bracket

```text
stable lower < minimum route-set-change radius <= unstable upper.
```

The single primary radius is `1.05 * unstable upper`. An incomplete bisection is
retained if the strict bracket remains valid; its width and
`bisection_complete=false` are reported.

Each boundary row runs in a separate spawned process with a 300-second wall
deadline. At the deadline the child is terminated and the row is recorded as
`TIMEOUT/INSTANCE_HARD_DEADLINE`; no partial verdict can be promoted. Completed
witness artifacts are first written in a rank-local work directory and promoted
only after the child returns a complete row. Timed-out work remains quarantined
and is excluded from witness counts.

The frozen order is:

1. exact candidate and legal unordered route-set feasibility;
2. Route A expert-wise gate elimination with guarded support;
3. F0 only for `UNKNOWN_GATE_SUFFICIENCY` or
   `UNKNOWN_EXPERT_WITNESS_NOT_LIFTED`;
4. concrete full-model replay for every unsafe candidate;
5. no automatic F1.

Unique `SAFE` requires more than one exact feasible route set, an `UNKNOWN`
route-invariance baseline, and final `SAFE_GATE_ELIMINATION` or
`SAFE_WEIGHTED_RANGE`. Its denominator is all 100 confirmatory samples, never
solved-only cases. This is a route-boundary certification yield, not natural
input prevalence.

## Numerical SAFE policy

The runner checks the tracked numerical policy against the implementation:

- solver success and status 0 are required for optimized bounds;
- MILP certification uses a finite `mip_dual_bound`;
- pure LP certification uses only a status-0 optimal objective;
- a non-optimal primal incumbent is never a lower-bound certificate;
- post-recovery feasibility and integrality tolerances are both `1e-7`;
- optimized bounds receive `1e-9 + 1e-9 * scale` outward correction followed
  by `nextafter` toward the unsafe direction;
- F0 `SAFE` requires the corrected lower bound to be strictly greater than
  `1e-7`;
- gate-elimination `SAFE` requires every violation-feasibility query to be
  proved infeasible, not merely to lack an incumbent.

Relaxation violations remain `UNKNOWN`; only concrete full selected-softmax
replay may produce `UNSAFE`.

## Primary endpoints and GO rule

The audit reports Wilson 95% intervals for unique `SAFE`, sample-cluster
bootstrap intervals for candidate reduction, unconditional and route-unstable
width distributions, guard accounting/cost, matched no-support versus support
solve coverage, F0 incremental resolution, and paired runtime overhead.

Public baseline work unlocks only if all preregistered conditions pass:

- zero independent-audit issues and every unsafe witness replayed;
- at least 10 unique safe samples and at least 10% yield;
- end-to-end solved rate at least 60%;
- exact candidate reduction at least 20% on fixed route-unstable rows;
- conditional width median below 0.7 and p90 below 1;
- F0 resolves at least 25% of base semantic incompleteness;
- all guard identities close and no silent numerical fallback occurs.

Failure does not trigger F1 automatically. The unresolved composition is
reported before any method or model change.
