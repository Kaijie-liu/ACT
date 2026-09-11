# Opt-in route-complexity scheduling with matched scoped facts

Implementation stage, 2026-09-11. No new trained-model pairing has been run.
The defaults, four-arm R1 and 60-request reuse R1 remain unchanged. This stage
does not implement candidate supersets, external tools, or request-level LP
proof export. It does not reopen a sealed holdout or backend search.

## Public entry and two configurations

Use `verify_staged_linf` or its existing CLI with either:

- `configs/route_complexity_reuse_v1.json`: adaptive staged arm;
- `configs/monolithic_matched_reuse_v1.json`: matched monolithic arm.

The JSON files differ only in `comparison_method`. Both enable scoped reuse
and the same version-1 schedule, with a 300-second cooperative request budget.
No config without `route_complexity_schedule` opts into this behavior.
Eval, CPU/float64, selected-softmax top-2 and ANY_LEGAL_TOPK remain required.
Numerical acceptance gates and F0 mathematical relaxations are unchanged.

## Common preparation, charged independently to each arm

1. Build/propagate the router, query all candidate memberships and unordered
   top-2 sets using the live remaining budget. Incomplete analysis returns
   UNKNOWN (or TIMEOUT at the global deadline), never a SAFE candidate superset.
2. Propagate every candidate expert on its membership guard, with support
   tightening disabled. Save guarded output intervals in canonical expert order.
   This is a cheap **propagation** prelude, not free compute or solver-backed
   optimal support. Both arms run this exact policy from their own model/input.
3. Extract the same kind of request-scoped interval property facts as the
   previous reuse implementation. Bind model, represented box, router frame,
   property, expert, tie policy and numerical policy. The pair-to-membership
   containment rule is unchanged. No partial MILP result is promoted to a fact.

The source policy is `COMMON_GUARDED_INTERVAL_NO_SUPPORT_SOLVES`. It is shared
between the new arms, but is **not identical to the support-tightened Tier-1
interval source in the historical 60-request run**. Consequently new timings
must be measured against a rerun matched arm; old results are not a pure
scheduling ablation. No cross-request cache or historical oracle is used.

## Scheduling and live remaining budget

- Exactly one feasible pair: the adaptive arm skips expert solving and runs
  direct monolithic F0 for this single pair. The selected-softmax weights remain
  variable; all output properties still need a proof or a valid reuse fact.
- More than one feasible pair: the adaptive arm spends at most a configured
  **25% of the remaining budget** on Tier-1 expert queries, sharing that slice
  among the remaining experts. Already interval-proved experts need no query.
  A concrete full-model witness returns immediately. Otherwise, if Tier 1 is
  incomplete, all remaining time is made available to per-pair F0, including
  when Tier 1 was solver-limited rather than semantically inconclusive.
- Matched monolithic: after the identical prelude, spend the remaining budget
  on the joint F0 formulation; no extra Tier-1 solver phase. It has the same
  interval facts, not privileged results from the adaptive arm's extra queries.

The 25% slice is an explicit development design choice, not a threshold fitted
to a new evaluation or an established optimum. Property solve grants divide
the current remainder by the still-unproved obligations in that formulation.
Margin/difference/support queries retain their caps but are clipped to the
live remainder; construction and propagation consume the same clock. Branches
or properties whose facts discharge all obligations skip the associated work.

`RequestBudget` is cooperative: native propagation/solver calls can overrun
their nominal grant and cannot be forcibly interrupted in-process. Checkpoints
stop further work; late positive results are not accepted as budgeted SAFE.
TIMEOUT keeps elapsed time and explicitly censored partial-row counters, never
zero-cost successful solving. The existing CLI starts its clock before model
and data loading; Python import/startup and termination need an **external
process watchdog** for a hard end-to-end cap. API callers may pass an earlier
monotonic `budget_started_at` to charge their loading too. The caller must not
report the cooperative clock alone as a hard end-to-end runtime guarantee.

## Monolithic proof partition

For each property, partition all exact feasible pairs into disjoint sets:

- pairs discharged by the two scoped expert facts;
- residual pairs included in the monolithic F0 disjunction.

The global property lower bound is the minimum of the residual solver bound
and all reused pair bounds. If every pair is reused, no solver query is needed.
The record retains each reused pair/proof, the exact residual pair list, and
the residual solver's `pair_count`. SAFE requires the partition to cover every
tie-legal pair exactly once and every output property. The structural auditor
reconstructs the scoped facts and rejects omitted/duplicate pairs, changed
properties, inflated reused bounds, wrong solver counts and invalid schedule
decisions. It is not an independent re-proof of HZ propagation or MILP bounds.

Tier-1 solver proofs are not silently added to the common fact inventory. This
first version reuses only the registered interval facts, including in the new
monolithic arm. Separate proof export is still needed to broaden the fact kind.

## Validation and next boundary

The focused suite passes 41 tests: ten new budget/scheduler controls plus 31
existing staged, reuse, paired-comparison, monolithic and LP-checker tests.
Controls cover single-pair weighted verification, all three tied pairs, partial
and complete monolithic proof reuse, identical prelude intervals across arms,
wrong proof partitions and budgets, incomplete enumeration, common-prelude
and F0 timeouts, full-model witness replay, and early Tier-1 witness termination.
These are analytic implementation controls, not official-scale effects.
The updated structural SAFE checker also accepts all 403 historical SAFE
packages from the four-arm and scoped-reuse R1 runs, without rerunning their
solvers or changing any raw artifacts. This is backward-compatibility checking,
not independent numerical re-proving of those SAFE decisions.
An initial test invocation named a nonexistent `test_paired_followup` module;
the actual suite uses `test_paired_comparison`, with no code failure hidden.

Before any future trained-model timing comparison, freeze its cohort, ordering,
new source identity, identical outer watchdog and failure rules separately.
Keep the historical R1 directories immutable. No run is launched by adding
these configs, no claim of overcoming monolithic's historical coverage lead
is made, and no high-accuracy or independently checked full-model SAFE follows.

## Frozen paired execution R1

`configs/route_complexity_paired_r1.json` registers this next engineering step
before outcomes: the same observed ranks 0--9, three accepted bal010 models,
2/255, two arms, 300-second hard subprocess caps. It binds both method-config
hashes and the ordered selection manifest. The source identity is recorded at
launch. The six-request smoke uses rank0 on all models and both arms. The full
60-request run repeats rank0 rather than borrowing its smoke result or time.
Maximum worker budgets: 30 minutes smoke, five hours full, plus auditing.

`python -m act.pipeline.moe.route_complexity_paired --pipeline` runs smoke,
re-audits the smoke gate, then runs the full schedule and final audit. No resume,
overwrite, replacement or outcome-based early stopping. Full entry requires
the completed smoke with the same code and config, at least one full package
per arm and one paired common-fact comparison. TIMEOUTs remain in all
denominators; all-timeout smoke cannot establish conformance. A worker/audit
error stops and preserves the failed directory. A process lock prevents two
copies of this runner; source/clean-worktree checks prohibit editing mid-run.

Model order rotates and arm order alternates, giving five first/five second
positions per model/arm. Both arms independently compute their prelude. All
worker costs include process startup, loading, router queries, fact extraction,
propagation and solving. Post-run audit costs are excluded equally. BLAS and
OpenMP are limited to one thread; host load is recorded. No GPU work is used.

The final auditor replays UNSAFE, checks literal paired model/input/property
identity, scoped reuse (including monolithic partitions), and equality of the
actual guarded interval facts whenever both preludes completed. Incomplete
preludes/packages are explicitly unavailable, not silently equal. Report all
four states, adaptive-only/monolithic-only SAFE and solved, route-changing SAFE,
observed paired cost and partially observed counters. Counts describe ten
shared inputs / thirty model-input pairs, not sixty independent samples.

This is a comparison of two new schedules sharing a new common prelude, not a
pure ablation against the historical support-tightened reuse run. No effect
size or statistical-significance threshold is added after results. Negative
results are acceptable endpoints. Do not alter the 25% slice in this run.
The final JSON and all raw packages stay local until a completed-stage review
is committed/pushed; the background runner does not push automatically.

Prelaunch validation: 47 focused tests pass (six new runner registration/gate,
counter and common-fact tests plus the existing 41-test implementation suite).
Planned tmux session: `moe-route-complexity-r1`; log:
`data/moe/results/route_complexity_pipeline_20260911_r1.log`. Inspect live
state before claiming started/completed or launching another copy.
