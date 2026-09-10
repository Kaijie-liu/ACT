# Four-method common-task paired follow-up

This is a follow-up on the already observed 100-image, three-model common
`2/255` task. It is not a fresh holdout or a replacement of the 2/3 bundle
result. The selection and base numerical config are hash-bound in
`configs/paired_followup_r1.json`; no checkpoint is trained or selected here.

## Methods and budget

Each request has a 300-second subprocess wall cap including Python startup,
checkpoint/data loading, router analysis, support, graph construction and
property solves. Every method computes its own exact route sets. There is no
free reuse of the earlier census. Evidence audit/replay occurs after measured
execution for all methods. A late/incomplete package cannot override an outer
timeout. Each interrupted query receives a terminal record with null missing
times; failures and retries remain on disk.

- `staged`: unchanged v1 candidate, Tier-1 and conditional F0 budgets.
- `route_invariance`: same route analysis and downstream staged/F0 path,
  but returns UNKNOWN before expert propagation if multiple legal unordered
  top-2 sets exist. A stable set does not freeze its gate weights.
- `monolithic_f0`: same exact route analysis, guarded pair support and F0
  McCormick semantics. One joint disjunctive MILP per property represents all
  feasible pairs. Each property can consume the remaining outer cap (solver
  limit 300 seconds); natural property order is fixed. There is no Tier 1.
- `tier1_only`: same guarded first-tier semantics, with escalation allowance
  increased from 25 to 300 seconds per branch, subject to the same outer cap.
  Semantic expert failure remains UNKNOWN; it never invokes F0.

Internal schedules intentionally reflect different algorithms; the total cap,
solver/numerical policy, represented input, model and full-model witness rules
are common. A single difficult branch may consume the cap. This is part of
the registered schedule, not evidence that all possible baseline schedules
have been optimized. No optimal-status gate is weakened.

## Interleaving and costs

Model order rotates by sample rank. Within each model/sample block, method
order rotates by rank plus model index. Every method occupies each position
exactly 25 times per model in the 100-rank run. Execution is sequential with
single-thread BLAS/OpenMP settings; system load is recorded. This shared
server is not claimed to be an isolated timing environment.

Smoke is the first registered rank on all three models/all four methods:
12 jobs, at most 1 hour of worker budget. It is a correctness/runner gate,
not a yield-based selection. Full follow-up is 1,200 jobs, at most 100 hours
of worker budget plus audits/loading outside workers. Both timeouts and
negative results count; all jobs run even if a desirable difference appears.
No full run starts until all 12 smoke rows pass the independent structural
audit and every UNSAFE witness replays.

The primary descriptive outputs are per-model SAFE, UNSAFE, UNKNOWN, TIMEOUT
counts and paired gained/lost SAFE and solved sets for each baseline. Report
paired cost with censoring and server-load qualifications. A SAFE/UNSAFE
conflict fails the audit. Results cannot establish cross-architecture
superiority or independently checked numerical proof validity.

## Executable paths

Run `python -m act.pipeline.moe.paired_followup --smoke` in `act-py312`.
Use `python -m act.pipeline.moe.audit_paired_followup RESULT_DIRECTORY` to
independently audit identities, coverage, status conflicts, and witnesses.
Without `--smoke`, the runner executes the full frozen task after its smoke
gate. `--resume` requires the same code/config identity and an ordered result
prefix; failed attempts are retained. Do not run another writer concurrently.

Tests cover stable variable-weight F0, tie-inclusive invariance rejection
before expert solves, monolithic weighted safety and missing-property mutation,
Tier-1-only fallback exclusion, unchanged numerical policy, method-position
balance and distinct SAFE versus solved comparisons. The original v1 tests
remain green. No F0 proof-reuse optimization is enabled during this baseline
measurement; it will need its own subsequent comparison.

## Completed R1 results (reviewed 2026-09-11)

All 1,200 jobs completed; the original automatic audit and the independent
2026-09-11 re-audit both pass with zero issues, 1,075 complete packages, and
171 full-model UNSAFE replays. The 125 outer-deadline requests have terminal
records and remain in their 100-request denominators. Other TIMEOUT labels
can have complete packages (e.g. expert/support timeout). Audit integrity is
not independent numerical re-proving of SAFE.

Derived result: `results/paired_followup_full_review_20260911.json`, generated
by `analyze_paired_followup.py`, binds raw rows, runtime, summary and re-audit
hashes and retains all gained/lost rank lists. No new solver query was used in
the analysis. The execution source hash/HEAD remain those of the frozen run.

| Model | Method | SAFE | UNSAFE | UNKNOWN | TIMEOUT | Solved/100 | Mean observed seconds |
|---|---|---:|---:|---:|---:|---:|---:|
| seed0 | staged | 30 | 24 | 39 | 7 | 54 | 67.46 |
| seed0 | route invariance | 22 | 10 | 63 | 5 | 32 | 29.21 |
| seed0 | monolithic F0 | 46 | 19 | 2 | 33 | 65 | 158.21 |
| seed0 | Tier-1-only | 21 | 10 | 64 | 5 | 31 | 37.58 |
| seed1 | staged | 26 | 21 | 38 | 15 | 47 | 76.86 |
| seed1 | route invariance | 18 | 8 | 62 | 12 | 26 | 30.58 |
| seed1 | monolithic F0 | 36 | 18 | 5 | 41 | 54 | 180.65 |
| seed1 | Tier-1-only | 22 | 8 | 60 | 10 | 30 | 68.85 |
| seed2 | staged | 44 | 22 | 26 | 8 | 66 | 62.46 |
| seed2 | route invariance | 37 | 6 | 51 | 6 | 43 | 26.85 |
| seed2 | monolithic F0 | 52 | 18 | 1 | 29 | 70 | 146.63 |
| seed2 | Tier-1-only | 32 | 7 | 54 | 7 | 39 | 44.49 |

SAFE is scoped to the frozen HZ/HiGHS acceptance policy. These are selected
clean-correct-input fractions, not full-test certified accuracy. Costs include
all capped executions, not only solved rows; they exclude post-run audit and
are not estimates of uncensored time to solve. Shared-server timing and the
distinct registered internal schedules prevent an unconditional speedup claim.

The following pairs are **staged-only / baseline-only**, not net counts:

| Model | Baseline | SAFE discordance | Solved discordance | Median paired time difference (staged minus baseline), s |
|---|---|---:|---:|---:|
| seed0 | route invariance | 8 / 0 | 22 / 0 | 0.75 |
| seed1 | route invariance | 8 / 0 | 21 / 0 | 0.30 |
| seed2 | route invariance | 7 / 0 | 23 / 0 | 0.25 |
| seed0 | Tier-1-only | 12 / 3 | 26 / 3 | 13.89 |
| seed1 | Tier-1-only | 7 / 3 | 21 / 4 | 0.18 |
| seed2 | Tier-1-only | 15 / 3 | 30 / 3 | 0.82 |
| seed0 | monolithic F0 | 4 / 20 | 9 / 20 | -71.30 |
| seed1 | monolithic F0 | 10 / 20 | 17 / 24 | -107.24 |
| seed2 | monolithic F0 | 5 / 13 | 10 / 14 | -55.15 |

Staged increases SAFE and solved counts over invariance and Tier-1-only in
each model, at additional observed cost. Against monolithic, staged loses
16/10/8 SAFE and 11/7/4 solved outcomes net. Neither solution set contains
the other. Staged has lower observed cost and more route-changing SAFE
(8/8/7 versus monolithic 5/1/3), but cannot be described as the overall
coverage winner. The three models and common inputs must not be pooled as
1,200 independent observations; no post-hoc significance gate is introduced.

Of the 53 monolithic-only SAFE model-input pairs, staged reasons are weighted
solver limit (29), expert timeout (11), candidate solver limit (12) and support
timeout (1). These are associated stop locations, not a causal separation of
schedule, formulation, propagation or solver effects. In particular,
monolithic gives one property up to the remaining 300-second outer cap while
staged retains v1 stage budgets. This experiment does not isolate decomposition
alone and does not justify changing the frozen schedule to chase these rows.

The historical 20-row result, original 2/3 bundle and strict large-model pilot
remain unchanged. Next work is scoped proof reuse and checked bound evidence,
not a claim of universal staged dominance or another unregistered R1 closure.
