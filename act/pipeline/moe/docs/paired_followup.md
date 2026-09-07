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
