# Relation R1: completed observed-input mechanism follow-up

Execution `5bdbd25b26dfd2227fb1b838d7f31e30a65f9c74`. Smoke6/6 and
full60/60 finished2026-09-14 (full10:07–12:33 Sydney). Separate-process
re-audit exactly reconstructs both saved summaries. Full PASS/0issues,
48 complete packages,16 concrete UNSAFE replays,12 retained outer TIMEOUTs.
All30 charged common-fact pairs match. Archive:
`../results/relation_ablation_review_20260914_r1.json`; reconstruction tool
`../review_relation_ablation.py` (requires original executable-source bytes).

Each model/arm has ten observed inputs at2/255 and300seconds. Both arms use the
same adaptive schedule and scoped reuse; only expert factor sharing differs.
The product keeps guarded marginals, not merely independent output intervals.

| Model | Shared SAFE/U/UNKNOWN/TIMEOUT | Independent SAFE/U/UNKNOWN/TIMEOUT | Mean seconds shared/independent | Paired median seconds |
|---|---:|---:|---:|---:|
| seed0 | 4/2/2/2 | 4/1/2/3 | 139.81/166.97 | -3.08 |
| seed1 | 3/3/2/2 | 2/3/3/2 | 153.07/159.48 | -0.62 |
| seed2 | 4/4/1/1 | 2/3/3/2 | 123.38/132.84 | -4.08 |

Shared gains3 SAFE and5 solved model-input pairs, no losses. These are30
model-input pairs on10 distinct images, not60 independent samples or a new
holdout. The two additional solved gains are UNSAFE, not certificates.

| SAFE gain | Dataset index | Legal pairs | Shared source | Independent stopping reason |
|---|---:|---:|---|---|
| seed1/rank7 | 4018 | 3 | Tier2 F0 | UNKNOWN_WEIGHTED_RELAXATION |
| seed2/rank5 | 4014 | 2 | Tier2 F0 | UNKNOWN_WEIGHTED_RELAXATION |
| seed2/rank7 | 4018 | 1 | direct monolithic F0 | UNKNOWN_MONOLITHIC_SOLVER_LIMIT |

The first two establish complete multi-legal-route endpoint gains against the
registered correlation-discarding representation. The third has a solver-limit
confound and is not a route-changing gain. A stopping label is not a complete
causal diagnosis of every intermediate bound. Product construction changes
factor count and solve cost; do not attribute every finite-budget difference
uniquely to relaxation tightness. Mean cost savings are tail-sensitive.

All27 jointly recorded gate ranges agree exactly;3 pair-ranges are recorded
on only one side. Unrecorded ranges are not counted as agreement. Scoped fact
equality is complete through durable snapshots; neither that nor the package
audit is an independent proof of all SAFE bounds. SAFE uses the unchanged
HZ/HiGHS numerical policy. Analytic cancellation controls independently isolate
the relationship (.2 shared versus-.8 product); their tolerance is not a new
acceptance threshold. Old900 confirmation, composite failures and sealed
backend pilots are unchanged. No further sample expansion is queued.
