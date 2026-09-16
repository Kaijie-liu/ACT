# Source-cache attribution V1 — protocol, not results

Separate follow-up to the sealed upstream reuse experiment. Reuse the four
already observed inputs **220, 222, 230, 232**, epoch-89 convolutional checkpoint,
materialized float64 boxes and epsilon 2/255. No sample expansion, clean scan,
route/bound-based selection, precision intervention or query reordering.
This is development attribution, NOT an independent confirmation cohort.

## Arms and charging

| Arm | Upstream source cache | Upstream exact matrix cache | Portable tail cache |
| --- | --- | --- | --- |
| matrix_only | OFF | ON | ON |
| both | ON | ON | ON |

Eight requests, rank-blocked alternating order: matrix_only/both, both/matrix_only,
matrix_only/both, both/matrix_only. Same support-first query order, all source,
construction, range and dual checks. No verdict reuse or check elision.
Each arm independently loads, propagates, proposes, packages and checks.
All operations, including source decode/freeze/copy, CSR parsing and terminal
publication, share the original 300-second start; work cutoff 298, proposal cap
60, tail reserve 80. One CPU worker/thread, resource gate, no retry or resume.
Resource wait and post-terminal archival audit are recorded separately.

New namespace `source_cache_ablation/` is a versioned fork of the frozen
`reuse_supervised/` supervisor. Only arm configuration/identity and study
selection/reporting change. Parent sources, freeze and outcomes are untouched.
New raw destination: `data/moe/results/source_cache_ablation_comparison_20260916_v1`.
Parent selection tensors are reused by SHA-256, not recaptured or substituted.

## Acceptance and interpretation, frozen before execution

Primary: paired complete independent checks AND full charged cost, retaining
missing evidence, checked nonpositive, incomplete route coverage, timeout,
error and conditional positive separately. Report each request, not only a
mean or successful runs. Secondary: obligation completeness, all paired lower
bounds where both present, source hashes and exclusive timing categories.
Missing or different evidence is explicitly reported, never treated as equal.
Three exact dual evaluations per successful query remain required by the
inherited proposal path; no claim that source caching is individually useful
or useless before this comparison.

This small observed cohort supports descriptive implementation attribution, not
general statistical superiority. A default is NOT changed automatically:
consider full-cost differences jointly with evidence/endpoint losses and source
consistency. Lower CSR time alone is not a success condition. Mixed outcomes
remain mixed. No additional runs to rescue a direction; ERROR stops and leaves
all remaining slots in the roster. Completed negative cases remain recorded.

## Gates and commands

1. Run numbered controls: `python -m source_cache_ablation.controls`.
2. Freeze using the passing receipt: `python -m source_cache_ablation.study freeze --controls PATH`.
3. Separate process `python -m source_cache_ablation.study reconstruct` verifies
   exact parent selection/configuration and materialized hashes; no model calls.
4. Commit/push clean feature branch before a separately authorized execution:
   `nice -n 10 /data1/Kane/miniconda3/envs/act-py312/bin/python -m source_cache_ablation.study launch`.

**This stage freezes only; do not launch automatically.** Controls may solve
analytic fixtures; they do not query any of these four real verification tasks.
Old frozen results are not recalculated or modified. Numerical/conditional
guarantees, network→HZ, guard and route-exclusion trust remain unchanged.
