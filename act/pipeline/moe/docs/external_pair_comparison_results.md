# Complete-cost external path R1 — completed and independently re-audited

Execution `0de4fe1c7`; six smoke requests and all 60 full requests complete.
Separate-process audit exactly reproduces both saved audits, PASS/zero issues.
Full run: 25 ACT packages, 30 external terminal records, 9 dynamic-model UNSAFE
replays and 5 outer timeouts. All timeouts remain in their original denominators.
Source and raw-file identities, per-request discordances and all-state costs:
`../results/external_pair_comparison_review_20260914_r1.json`.
Reconstruct with `python -m act.pipeline.moe.archive_external_pair_comparison`.
No new solver/attack/verification queries were run to prepare this archive.
Final validation: 42 focused tests pass in the unchanged act-py312 environment;
a second archive reconstruction matches the committed artifact exactly.

## Complete request results

Ten previously observed images, three fixed models, two arms, 2/255 and one
300-second outer budget per complete request. The external arm is **ACT route
frontend + plain CROWN whole-box variable-weight static pairs**, not an
independent dynamic-MoE verifier or full alpha-beta-CROWN/BaB configuration.

| Model | ACT HZ-policy SAFE / UNSAFE / UNKNOWN / TIMEOUT | CROWN numerical POSITIVE / UNKNOWN | Mean complete seconds ACT / CROWN | Median paired seconds ACT minus CROWN |
|---|---|---|---|---|
| Seed 0 | 4 / 2 / 2 / 2 | 5 / 5 | 139.00 / 4.27 | 79.52 |
| Seed 1 | 3 / 3 / 2 / 2 | 4 / 6 | 152.64 / 4.23 | 121.03 |
| Seed 2 | 4 / 4 / 1 / 1 | 4 / 6 | 122.68 / 4.18 | 93.25 |

Each row/arm has denominator ten. Neither arm has ERROR; the external arm has
no TIMEOUT or replayed UNSAFE. Aggregate ACT has 11 SAFE,9 UNSAFE,5 UNKNOWN,
5 TIMEOUT; the external path has 13 numerical positive filters and17 UNKNOWN.
Nine versus zero UNSAFE records do not establish an attack-strength ranking:
the external path only uses five fixed conformance probes, not a matched attack.
Negative CROWN bounds are never accepted as concrete counterexamples.

There are eight shared positive requests, three ACT-only policy SAFE and five
CROWN-only numerical positives. These are not interchangeable proof classes.
ACT's 20 resolved requests versus13 external positive requests includes its
counterexample search; it is not a 20-versus13 certificate advantage.

## Complementarity and limits

| ACT-only SAFE | Legal pairs | ACT completion | Worst whole-box CROWN lower bound |
|---|---|---|---|
| Seed 0/index4029 | 1 | direct monolithic F0 | -0.61573 |
| Seed 1/index4018 | 3 | Tier 2 F0 | -1.35138 |
| Seed 2/index4014 | 2 | Tier 2 F0 | -0.38805 |

All CROWN queries in these rows completed; these are not external frontend
rejections or timeouts. The two multi-pair rows also occur in the separate
relationship ablation, but the present comparison does not isolate guard,
abstraction and scheduling as separate causal factors. The single-pair gain
cannot be attributed to guard-domain reduction merely from its status.

CROWN-only positives: seed0/index4009 and4014, seed1/index4006 and4022,
seed2/index4009. ACT returned solver-limit UNKNOWN on all five: four direct
monolithic cases and one multi-pair weighted case. They are real losses for
this configuration, not ignored errors. The external path is executable,
has more positive filters overall, and is much cheaper in this observed set.

Among16 single-pair model-input requests, ACT/CROWN positive counts are8/11
(ACT-only1,CROWN-only4); among14 multi-pair requests they are3/2
(ACT-only2,CROWN-only1). These explanatory strata are small and observed;
they do not establish general superiority on multi-route inputs.

All-request mean costs are138.11 versus4.23 seconds, totals4143.18 versus126.85
seconds. They include imports, route analysis, checkpoint/tensor loads,
cross-environment handoff and all bounds; common immutable raw-input preparation
is disclosed and excluded equally, as are subsequent audits. Five300-second
timeouts are included. Different outcomes and numerical contracts prohibit a
same-result acceleration claim; there is nevertheless a clear cost challenge
from this working external path. No best-of-two portfolio score is reported.

## Consequence for the project

This completes the requested bounded external comparison, not the independent
external full-dynamic-model or high-accuracy strict-certification goals. It
supports limited complementarity, not ACT dominance. The hundred-input internal
confirmation, relationship ablation and rational request proof remain separate
evidence. Do not expand this ten-input set, tune25%, or add an outcome-selected
portfolio. A new model or algorithm experiment needs its own frozen design.
