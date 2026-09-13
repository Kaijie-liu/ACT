# Hundred-input confirmation: independently reviewed results

Completed 2026-09-14 01:50 Sydney; execution HEAD
`bc0791976b00879c28c268692aecc5854b3bc091`. The 100-input full run took
about 28.89 hours including its final audit (smoke separate).
All900 requests and all9 old-input smoke requests completed. A separate process
reran the frozen structural auditor and concrete UNSAFE replays and reproduced
the saved summaries exactly. The review also verifies hashes of all Python
execution sources at the frozen HEAD; newly added archival tools are excluded
from that identity, not changes to any old execution source.

Primary artifact: `results/schedule_confirmation_100_review_20260914_r1.json`.
Reconstruct with `python -m act.pipeline.moe.review_schedule_confirmation_100 --check`
in act-py312. This reads records/replays concrete witnesses, not new optimization
queries. `--write` refuses to overwrite an existing archive. Original roots,
thresholds, manifests, failed attempts and the thirty-input result are untouched.

## Complete results

Each method/model denominator is100; all costs include outer-capped requests.
S/U/?/T mean SAFE, concretely replayed UNSAFE, UNKNOWN and TIMEOUT.

| Model | Adaptive S/U/?/T | Matched S/U/?/T | Legacy S/U/?/T | Mean seconds A/M/L |
|---|---|---|---|---|
| seed0 | 59/30/9/2 | 50/26/13/11 | 46/22/2/30 | 70.38/101.79/153.61 |
| seed1 | 57/24/9/10 | 47/18/12/23 | 45/18/4/33 | 97.53/131.74/162.77 |
| seed2 | 63/23/8/6 | 59/19/7/15 | 50/18/1/31 | 77.21/101.17/143.44 |

Totals A/M/L: SAFE179/156/141, UNSAFE77/63/58, solved256/219/199,
UNKNOWN26/32/7, TIMEOUT18/49/94. These are300 model-input pairs/arm, only100
shared distinct images. SAFE ratios are not whole-test-set certified accuracy.
Primary matched SAFE gains9/10/4, losses0/0/0; solved net13/16/8, no losses.
Legacy SAFE gains14/13/13, losses1/1/0; solved gains22/19/18, losses1/1/0.
Positive net benefit against legacy is NOT set dominance.

Frozen input-block bootstrap, all three models carried together, 10,000 draws,
seed20260912, descriptive95% percentile intervals in percentage points:

| Comparison | SAFE mean difference / interval | Solved mean difference / interval |
|---|---|---|
| Adaptive - matched (primary) | +7.67 [4.67,11.00] | +12.33 [8.67,16.33] |
| Adaptive - legacy (secondary) | +12.67 [8.67,17.00] | +19.00 [14.33,24.00] |

These are unadjusted descriptive intervals for these fixed models/cohort, not
family-wise tests, a random-training-seed population guarantee or a universal
advantage. The thirty-input study stays separate, not pooled to get these intervals.

## Cost and interpretation

| Model | Mean paired A-M / A-L seconds | Median paired A-M / A-L seconds |
|---|---:|---:|
| seed0 | -31.41 / -83.23 | -0.322 / -46.318 |
| seed1 | -34.21 / -65.24 | -0.151 / -21.614 |
| seed2 | -23.96 / -66.23 | -0.076 / -22.572 |

Overall observed means A/M/L are81.71/111.57/153.27 seconds. Small primary
medians and larger means locate savings in the tail, not uniform acceleration.
State-conditioned cost tables are in the supplement; different arms' SAFE
subsets differ, so those tables are descriptive, not same-instance speedups.
UNKNOWN is not a quick successful decision. There is no substitution of earlier
thirty-input timings for this run.

## All23 primary SAFE gains: route and decision source

All23 have multiple exact tie-legal pairs: fourteen with2, eight with3, one with4.
Two finish at Tier1 gate elimination; twenty-one finish at F0 and each of those
records at least one scoped reused property. This is within-run source accounting,
NOT an independent reuse-off causal ablation or counts of optimal MILP solves.
All adaptive multi-route SAFE counts are20/16/17, matched11/6/13,
legacy6/3/4. Route-changing here means multiple exact `ANY_LEGAL_TOPK` sets,
not an independently found literal GPU dispatch-flip witness.

| Model | Rank | Dataset index | Legal pairs | Decision | Reused / remaining recorded F0 rows | Matched status |
|---|---:|---:|---:|---|---:|---|
| seed0 | 0 | 4088 | 2 | TIER2_F0 | 7 / 11 | UNKNOWN |
| seed0 | 17 | 4145 | 2 | TIER2_F0 | 5 / 13 | UNKNOWN |
| seed0 | 27 | 4172 | 2 | TIER2_F0 | 7 / 11 | TIMEOUT |
| seed0 | 33 | 4188 | 2 | TIER2_F0 | 9 / 9 | UNKNOWN |
| seed0 | 43 | 4223 | 2 | TIER2_F0 | 7 / 11 | UNKNOWN |
| seed0 | 45 | 4232 | 3 | TIER2_F0 | 12 / 15 | UNKNOWN |
| seed0 | 48 | 4236 | 2 | TIER2_F0 | 3 / 15 | UNKNOWN |
| seed0 | 78 | 4340 | 2 | TIER2_F0 | 6 / 12 | UNKNOWN |
| seed0 | 86 | 4361 | 2 | TIER2_F0 | 9 / 9 | UNKNOWN |
| seed1 | 6 | 4104 | 3 | TIER2_F0 | 12 / 15 | UNKNOWN |
| seed1 | 11 | 4128 | 2 | TIER2_F0 | 6 / 12 | TIMEOUT |
| seed1 | 15 | 4141 | 2 | TIER2_F0 | 4 / 14 | UNKNOWN |
| seed1 | 43 | 4223 | 3 | TIER2_F0 | 7 / 20 | TIMEOUT |
| seed1 | 50 | 4248 | 2 | TIER2_F0 | 4 / 14 | UNKNOWN |
| seed1 | 52 | 4256 | 2 | TIER2_F0 | 12 / 6 | TIMEOUT |
| seed1 | 67 | 4311 | 3 | TIER2_F0 | 7 / 20 | TIMEOUT |
| seed1 | 68 | 4314 | 2 | TIER2_F0 | 1 / 17 | TIMEOUT |
| seed1 | 89 | 4371 | 4 | TIER1_GATE_ELIMINATION | 0 / 0 | TIMEOUT |
| seed1 | 99 | 4389 | 2 | TIER1_GATE_ELIMINATION | 0 / 0 | TIMEOUT |
| seed2 | 64 | 4297 | 3 | TIER2_F0 | 19 / 8 | UNKNOWN |
| seed2 | 69 | 4316 | 3 | TIER2_F0 | 11 / 16 | UNKNOWN |
| seed2 | 85 | 4360 | 3 | TIER2_F0 | 13 / 14 | TIMEOUT |
| seed2 | 89 | 4371 | 3 | TIER2_F0 | 20 / 7 | UNKNOWN |

## Legacy gains and retained losses

seed0 gained SAFE dataset indices: 4088, 4145, 4154, 4172, 4188, 4214, 4218, 4223, 4232, 4236, 4314, 4340, 4361, 4371.

seed1 gained SAFE dataset indices: 4104, 4123, 4128, 4141, 4187, 4223, 4229, 4248, 4256, 4311, 4314, 4371, 4389.

seed2 gained SAFE dataset indices: 4088, 4103, 4146, 4154, 4186, 4274, 4279, 4297, 4316, 4317, 4339, 4360, 4371.

| Model | Rank / index | Legal pair | Adaptive result / seconds | Legacy result / seconds | Unfinished adaptive property rows |
|---|---|---|---|---|---|
| seed0 | 20 / 4150 | [[4, 5]] | UNKNOWN_MONOLITHIC_SOLVER_LIMIT / 248.88 | SAFE_MONOLITHIC_WEIGHTED_RANGE / 259.11 | [0, 3] |
| seed1 | 16 / 4142 | [[2, 6]] | UNKNOWN_MONOLITHIC_SOLVER_LIMIT / 244.92 | SAFE_MONOLITHIC_WEIGHTED_RANGE / 270.52 | [1] |

The two legacy-only SAFE are single-pair cases: adaptive returns a complete
UNKNOWN package with `UNKNOWN_MONOLITHIC_SOLVER_LIMIT`, not an outer kill.
Both record nine weighted query rows and zero reused pair properties (available
fact inventory is not sufficient to justify a pair reuse). Property-level status1
and unfinished row indices are preserved in the supplement. Legacy spends more
time and completes the corresponding obligations. This identifies stopping
location under different registered schedules, not a unique causal proof that
preprocessing, budgeting or random timing alone explains the loss.

## Package/timeout/snapshot accounting

| Arm | Package + snapshot | Package, no snapshot | No package + snapshot | No package, no snapshot |
|---|---:|---:|---:|---:|
| Adaptive | 282 | 0 | 18 | 0 |
| Matched | 251 | 0 | 49 | 0 |
| Legacy | 0 | 206 | 0 | 94 |

739 complete packages plus161 outer TIMEOUT terminals account for all900.
All600 scheduled snapshots survive, giving300/300 equal common-fact pairs;
67 belong to outer-killed requests. Legacy has no snapshot by design. Missing
packages never become positive results; each of the161 missing-package
terminals is listed with hash, job ID and available snapshot. The review
compares every terminal JSON with its append-only ledger row. The saved audit
and ledger/runtime hashes bind the full900, not just the discordant SAFE.
All198 method-run UNSAFE witnesses replay; this is not198 distinct images.

## Scientific scope and next separate work

This is new-verification-endpoint confirmation of net SAFE/solved benefit under
the fixed internal comparisons, not merely a ten-image development signal.
It does NOT revise the old2/3 composite result, nor establish high-accuracy or
cross-architecture evidence, external-tool dominance or whole floating-point
execution proof. HZ/HiGHS SAFE retains the frozen numerical acceptance policy.
Structural re-audit is not independent reproduction of every bound.

The primary comparison isolates execution organization with common HZ/guard/F0
facts, not HZ shared-input relationships alone. Keep the25% policy frozen and
do not add200/500 inputs to chase stronger intervals. Next stages are separately
specified in `docs/post_confirmation_workstreams.md`: a relation-only ablation,
one external semantic-compatibility path, and request-level independent bound
evidence. None is silently merged into these900 endpoints.
