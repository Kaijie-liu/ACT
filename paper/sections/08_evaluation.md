# Evaluation

We evaluate four distinct questions: whether the frozen method improves complete
request outcomes on new inputs; whether shared-input relationships contribute;
how an executable external-backend path compares; and what a separately checked
request proof establishes. These experiments have different cohorts and evidence
contracts. We do not pool them or use later follow-ups to amend earlier gates.
The complete historical tables, component studies and failed gates are retained
in [the evaluation appendix](../appendices/historical_evaluation.md).

## Experimental contract

The main models are three frozen bal010 training runs: output-layer
selected-softmax top-2 with eight experts and approximately 48% clean accuracy.
Verification uses CPU/float64, all tie-legal unordered pairs and a clipped
2/255 input box. Each comparison arm independently computes its route coverage
and pays for required preprocessing and solving. A 300-second outer request
cap includes unsuccessful work; method order is rotated. No historical route
census or proof answer is supplied free to another arm.

`SAFE` in HZ tables means acceptance under the frozen HZ/HiGHS numerical policy.
`UNSAFE` requires a domain-valid complete-model replay. `UNKNOWN` and `TIMEOUT`
remain in the denominator. A non-outward-rounded CROWN positive is a numerical
filter, not interchangeable formal SAFE. The rational evidence experiments
have a separate conditional-on-lowering contract. Auditing identities and
obligation coverage is distinguished from independently checking numerical
lower bounds.

## New-input confirmation of the frozen schedule

The schedule was developed on observed inputs and separately examined on a
30-input cohort. We then froze 100 additional common clean-correct inputs,
excluding previous verification indices. Selection used dataset order and clean
correctness, not route complexity or expected outcome. Earlier whole-test
telemetry is disclosed: the novelty is in verification endpoints, not images
never inspected in any form. The three configurations and 25% allocation were
unchanged before the new experiment. The primary comparison is matched
monolithic, which independently obtains the same cheap scoped facts; legacy
monolithic remains a secondary strong reference.

| Model | Adaptive SAFE / solved | Matched monolithic SAFE / solved | Legacy monolithic SAFE / solved |
|---|---:|---:|---:|
| Seed 0 | 59 / 89 | 50 / 76 | 46 / 68 |
| Seed 1 | 57 / 81 | 47 / 65 | 45 / 63 |
| Seed 2 | 63 / 86 | 59 / 78 | 50 / 68 |

Each cell has denominator 100; solved is SAFE plus replayed UNSAFE, not
certified accuracy. All 900 calls are accounted for: 739 complete packages,
198 UNSAFE replay records and 161 outer timeouts. All 300 adaptive/matched
common-fact pairs agree, including snapshots retained before later termination.
The independent structural re-audit reproduces the archived result.

Relative to matched, adaptive gains 23 SAFE and 37 solved model-input pairs,
with no losses. All 23 SAFE gains have multiple exact legal pairs: two finish
in Tier 1 and 21 in F0, where reuse is recorded. Participation of reuse is not
itself a per-case causal ablation. Relative to legacy, SAFE gains/losses are
40/2, for a net 38; the two genuine losses remain visible. Thus the result is
not universal solution-set dominance.

The primary mean SAFE difference is 7.67 percentage points, with the frozen
input-clustered descriptive bootstrap interval [4.67, 11.00]. Resampling keeps
all three fixed models within an input block. The 300 model-input pairs are
not independent images and these intervals do not infer over unseen model
families. Mean all-request costs are 81.71, 111.57 and 153.27 seconds for
adaptive, matched and legacy respectively. Small per-model paired medians and
larger mean savings indicate concentrated long-tail benefits, not uniform
same-outcome speedups.

Evidence: `act/pipeline/moe/results/schedule_confirmation_100_review_20260914_r1.json`.
The independent 30-input study and historical 2/3 composite result remain
separate in the appendix. This is a new-input comparison of internal
configurations, not a claim of superiority over independent public tools.

## Direct relationship ablation

On ten previously observed inputs and the three fixed models, we freeze the
same schedule and change only joint expert factor sharing to a block-diagonal
independent product. The latter retains each expert's marginal constraints but
allows different input-factor assignments, forming a sound outer enlargement.
All 60 calls are retained: 48 packages, 16 UNSAFE replay records and 12 outer
timeouts. Common facts agree in all 30 pairs.

| Model | Shared SAFE / solved | Independent product SAFE / solved |
|---|---:|---:|
| Seed 0 | 4 / 6 | 4 / 5 |
| Seed 1 | 3 / 6 | 2 / 5 |
| Seed 2 | 4 / 8 | 2 / 5 |

There are three SAFE gains and five solved gains without losses. Seed1/4018
and seed2/4014 have three and two legal pairs respectively; the independent
arm completes its relaxation but remains UNKNOWN. These two cases directly
support a precision benefit of joint relationships. Seed2/4018 has one pair
and faces a solver-limit UNKNOWN in the independent arm, leaving a budget
confound. The outer product also changes variable count and solver cost; this
observed follow-up is not a new confirmation or a claim that sharing is
necessary on every request.

Evidence: `act/pipeline/moe/results/relation_ablation_review_20260914_r1.json`.

## Complete-cost executable external path

We compare ACT adaptive with **ACT route frontend + plain CROWN whole-box
variable-weight static pairs** on ten observed inputs and three models. The
external path recomputes complete route coverage and retains the router,
experts and changing softmax weights in each static graph. It does not use
pair-guard input conditioning. Whole-box success for all feasible pairs is a
valid sufficient path; failure of that stronger condition is not a model
counterexample. This is neither a raw dynamic-model encoding nor full
alpha-beta-CROWN/BaB.

| Model | ACT SAFE / UNSAFE / UNKNOWN / TIMEOUT | CROWN numerical positive / UNKNOWN | Mean request seconds ACT / CROWN |
|---|---|---|---:|
| Seed 0 | 4 / 2 / 2 / 2 | 5 / 5 | 139.00 / 4.27 |
| Seed 1 | 3 / 3 / 2 / 2 | 4 / 6 | 152.64 / 4.23 |
| Seed 2 | 4 / 4 / 1 / 1 | 4 / 6 | 122.68 / 4.18 |

Both arms have denominator ten per model. Total positive outcomes are 11
HZ-policy SAFE versus 13 CROWN numerical filters: eight shared, three ACT-only
and five CROWN-only. All three ACT-only cases have completed nonpositive CROWN
queries, not frontend errors or timeouts. Two are multi-route. All five
CROWN-only cases are ACT solver-limit UNKNOWN. All-request mean costs are
138.11 versus 4.23 seconds. The external path is substantially cheaper and has
more positives overall in this set; ACT's additional counterexample outcomes
must not be counted as extra safety certificates. The external probes are not
a matched strong attack.

The complete cost includes loading, cross-environment imports, graph building,
all pair bounds and terminal submission. Immutable raw-input preparation is
separately disclosed and excluded equally; subsequent audits are separate.
This experiment establishes limited complementarity, not ACT dominance or
interchangeable numerical guarantees. Evidence:
`act/pipeline/moe/results/external_pair_comparison_review_20260914_r1.json`.

## Independent request-level evidence and transfer limits

On the earlier fixed seed0/index3000 request, a registered sequence of controls
checks all 18 output obligations. The final R3 reconstructs the weighted
McCormick LP from the stored pre-F0 shared expert HZ with rational arithmetic;
15 obligations use membership facts and three use residual LPs. It removes
trust in floating F0 construction, not in network-to-HZ propagation, guards
or route exclusions. R3's last-step time excludes previously paid R1/R2 work
and is not a full fresh-generation cost. The original R1 UNKNOWN remains.

We then register a fresh-generation protocol on all three ACT-only external
cases without borrowing any previous proof bounds, route census or common
facts. These cases are post-selected mechanism checks, not a success-rate
sample. The procedure uses 10-second proposal limits, an 1,800-second generation
cap, and only universal/dyadic gate ranges derived from checked router order.
There is no outcome-dependent sigmoid refinement or case replacement.

| Case | Legal pairs | Positive reused / residual / unresolved obligations | Complete request | Generation / separate check seconds |
|---|---:|---|---|---:|
| Seed0/4029 | 1 | 4 / 1 / 4 of 9 | UNKNOWN | 111.87 / 29.62 |
| Seed1/4018 | 3 | 25 / 1 / 1 of 27 | UNKNOWN | 90.23 / 29.59 |
| Seed2/4014 | 2 | 14 / 1 / 3 of 18 | UNKNOWN | 141.88 / 41.37 |

All generations complete without timeout. A valid checked negative bound is
not a positive certificate: 46 of 54 obligations are positive, but none of the
three requests is completely discharged. This neither refutes their original
HZ-policy SAFE nor proves that integer reasoning is the only missing ingredient.
Binary relaxation, gate enclosure and other abstraction gaps are not separately
identified by this experiment. The independent review records each obligation,
remaining trusted components, source binary counts and substantial serialized
evidence sizes. No partial result is promoted to request SAFE.

Evidence: `act/pipeline/moe/results/request_lp_rational_review_20260914_r3.json`
and the separate ACT-only transfer archive
`act/pipeline/moe/results/request_lp_act_only_review_20260915_r1.json`.

## Convolutional transfer and evidence-mode limits

The separately preregistered convolutional family has four independent experts,
155,052 parameters and output-layer selected-softmax top-2 routing. Seed 17 was
trained for 100 fixed epochs; the earliest validation-maximizing checkpoint
(epoch 89) achieves 68.08% validation and 67.06% full-test accuracy. Selection did
not use verification outcomes. Full-shape conformance and training reviews are
recorded in `conv_training_review_20260915_r1.json`. Failed initial smoke and
budget repairs remain in [the transfer chronology](../appendices/transfer_evidence_history.md);
they are not additional performance studies pooled into the final comparison.

The V2 execution protocol completes 90 requests on 30 previously unexecuted
inputs, with the same checkpoint, 2/255 and 300-second request cap. There are
90 terminal records, 30 matching common-fact pairs and 35 full-model witness
replays concerning 18 distinct inputs. No outer watchdog termination is missing
from the ledger; the ACT TIMEOUT entries below are internal budget terminals.

| Method | Positive (grade) | Replayed UNSAFE | UNKNOWN | TIMEOUT | Mean seconds |
|---|---:|---:|---:|---:|---:|
| Adaptive | 0 HZ-policy | 17 | 0 | 13 | 188.31 |
| Matched monolithic | 0 HZ-policy | 11 | 0 | 19 | 217.50 |
| ACT routing + plain CROWN | 1 numerical filter | 7 | 22 | 0 | 4.40 |

Every denominator is 30. All six adaptive-only decisions are counterexamples,
not additional certificates. The external path is cheaper and produces the
only numerical positive. This study does not establish cross-family
certificate gains. Evidence: `act/pipeline/moe/results/conv_full_v2_review_20260915.json`.

A separate optional evidence-mode study on 20 new convolutional clean-correct
inputs also retains 2/255 and 300 seconds. Matched V2 returns 10 UNSAFE and 10
TIMEOUT; evidence mode returns 20 TIMEOUT; CROWN returns 7 UNSAFE and 13 UNKNOWN.
No arm produces a positive request of its respective evidence grade. Mean
all-request costs are 220.68, 283.09 and 4.08 seconds. The evidence path does not
perform an equivalent counterexample search. The 17 UNSAFE records identify 12
distinct unsafe inputs; the remaining eight are not thereby certified safe.

Saved evidence distinguishes execution failure from an unclosed bound. Seven
requests have complete local prechecks but nonpositive bounds, six have missing
obligations and seven have no completed precheck. These observations cannot
establish that cheaper serialization or more solver time would yield positives.
The cohort is closed; no later control relabels its terminals.
Evidence: `docs/general_evidence_execution_v1_results.json`.

## Complete proofs versus complete experimental requests

Input 98 supplies a complete conditional convolutional proof, but is an observed
single-pair control, not a new route-changing certificate. Its pre-F0 rational
construction checks eight residual output obligations plus one reused fact.
The minimum is `199593373867685/1125899906842624` (approximately 0.1772745).
It removes trust in floating F0 construction while retaining network-to-HZ,
guard and route-exclusion assumptions. The portable bundle checks without a
checkpoint, dataset, solver or historical directory. Original packaging reduced
428,185,262 proof-dependency bytes to a 7,181,520-byte bundle; the isolated check
took 30.222 seconds. These costs exclude original bound generation.

In a separately frozen development comparison, fresh evidence generation,
packaging and checking on input 98 completed in 185.317 seconds; matched V2
returned TIMEOUT in 295.697 seconds. This one fixed-order postselected control
establishes budgeted feasibility, not a population speedup. The subsequent
20-input negative study above is the relevant test of that mode's transfer.
Evidence: `docs/portable_conv_proof_v1_review.json` and
`docs/optional_evidence_dev_v1_review.json`.

The first-family ACT-only transfer controls likewise remain 0/3 complete checked
requests despite 46/54 positive obligations. Input 16 remains unclosed on two
properties. A checked lower bound at or below zero says only that this bound
does not prove the obligation; it does not establish an unsafe model or an
unavoidable LP gap. Auxiliary feasible-point diagnostics have not removed that
uncertainty: the closed four-LP SoPlex comparison passed input readback but
admitted no independently checkable feasible point under the frozen bit cap.
Native optimality reports are not proofs. This limit and the arithmetic
development history belong in the appendix, not in the main performance claim.

Taken together, the experiments distinguish **method gain, relation mechanism,
external complementarity and conditional proof capability**. None substitutes
for the others. High-accuracy real-scale strict certificates, cross-family
route-changing certificate gains and independent raw-dynamic-model competition
remain open. All four-state tables can be
[rebuilt from committed reviews](../results/main_tables.md) without private
models; rebuilding the tables is not independently reproving their SAFE bounds.
