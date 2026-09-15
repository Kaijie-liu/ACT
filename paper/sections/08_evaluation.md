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

## Scope of the empirical conclusion

A second, preregistered convolutional output-top-2 family has now completed
training: four independent convolutional experts, 155,052 parameters, seed 17 and
100 fixed epochs. The earliest validation-maximizing checkpoint (epoch 89)
obtains 68.08% on 5,000 validation inputs and 67.06% on the full 10,000-image test
set. An independent process checks the training/selection identities and
replays both concrete evaluations. Full-shape ACT and external static-pair
conformance controls pass. A separately frozen six-call trained-model smoke
(two observed inputs, three arms, 300 seconds per complete request) now ends
with adaptive one replayed UNSAFE and one outer TIMEOUT, matched monolithic
two outer TIMEOUTs, and plain-CROWN two completed UNKNOWN records. The terminal
audit passes, including both common-fact comparisons, but the smoke gate fails:
monolithic has no complete non-error record. The 90-call full experiment has
not started at that historical gate. This was a budget-conformance limitation, not a measured
cross-architecture advantage or evidence that the timed-out properties are
safe. Evidence: `act/pipeline/moe/results/conv_training_review_20260915_r1.json`
and `act/pipeline/moe/results/conv_three_arm_smoke_review_20260915_r1.json`.
A separately frozen timing follow-up on one old monolithic smoke request,
without changing the algorithm or300-second cap, records45.02s of pair
propagation,14.01s of union construction and208.61s across eight returned
property-native calls, all solver-limit UNKNOWN. The ninth call is externally
censored. Its native allocation exceeds the remaining request time by1.56s
because construction occurs after allocation. This is a measured accounting
defect, not evidence that repairing it yields positive bounds; the profiled
request and original smoke remain failures. Instrumentation costs are charged,
and this single diagnostic is not a comparative speed result. Evidence:
`act/pipeline/moe/results/conv_f0_timing_review_20260915_r1.json`.

A separately frozen V2 budget adapter subsequently passes old-input smoke;
the new full protocol then completes all 90 requests on the original 30
previously unexecuted inputs, with no outer timeout. Both ACT arms use V2;
plain CROWN retains its original configuration. Fresh review matches the two
automatic audits: 90 complete records, 30 equal common-fact pairs and 35
full-model witness replays (18 distinct inputs).

| Method (30 inputs) | HZ SAFE / replayed UNSAFE / internal TIMEOUT | CROWN positive / replayed UNSAFE / UNKNOWN | Mean request seconds |
|---|---|---|---:|
| Adaptive | 0 / 17 / 13 | — | 188.31 |
| Matched monolithic | 0 / 11 / 19 | — | 217.50 |
| ACT routing + static weighted CROWN | — | 1 / 7 / 22 | 4.40 |

All six adaptive-only decisions relative to matched are UNSAFE, not extra
certificates. Its paired mean time saving is 29.19s but the paired median
difference is +0.014s, so no uniform speedup is claimed. CROWN's index98
numerical positive is not formal SAFE; its index113 witness is missed by
both ACT arms. This convolutional transfer therefore provides no HZ SAFE or
cross-architecture certificate advantage under this frozen protocol.
Evidence: `act/pipeline/moe/results/conv_full_v2_review_20260915.json`.

A separate saved-log analysis finds419 returned F0 property queries, all
solver-status1 limits; none is a completed nonpositive relaxation result.
All32 ACT timeouts reach F0 after exact route enumeration. Adaptive leaves14
pair-property scopes unqueried across four multi-pair requests; monolithic
queries every non-reusable scope in its timeout subset. There are54 positive
diagnostic full-objective dual records, including sign coverage of all
obligations on two single-pair inputs (16 and98, the latter with one interval
fact). They remain unaccepted under the frozen optimal-status policy and are
not independently checked certificates. The observations motivate proof-bound
checking and budget studies, not retrospective SAFE promotion or a claim that
more time will solve every timeout. Evidence:
`act/pipeline/moe/results/conv_full_v2_obligations_20260915.json`.

Two subsequent post-selected evidence controls capture the first residual
F0 property at inputs16 and98, under the same construction recipe, before any
weighted-property MILP solve. Continuous relaxations of these newly stored HZ
objects yield exact-rational checked positive bounds of3.6908 and5.7124;
independent checker processes do not load numerical solvers. This demonstrates
sign-sufficient LP evidence for two supplied obligations, not whole-network or
request SAFE, and not equality with the unsaved historical MILP coefficients.
The trust boundary still includes floating F0 construction and upstream HZ/
guard lowering. A declared procedural deviation is retained: the local freeze
commit preceded execution, but a GitHub server error delayed successful remote
publication until after launch. These are feasibility controls, not an
unqualified protocol-compliant confirmation. Evidence:
`act/pipeline/moe/results/conv_sign_lp_review_20260915_r1.json`.

A separately frozen all-obligation follow-up on the same two observed
convolutional controls now checks18/18 required properties. Input16 has seven
positive supplied-HZ LP bounds and two nonpositive lower bounds (-0.27375 and
-0.12559 for competitors7 and9), so remains unclosed. Input98 has eight
positive residual LP bounds plus one independently checked scoped interval
fact, yielding a complete conditional request bound of0.1772745. Its two
isolated checks and fresh review agree. The new remote-publication-before-run
gate passes. All-stage proof-study costs are98.35s and60.13s, not a matched
speed comparison with the original300-second verifier. The trust boundary
still includes network/guard lowering, route exclusions and floating F0
construction; the previous pre-F0 rational construction result does not
transfer automatically. These post-selected single-pair controls establish
one conditional complete request proof, not deployed-float SAFE, additional
route-changing coverage, or high-accuracy-scale certification. Original
TIMEOUTs and the production acceptance policy remain unchanged. Evidence:
`act/pipeline/moe/results/conv_request_sign_lp_review_20260915_r1.json`.

For input98 only, a new pre-F0 rational-construction experiment removes the
floating weighted-lowering assumption while retaining all nine positive
obligations. It uses26 LP proposals (two router order,16 disagreement and eight
weighted), plus the interval reuse fact. The more conservative dyadic gate
range [1/2,1] suffices; residual bounds range from1.615575 to6.748146, while
the request minimum remains0.1772745. Two isolated checks and fresh review
agree. Its179.01s all-stage proof-study time is not a comparative300s verifier
measurement. An initial type-conversion failure before any LP proposal is
preserved; the repaired version only canonicalizes NumPy +/-1/0 property
scalars, without changing mathematics or ranges. This is one single-pair
conditional convolutional request proof with fewer trusted components, not
high-accuracy strict certification, route-changing coverage or production SAFE.
Input16's two unclosed obligations are not revisited. Evidence:
`act/pipeline/moe/results/conv_pre_f0_review_20260915_r2.json`.

A separately frozen engineering comparison then assigned the original matched
V2 path and an optional evidence path300s each on this same observed input98.
Matched returned TIMEOUT in295.697s; the optional path completed all nine
conditional proof obligations in185.317s, including fresh capture,26 proposals,
local aggregation, portable packaging and an independent isolated check. A
separate archival audit reproduced the proof. This one postselected positive
control with fixed arm order establishes budgeted feasibility, not population
speedup, a new route-changing result or production acceptance-policy change.
The earlier179.01s study remains a different execution, not a benchmark value.
The optional adapter is deliberately restricted; broader request support and
net benefit remain open. Evidence: `docs/optional_evidence_dev_v1_review.json`.

The confirmation supports new-input benefits against the registered internal
comparators; the ablation supports a relation mechanism; the external study
exposes both complementarity and a substantial cost challenge; the proof
studies reduce trust in a part of the chain while exposing transfer limits.
These are different contributions, not four measurements of one success rate.
Cross-architecture benefits, high-accuracy deep-model strict certificates and
independent full-dynamic-model external competition remain unachieved. Historical
negative results are retained in the appendix rather than replaced by this
selection of primary questions.
