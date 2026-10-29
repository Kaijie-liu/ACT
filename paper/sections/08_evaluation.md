# Evaluation

## Complete-request cascade primary comparison: practical gate not met

The seventh-revision B4 primary cohort contains 15 fixed multilayer MLP models
(L2E8, L3E8, L2E16, residual R2E8 and R3E8, three seeds each) and 20 shared
clean-correct CIFAR-10 inputs. Inputs are selected in ascending order outside
the bound historical exclusion union, without using routes or verification
outcomes. Earlier whole-test telemetry is disclosed. Each model-input object
has two independently executed 300-second calls at radius 2/255: coupled
GenBaB with prefix pruning, and the same engine/frontend with M-A enabled.
The optional decoupled phase receives at most 120 seconds per history inside
the request deadline; an unfinished phase falls back only with remaining time.
M-B and M-C fail diagnostic admission and are absent from both arms.

| Complete primary calls, 300 per arm | Coupled off | Cascade on |
|---|---:|---:|
| Numerical POSITIVE | 116 | 122 |
| Arm-generated UNSAFE_REPLAYED | 0 | 0 |
| UNKNOWN | 8 | 5 |
| TIMEOUT | 176 | 171 |
| ERROR | 0 | 2 |
| Mean full-request seconds | 189.287 | 192.983 |
| Median full-request seconds | 288.350 | 288.337 |
| Total full-request seconds | 56,786.217 | 57,894.897 |

The positive intersection is 113, with nine gains and three losses. The net
effect is 6/300 = **2 percentage points**. The frozen 10,000-replicate bootstrap
resamples 20 input blocks carrying all 15 models and both arms; its 95%
percentile interval is **[0,4] points**. The primary practical threshold of at
least five points fails, and the interval's lower endpoint is not positive.
The median cost ratio, 0.999955, meets the cost condition, but near-equal medians
in a timeout-dominated table do not establish general speedup. The registered
decision is STOP, not an invitation to collect more samples or change the phase
cap. This estimates behavior on the fixed models, not a population of model
architectures or 600 independent observations.

| Family, 60 objects | Off positives | On positives | Gain / loss |
|---|---:|---:|---:|
| L2E8 | 24 | 28 | 5 / 1 |
| L3E8 | 27 | 26 | 0 / 1 |
| L2E16 | 53 | 53 | 0 / 0 |
| R2E8 | 7 | 10 | 4 / 1 |
| R3E8 | 5 | 5 | 0 / 0 |

Post-hoc trace analysis identifies a local mechanism effect: all nine gains use
26 decoupled history closures and no coupled fallback. Of 52 accepted endpoints,
48 close through initial bounds and four through nonlinear branch-and-bound.
Yet the gains save 1,530 seconds while 113 shared-positive objects add 2,459
seconds. Losses add 334 seconds; the 175 pairs with neither arm positive use
155 fewer seconds, which is not successful acceleration. These four groups
account for the overall increase of 1,109 seconds. Two losses spend much of
their budget in unfinished decoupled phases; a third completes 16 such phases
but cannot finish the remaining histories before the deadline. This does not
identify a unique relaxation deficiency or prove a counterfactual phase cap.

The separately executed minus-M-A ablation uses the frozen six-model subset
and the already paid MAIN on runs as its full-arm comparator. It gives 64 versus
68 positives, six gains and two losses, a net 3.33 points with interval
[-0.833,8.333] points. Median full-request costs are 35.894 versus 132.610
seconds. This subset cannot replace the primary decision. Although MAIN off and
minus-M-A implement the coupled configuration, their separately paid runs differ
on four terminal outcomes, including one positive becoming a timeout. This
execution variability is reported rather than counted as pure mechanism gain.

The public PGD records contain 26 valid full-model counterexamples, with no
conflict against either arm's positive. They remain public evidence, not free
UNSAFE credit supplied to a verifier. Public route census observes at least two
native histories on 12 objects; that stratum has zero gains and zero losses.
The other 288 objects contain all nine gains and three losses, but one observed
history does not prove route stability. Exact reachable-history counts remain
unknown. The frozen 4/255 cohort is reported separately below; the single-layer
queue is still running. Neither replaces the primary denominator or gate.

All failed attempts are retained. An initial publication/cleanup failure is
charged at its observed upper cost of 301.579 seconds and remains ERROR; a
later deadline-signal error is also retained. Only unexecuted calls continue.
Disclosed supervision repairs reserve more time for cleanup in both arms,
reducing worker time from 294 to 288 seconds without changing the 300-second
request cap or mathematical acceptance rules. The engine identity is unchanged.
Primary stage accounting is 115,239.317 seconds, including 357.731 seconds of
selection/public work and the remaining startup, reception, repair and audit
costs in addition to request costs. The ablation stage costs 17,570.300 seconds.
Additional read-only reviews are recorded separately; they cannot repair an
online failure or independently establish CROWN's arithmetic guarantees.

The frozen source/terminal/coverage audit and input-block statistics have been
independently recounted, not the numerical bounds re-proved. All positives in
this table are **GenBaB numerical filters**, not source-complete or native
floating-execution certificates. Evidence: [protocol, complete tables and
execution interpretation](../../docs/moe_aware_b4_20261026.md) and
[bound records and independent recount](../../docs/moe_aware_b4_records_20261027.json),
keys `main_and_ablation_review` and `main_gain_loss_trace_review`.

## Double-radius cohort: small local benefit, no observed routing trend

The registered 4/255 cohort completes all 240 calls on six fixed models and
twenty shared inputs. Coupled/cascade numerical positives are 6/8; TIMEOUTs are
114/112, with no complete UNKNOWN, arm UNSAFE or ERROR. All six baseline
positives are retained; two gains give +1.667 percentage points, with the
registered input-block interval [0,4.167]. Both gains are L2E16 seed 1; all four
endpoints close in initial propagation without fallback. Each request has one
surviving checked numerical history, not a proof of native route variation.

The thirteen objects with multiple observed histories show no gain or loss;
the two gains have only one observed history. On the same six-model subset at
2/255, positives were 65/68. Greater observed routing variability therefore
does not exhibit the hypothesized greater net benefit in these records. Exact
legal-history counts remain unknown; the joint change in radius, robustness
and execution difficulty prevents a unique causal interpretation.

Public PGD supplies 31 replayed counterexamples, all among the 112 objects
where both verification arms time out, and none conflicts with a positive.
Keep the original 120-pair denominator and original arm outcomes. All-request
means are 275.446/273.588 seconds and medians 288.784/288.775; the median paired
difference is -0.002610 seconds. The total stage cost is 66,220.116 seconds,
including auxiliary work and orchestration. This is not universal acceleration
or a replacement primary success. The 240 receipts, source identities and
input-block bootstrap were independently recounted in 6.810 seconds, with
0.101 seconds for gain-trace extraction; no bound was independently re-proved.
Evidence: `radius_final_review` in the same [B4 record](../../docs/moe_aware_b4_records_20261027.json).

## Multilayer mechanism diagnostics: no admitted M-B or M-C benefit

The completed seventh-revision diagnostic tests two optional mechanisms before
any complete-request performance claim. M-B compares guard-dual optimization
with its zero-guard comparator on 11 fixed histories. M-C compares no partition,
two closed first-margin cells, and four cells on 27 fixed multilayer histories
where the earlier gate-frozen diagnostic closed but gate decoupling did not.
These are observed diagnostic cohorts, not new-input confirmation. Each call
has one 300-second total budget, including all cells and last-gate endpoints.
The registered admission rule requires at least five additional complete
history closures **and** a positive paired median score improvement.

The original M-B comparison used independent optimization trajectories:
20 alpha-only evaluations versus 20 joint alpha/guard evaluations, with the
latter switching from zero to a fixed positive multiplier initialization.
It did not retain the alpha-only trajectory's envelope. The table remains a
valid comparison of those implementations, but cannot isolate the benefit of
adding guards to an already optimized alpha baseline. A separately authorized
[baseline-preserving diagnostic](../../docs/guard_dual_refinement_20261029.md)
on the same eleven executed histories is now complete; it retains the full
prefix and includes a paid extra-alpha control. It does not amend the old table
or the old B4 arms. The original M-C test concerned first-layer cells encoded
as output-side guards, not the proposed last-layer dual-conditioned cells.

| Comparison | Paired histories | Closure gain / loss | Complete score pairs | Median paired score difference | Admission |
|---|---:|---:|---:|---:|---|
| M-B / zero-guard | 11 | 0 / 0 | 11 | −2.410982 | Fail |
| M-C two cells / no partition | 27 | 0 / 0 | 25 | +0.149568 | Fail |
| M-C four cells / no partition | 27 | 0 / 0 | 19 | +0.427032 | Fail |

All 103 registered scientific entries are retained and complete. Accounting
includes 111 physical attempts, including preserved tool errors and diagnostic
probes; no failed attempt is used as a comparator outcome. Paired variants use
the same engine within each object, with disclosed compatibility repairs across
execution continuations. Every M-B call returns UNKNOWN. M-C's three arms return
3/1/1 UNKNOWN and 24/26/26 TIMEOUT respectively, with zero complete history
closures. The audit reproduces identities, counts, coverage bookkeeping and
arithmetic on stored scores; it does not independently prove CROWN's bounds.

M-B scores are conditional property lower bounds. M-C scores are minima over
complete initial tables for the guarded maximum specifications, cells and
endpoints; they are not conditional property bounds. The two score types are
not compared numerically. M-C's positive medians concern the 25 and 19 observed
pairs only: the missing 2 and 8 score pairs remain missing, not zero. On the
same 19 objects with all three initial tables complete, median initial-bound
costs are 62.240, 123.406 and 207.825 seconds. Median per-object cost ratios are
1.941 and 3.574. All-request median costs on the full 27-object cohort are
approximately 294.26 seconds in every arm. Thus improved initial scores did
not translate into complete proofs or end-to-end speedups under this budget.
The observed costs suggest a tradeoff, not a proof of the unique bottleneck.

Both mechanisms fail the unchanged admission rule and are excluded from B4.
This stops these finite implementations without claiming that all guard-dual
methods or all margin partitions are ineffective. The authorized complete-
request comparison tests **M-A alone**, with the single-layer boundary and
double-radius cohorts preserved. Its completed primary table above fails the
registered practical-effect threshold; the completed double-radius cohort has
two gains and the single-layer cohort remains pending. They cannot validate a
successful three-component algorithm. The historical studies
below keep their original cohorts, guarantees and outcomes.

Evidence: [final result, source bindings and independent recount](../../docs/moe_aware_stage5a_results_20261026.json),
[mechanism analysis](../../docs/moe_aware_bab_20261026.md), and
[frozen B4 design](../../docs/moe_aware_b4_20261026.md).

### Revised guard diagnostic: local effect, insufficient admission

The revised design uses the same eleven model/input/property objects and their
previously executed histories. Three old catalogue entries had missing history
IDs; both old arms' stored histories agree and bind those entries explicitly.
Each new call independently recomputes its prefix ranges. After the original
20-evaluation alpha-only prefix, the implementation restores the earliest
evaluated alpha state maximizing the sum of property bounds. One arm performs
40 additional alpha updates; the other performs 40 joint alpha/guard updates
from zero guard multipliers, with one charged small gradient probe. Both retain
the per-property maximum over the entire baseline and continuation trajectory.
No old numerical bound is supplied as a free candidate.

All 22 calls complete within 300 seconds: 21 UNKNOWN and one numerical
POSITIVE, with no errors or timeouts. The eleven paired baseline vectors and
selected alpha-state hashes agree. Both continuations improve all 99 stored
baseline property bounds without losing any. The joint arm turns eleven
individual properties positive; extra alpha alone turns none positive.
Only one entire history closes: residual L2E8 seed 1, input 1312. Its minimum
bound changes from −5.532507 to −5.268918 with extra alpha and to +1.059180
with guard refinement. This establishes a local guard effect under the
numerical policy, not a complete MoE request or independently checked proof.

The median minimum-bound improvement over the original prefix is 57.960907,
but much of that improvement also occurs with extra alpha. Against the paid
extra-alpha arm, joint refinement is better on eight histories and worse on
three, with a median difference of 0.323164; individual properties split
79 better and 20 worse. These are raw score units across fixed models, not
normalized accuracy improvements. Baseline retention does not imply dominance
over a separately continued optimizer.

The registered rule still requires at least five additional history closures
and positive median improvement. **One of eleven fails the first condition.**
The resulting decision closes this M-B attempt and the conditional last-margin
M-C follow-up. No new B4 or B5 is started. The completed primary cascade table
and its remaining characterization queues are unaffected.

Total request cost is 1,483.346 seconds; measured control, preparation, batch
and independent record recount total 1,502.449 seconds, with about 2.79 MB of
evidence and reserved accounting. The alpha/guard request medians are
76.425/73.595 seconds on a shared machine; this diagnostic is not a speedup
comparison. One synthetic-control failure is retained: its test incorrectly
required positivity across the analytic Lagrangian gap described in the
method. Only the assertion was corrected, without optimizer or admission-rule
changes. Evidence: [full result and per-property traces](../../docs/guard_dual_refinement_result_20261029.json)
and [fixed design, controls and interpretation](../../docs/guard_dual_refinement_20261029.md).

## Historical evaluation scope

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

A subsequent saved-only source-applicability audit sharpens this contract.
All 100 materialized input boxes fail exact containment of the requested
real `2/255` boxes around their saved centers, including when epsilon is instead
interpreted as its exact binary64 value. Reconstructing the frozen sparse input
HZ formula on those boxes reveals inward endpoints on 98/100 inputs
(5,736/307,200 coordinates; maximum gap `2^-56`). All 23 primary gain requests
are affected. These are exact input-containment findings, not recovered
historical layer traces or network counterexamples. The table is consequently
a **frozen-policy acceptance comparison**, not independently established
certified coverage of the requested box, the materialized box, or the native
floating program. Subsequent network/guard/output containment also remains
unproved for these saved results. Tiny local gaps cannot be dismissed using an
old final positive margin without a complete propagated-error argument.
The [source-applicability ledger](../../docs/main_table_source_applicability_20260921.md)
binds all gains and leaves all historical counts and failures unchanged.

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

Historical acceptance counts below are **not source-complete strict
certificates**: all 23 primary gains have the audited input-containment gap.
The counts are unchanged; a corrected-source experiment could have different
route coverage and outcomes.

| Model | Adaptive policy SAFE / solved | Matched policy SAFE / solved | Legacy policy SAFE / solved |
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
with no losses. All 23 SAFE gains have multiple solver-policy legal pairs: two finish
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

## MetaMoE author-checkpoint comparison: an adverse external result

This separate repaired follow-up executes all 20 calls on the same ten already
observed inputs (five CIFAR-10, five MNIST), without splicing the stopped
14-normal-return / one-error / five-not-started batch. Both arms use the same
checkpoint, materialized input, normalized-space `2/255` box clipped to
`[-10,10]`, 19 global output margins, and 300-second total request budget.
This is class-separated raw-score top-1, not the main weighted top-2 task or
a pixel-space radius. The author path is **author backend + disclosed lAs
compatibility wrapper + strict route-invariance sufficient adapter**.

| Arm / positive evidence grade | N | Positive | Replayed UNSAFE | Complete UNKNOWN | Outer TIMEOUT | ERROR | Mean charged seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| ACT / HZ policy acceptance | 10 | 4 | 0 | 6 | 0 | 0 | 28.56 |
| Repaired author adapter / numerical sufficient filter | 10 | 9 | 0 | 0 | 1 | 0 | 36.54 |

Four positives are shared, none are ACT-only, and all five CIFAR-10 inputs are
author-only. Every ACT positive is route-stable. There is no new route-changing
or source-complete certificate, and no evidence of ACT competitive advantage
on this cohort. Different grades do not erase the practical coverage deficit;
they also do not justify calling the author positives strict certificates.

All-request mean cost favors ACT only because the author has one unresolved
300-second timeout. On nine normally returned pairs ACT costs 16.03 seconds
more on average; on the four common positives costs average 10.55 versus
7.17 seconds. Earlier UNKNOWN is not faster successful verification. The
repaired MNIST/7 request performs seven checked metadata restorations and
native insertions before timeout: compatibility is fixed, proof completion is
not. No sample, time, threshold or relaxation was changed to rescue results.

Saved logs show 209 ACT output obligations: 116 violation regions excluded
under the policy and 93 local timeouts. For four CIFAR requests, roughly
1.5-second per-property slices repeatedly pay worker/setup overhead. MNIST/7
also has unresolved route coverage and base feasibility. These are execution
stopping points, not proof that the relaxation is insufficient or that extra
time would prove safety. No new solving accompanies this analysis.

Evidence: [full result and saved-only diagnosis](../../docs/metamoe_las_followup_result_20260923_r1.md).
This finite, observed-cohort result favors the author sufficient path; it does
not establish general author-tool superiority. It remains separate from the
historical MetaMoE-style reimplementation and the ACT-fronted CROWN comparison.

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

A separately frozen, postselected follow-up targets the sole blocked property
of seed1/index4018. Independently checked sigmoid bounds narrow its gate from
[0.5,1] to approximately[0.51575,0.76666]. Reusing the original dual gives
-0.55171 instead of -0.60409; one separately registered new dual proposal gives
-0.50884. Rechecking all27 obligations still yields26 positive and one
nonpositive, not a complete certificate. The60.41s saved-source procedure
excludes earlier propagation/support generation and is not production timing.
Neither these lower bounds nor native success establishes an unavoidable LP
gap. The follow-up is closed without more cases or tuning, and does not alter
the original0/3 result. Evidence: `docs/checked_gate_candidate_v2_results.md`.

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

## Complete proofs versus checked source construction

The positive stored input98 proof and the newer source-checked construction
are **different evidence objects**. The former checks all nine output
obligations (eight residuals and one reused fact), with minimum
`199593373867685/1125899906842624`, approximately0.1772745. Direct rational
McCormick construction removes trust in floating F0 lowering. The portable
7.18MB bundle checks without a checkpoint, dataset, solver or historical
directory, but still assumes soundness of the supplied expert enclosure.
A separate extension checks its five route exclusions without establishing
the missing expert-source trace. This is a conditional positive proof of a
postselected, single-pair request, not a route-changing certificate.
Evidence: `docs/portable_conv_proof_v1_review.json` and
`docs/router_source_v1_review.json`.

The newer path instead checks containment of the represented input box,
compensated affine operations, every remaining expert layer, factor maps,
route coverage and all nine output-LP constructions for a declared real graph.
It generates **new matrices and fresh bound candidates**. The final finite
row-selection comparison retains zero positive obligations: prefix has nine
checked nonpositive bounds; property-directed selection has eight and one
missing candidate (property7). Eight common bounds improve7.4554–11.3268,
but neither arm proves the request. Their full stored-source costs are158.40
and185.73s, excluding historical capture and separate audits, not matched
production speed measurements. Evidence: `docs/full_source_v1_review.json`
and `docs/property_ranges_v1_review.json`.

Neither result can fill the other one's missing proof link. The old positive
duals cannot be attached to the new matrices, and the checked new source cannot
retroactively justify the old enclosure. The protocols also differ in source
representation and gate ranges; their outcomes are not a causal estimate of
the cost or precision effect of removing one trust assumption. They establish
portable conditional positivity and checked declared-source construction
separately, **not a complete source-checked positive network proof**.

The final bounded diagnosis reuses only the eight shared saved records. Exact
accounting shows changes in both product and last-ReLU terms, but the assignments
have not passed exact feasibility checks. Single-component local replacements
remain negative; joint replacements are not feasible repairs or global bounds.
No isolated new intervention follows. Input98 is therefore closed without
retrying property7, expanding rows, changing budgets or weakening acceptance.
See `docs/property_diagnosis_v1_results.md`. This research stop is not a
proof of an unavoidable LP gap or model unsafety. The full chronological
results, costs and unsuccessful controls are preserved verbatim in
[the source-proof history](../appendices/source_proof_history.md).

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

### Author-model execution context (not a verifier competition)

A frozen CIFAR-100 Robust Experts reproduction variant completed200epochs for
dense ResNet18 and E4/k2/layer4 ConvMoE, with final-epoch-only selection and
full10k clean/PGD20/APGD20 evaluation. The accuracies were37.27/18.99/18.74%
and22.18/12.88/11.70%, respectively: this single-run pair did not reproduce a
MoE advantage. Independent saved-state review checked all400epoch checkpoint
hashes, final states, trajectories and aggregate records, not new predictions
or attacks. The source `entropy` objective additionally includes mean per-input
entropy, unlike the paper's printed negative entropy of the batch mean; the
run is consequently labelled a paper-hyperparameter/source-entropy compatibility
variant. These empirical results are not added to our verifier SAFE tables.
Details and the unmodified negative result are in the
[landing and recipe review](../../docs/robust_experts_landing_20260922_r1/README.md).
