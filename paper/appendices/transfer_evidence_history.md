# Historical transfer and evidence development

Preserved verbatim from the previous evaluation tail on2026-09-20. Later
paragraphs supersede earlier pending/smoke statements. These are separate
studies, not one pooled cohort. The current synthesis is in section08.

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

The subsequent generalized evidence-mode study completed60 requests on20 new
convolutional clean-correct inputs, selected by ascending index after excluding
878 used indices from5,696 source records. The same checkpoint,2/255 and300s
were retained. Matched V2 returned10 replayed UNSAFE and10 TIMEOUT; general
evidence returned20 TIMEOUT; ACT-fronted plain CROWN returned7 replayed UNSAFE
and13 UNKNOWN. No arm produced a complete positive result of its respective
evidence grade. Mean observed request costs were220.68,283.09 and4.08s,
respectively, including incomplete/capped requests. The evidence path seeks
positive proofs rather than performing equivalent counterexample search.
Common facts agreed on all20 pairs;15 inputs had one legal pair and5 had
multiple pairs. The17 UNSAFE run records concern12 distinct inputs.

The frozen complete-request negative result is not overridden by partial
checker output. Seven saved local prechecks cover all necessary obligations
but include nonpositive lower bounds; six report missing evidence; seven
requests have no completed local precheck. Among the13 saved prechecks,
135 obligations divide into3 positive,70 nonpositive and62 missing. These
stored observations were not promoted to independently completed request
certificates. Sixteen outer terminations occur during precheck(3), packaging(9)
or isolated checking(4), while four internal exits exhaust the proposal
reserve. Stop locations alone do not establish unique causes. In particular,
neither cheaper packaging nor additional solver time is shown to produce
positive requests. The earlier input98 success remains a postselected control,
not evidence of transferable coverage. The automatic final audit and separate
archival review preserve all60 terminals and evidence-grade distinctions.
See `docs/general_evidence_execution_v1_results.md` and its bound JSON index.

