# Hard-core gate decoupling — stage4

This is the fifth revision's diagnostic, not a new confirmation cohort or a
claim that changing engines is an algorithmic contribution. The scientific
question is whether replacing gate equalities by narrow independent ranges
allows the strong engine to close histories it previously did not close.
The opposite outcome must remain visible. Real stage4 execution is now running;
the pre-execution statements below remain historical records.

## Completed single-layer B2 diagnostics

All24 frozen B2 objects (19 distinct inputs) have both local calls completed:
4 both closed,7 V3-only closed,13 neither closed,0 V2-only. None is omitted.

| Variant | Local positive / diagnostic closed | UNKNOWN | TIMEOUT | Sum of terminal costs | Median local cost |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 independent gate + final endpoints | 4 | 4 | 16 | 6165.293s | 294.203s |
| V3 frozen-gate counterfactual | 11 | 9 | 4 | 4394.150s | 293.281s |

The48 receipt costs sum to10559.443s. The contiguous interval from the first
B2 start to the first1b start is10567.941s, including inter-call bookkeeping;
neither includes other cohorts or later offline review. V3 changes the function,
so its lower cost is not a same-guarantee speedup. All24 domain pairs match.
The [record](gate_decoupling_stage4_records_20261025.json),
`completed_b2_analysis`, binds48 ledgers and their checked artifact hashes,
per-object outcomes, endpoint statuses and source provenance. This is a record
review, not an independent numerical proof of the bounds.

Three V2 positives concern old returned-unproved histories; seed1/4212 concerns
an old unvisited pair. The historical guard-graph/engine confounds below remain,
so these are not four isolated gate-only wins over matched failed queries.
One of the7 V3-only objects has a known original same-history violation, as
documented below. Unknown bounds and absent endpoint returns do not establish
relaxation optimality or a unique ReLU cause. Range reconstruction is small
here (48 calls: median0.125s,maximum0.751s). Yet gate width0.00312 does not close,
while positives have widths0.02813 through0.37852: no narrow-gate success rule
is inferred from these observations. The complete stage4 queue, full193-object
denominator and primary decision remain pending;1b continues without new queries.

## Counterfactual closure with a known original violation

B2 seed1/input4356, selected pair `(2,3)`, now gives V3
`DIAGNOSTIC_CLOSED` in8.224s, while V2 ends `TIMEOUT` in294.168s.
V3 verifies all9 properties at initial bounds. V2's low endpoint verifies;
its high endpoint returns unknown before the outer timeout. Both variants bind
the same original request and rebuilt gate interval `[0.50720565,0.73956632]`.
The saved original full-model replay already has a valid same-pair negative
margin `-0.0575768502`. Thus, at the registered numerical-policy grade, a
V3-only closure can coexist with an original violation; it does not establish
that independent-gate relaxation lost a proof of an actually safe history.
The record's `known_b2_witness_history_relations` entry binds the new receipts
and old replay. This review loads no model or solver and does not reprove the
numerical bounds or witness-gate containment. Keep the known-witness label,
full denominator, order and pending final decision unchanged.

## Single-layer local closure: historical comparison limits

B2 seed1/input4150, pair `(6,7)`, is the first single-layer V2 positive.
Its old coupled query returned unknown after249.905s; V2 closes both endpoints
and all9 properties in77.162s total (endpoint solver costs62.146s/6.503s,
both BaB). V3 closes in17.809s. The two new calls bind the same gate interval
`[0.3819225635,0.7223526634]`, reconstructed within their budgets. Its width
about0.34043 does not support treating every local gain as a nearly fixed gate.
The [record](gate_decoupling_stage4_records_20261025.json) field
`singlelayer_first_local_closure_and_confound_review` checks the same original
request, checkpoint identity,9 properties,31 original-source hashes and both
new receipts. No additional model execution or solver query was performed.

There are two additional confounds in comparison with **historical B2**.
The old adapter first forms one shared maximum of all guard atoms and then
combines it with each property. Stage4 folds the atoms into each property.
The real predicates are equivalent, but the ReLU graphs differ and may affect
bound propagation and cost. Also, old B2 used compatibility snapshot r1;
Stage4 uses r3 with the later sparse-spec multiplication and branching-score
fixes. Matching CONFIG and the unchanged `verify_graph` function do not mean
identical engine sources. These differences prevent attributing this historical
contrast solely to gate decoupling or claiming an equal-cost speedup.

The current V2/V3 pair uses the same frozen r3 engine and guard construction.
Its accepted local outcomes remain unchanged. The old27 other pair positives
must not be combined with this new local closure into an online full-request
success. Numerical source limitations and the full193-object decision remain;
no interim gate, threshold, graph modification or extra coupled rerun follows.

## Live execution snapshot

Detached PID425771 runs immutable9bf3eacd6b462001e68c1c0257574601bc4410c5 in
`executions/gate_decoupling_stage4_20261025_r1`; raw outputs are in the matching
`results/gate_decoupling_stage4_20261025_r1`. The
[startup record](gate_decoupling_stage4_records_20261025.json) binds the process,
source/catalog/control identities, resource admission and original21h deadline.
At20:38:48 AEDT,4 local calls completed and the fifth had started: the first two
L2E16 seed0 histories (inputs1281,1288) each gave V2 POSITIVE and V3
DIAGNOSTIC_CLOSED, with matching gate-domain identities. Times were13.352/9.812s
and11.338/8.415s for V2/V3, including each call's range reconstruction.
The raw-receipt prefix audit passes with decision `PENDING_REGISTERED_QUEUE`.
These are two local-history numerical closures, not complete-request gains,
strict certificates or a stage4 decision. Live ledgers supersede this snapshot.

At20:57:24 AEDT the reviewed prefix has8 completed calls and4 paired histories,
with the ninth call running. The same record's `latest_research_observation`
binds each terminal/ledger and saved endpoint; all four raw gate-domain pairs
match. Inputs1281/1288 were saved-cut pending histories in the original run,
not established individual baseline failures. Input1431 times out in both
V2 and V3 solving after less than1s per-call range reconstruction. Input1476
closes V3 in173.668s total; V2's low endpoint verifies in5.284s solver time,
but its high endpoint exhausts the remaining shared budget. This distinguishes
observed terminal patterns, not their unique causes: no checked feasible upper
bound or relaxation-optimality evidence establishes that V2 is intrinsically
too loose. Total completed-call cost is1099.282738s, excluding the running call;
the separate0.081315s offline record review upgrades no online conclusion.
The registered queue/denominator/decision rules remain unchanged and pending.

At21:33:32 AEDT the reviewed prefix has18 completed calls/9 pairs:3 both
closed,3 V3-only closed and3 both timed out. L2E16 seed1/input1557 is the first
V2 positive whose selected history was recorded in flight in the old coupled
run, rather than merely pending. The same history `(4,15)/(0,13)` entered the
old solver with240.335s until its worker deadline, without a returned closure.
V2 now verifies both endpoints in18.020s total, V3 closes in11.416s, and the
gate-domain identities match. Source/terminal hashes are in the record's
`first_saved_inflight_local_closure`. This is a local numerical signal: the old
remaining budget is not a matched local rerun or proof of equal solver effort,
and neither this history nor these times establish a complete-request gain.
The complete fixed queue must still determine the stage decision.

The27-call reviewed prefix adds the first closure on a **returned-unproved**
old query: L2E8 seed0/input1366, history `(2,7)/(1,7)`. The old run pruned783
of784 histories; its one returned query remained `unknown` after288.656s in
nonlinear BaB. V2 now closes both endpoints/all9 properties in22.131s total;
the endpoint solver calls take10.723s and1.617s, both `incomplete_only`.
V3 closes in8.488s total. Registered engine configurations match, and both new
variants bind the same saved original-prefix domain: inner gate
`[0.6670377463,0.6865419324]`, final gate `[0.3574424584,0.9850951270]`.
The record's `first_saved_returned_unproved_local_closure` binds the raw sources
and receipts. This is evidence that a sound-in-principle, weaker gate relation
can be computationally easier for this bound engine; numerical execution still
inherits its CROWN/sigmoid policy. It does not isolate decoupling from final
endpoint treatment, establish a new abstract domain, or prove a runtime ratio:
the new local diagnostic reuses saved ranges and is not a matched full request.
Old pruning and new closure must not be spliced into an online success.

The same input1366 now has returned-unproved history closures on all three
L2E8 seeds. This is **one input across three trained models**, not three
independent inputs. The record's `returned_unproved_input1366_all_seeds` binds
all old queries, new endpoint receipts and domains:

| Seed | Old coupled query, unknown (s) | New V2 local call, positive (s) | New V3 diagnostic call (s) | V2 endpoint stages |
| --- | ---: | ---: | ---: | --- |
| 0 | 288.656 | 22.131 | 8.488 | Both initial bound propagation |
| 1 | 287.489 | 79.359 | 13.773 | Both initial bound propagation |
| 2 | 287.778 | 266.012 | 9.116 | Both nonlinear BaB |

Seed2's endpoint solver costs are157.362s and95.201s; its near-budget closure
limits any suggestion of uniformly cheap verification. Old costs cover one
solver query; new costs cover a local diagnostic using saved original-prefix
ranges. Their ratio is not an end-to-end speedup. Each old numerical census
pruned783of784 histories and left just this one: these cases do **not** establish
route-changing request gains or independently checked route stability.
They support studying the computational effect of the gate treatment, while
the full diagnostic and subsequent complete-request gates remain necessary.
The64-call prefix audit passes with11 joint closures,8 V3-only closures,
4 paired timeouts and9 paired missing sources; the stage decision is pending.

All11 fixed L2E16 hard-core objects have now finished both variants:3 both
close,3 close only V3,4 time out in both,1 lacks the first-history identity.
All22 calls cost3779.446s, including missing-source terminals; this excludes
other calls and final batch accounting. This subgroup is descriptive, not a
replacement for the193-object registered denominator. The27-call prefix audit
passes; the stage decision remains `PENDING_REGISTERED_QUEUE`.

Both two-layer subgroups are now complete (calls000--071). The record's
`completed_two_layer_analysis` binds the source files and the independent
terminal/domain/endpoint recount. This is an exploratory completed-subgroup
analysis, not an interim replacement for the full193-object decision rule:

| Family | Both close | V3 only | Both time out | Missing source | Total objects |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2E16 | 3 | 3 | 4 | 1 | 11 |
| L2E8 | 9 | 7 | 1 | 8 | 25 |

Of12 V2 positives,11 close both endpoints during initial bound propagation;
one needs nonlinear BaB at both endpoints. Initial-bound completion is not
uniformly cheap: one such call takes199.115s including all local costs.
The old selected-history identities for these positives are6 saved-cut pending,
3 in-flight and3 returned-unproved; pending histories are not demonstrated
individual baseline failures. The3 returned-unproved cases share input1366.

On all27 objects with comparable domains, V2 gate-range acquisition takes at
most1.274s. For the15 V2 timeouts,7 save neither endpoint result,5 save a verified
low endpoint but no high-endpoint result, and3 save an unknown low endpoint but
no high-endpoint result. Thus neither a missing high result nor a partially
positive pair establishes that the relaxation is intrinsically unprovable.
The first72 terminal costs total5147.010s for V2 and2614.554s for V3, including
missing-source terminals, excluding later calls and final batch accounting.
V3 has a different, counterfactual guarantee; this cost difference is not an
equal-task speedup.

Inner gate-width medians are0.03779 for joint closures,0.11386 for V3-only
closures, and0.00136 for paired timeouts. The paired-timeout group even includes
an inner range of width about1.11e-16. These descriptive, outcome-conditioned
statistics do not select a threshold or prove a causal gate-width effect;
they show why narrow inner gates alone cannot promise easy verification.
The80-call raw-prefix audit passes; all stage-level decisions remain pending.

The full1a diagnostic subgroup is complete at110 local calls /55 objects. The
record's `completed_1a_analysis` binds all110 ledgers, source kinds, endpoint
outcomes, costs and the frozen prefix audit. B2 calls continue in the original
queue; this subgroup does not replace the registered193-object population.

| Family | Both close | V3 only | Both time out | Missing source | Total objects |
| --- | ---: | ---: | ---: | ---: | ---: |
| L2E16 | 3 | 3 | 4 | 1 | 11 |
| L2E8 | 9 | 7 | 1 | 8 | 25 |
| L3E8 | 12 | 5 | 1 | 1 | 19 |
| Total | 24 | 15 | 6 | 10 | 55 |

The24 V2 local numerical positives break down by old history provenance:
11 were saved-cut pending,10 were recorded in flight without a returned
conclusion, and3 had returned unproved. Those last3 are the already documented
input1366 across three seeds. Thus24 must not be presented as24 individually
failed old solver queries that the new method beat. A saved-cut pending label
does not rule out unrecorded later work. A new local numerical positive on such
a history is diagnostic evidence, but not a matched win.

Both endpoints close during initial bound propagation on22of24 positives.
L2E8 seed2/input1366 needs nonlinear BaB at both endpoints; L3E8
seed1/input1288 needs it at the high endpoint. On45 available domains, gate-range
acquisition has median0.967s and maximum2.931s. The21 V2 timeouts have12 missing
endpoint pairs,6 verified low endpoints with the high result missing, and3
unknown low endpoints with the high result missing. This locates observed
execution cost and partial evidence; it does not certify a nonpositive optimum
or isolate an intrinsic relaxation gap.

All110 terminal costs sum11915.135s:8636.967s for V2 and3278.169s for V3, including
missing-source terminals. These exclude subsequent calls and final batch
accounting. V3 is a different counterfactual task, so the cost contrast is not
an equal-guarantee speedup. The separate1.342s frozen-source/raw-receipt audit
passes with no paired domain mismatch; it is not a reproof of numerical bounds.
The primary decision remains `PENDING_REGISTERED_QUEUE`. No early mechanism
selection, threshold change, or complete-request success is inferred.

Before those B2 objects execute, the existing three witness labels have also
been checked against the historical full-model replay receipts. The record's
`known_b2_witness_history_relations` binds the launch/outcome/receipt files and
witness hashes; all three request items exactly match the current catalog.
Seed0/input4341 has a saved negative replay on pair `(4,6)`, and seed1/input4356
on `(2,3)`: both match their selected diagnostic histories. Seed0/input4237 has
its negative replay on `(6,7)`, while the selected history is `(5,6)`.
Thus failure to prove the first two original histories cannot simply be
attributed to verifier difficulty; a local positive on the third history would
not conflict with an unsafe different route. These are historical numerical
full-model replays, not newly run or independently exact witness checks.
The frozen cohort, order, denominator and decision rules remain unchanged.

V3 removes variable gate products but retains router and guard computation.
For this L2E16 family the graph adds56 ReLU max steps for guard atoms, plus8
selected-expert and2 router ReLUs. The saved input1476 V3 graph lists66 available
ReLU split nodes, consistent with that construction. These are graph nodes,
not visited branches, scalar activation counts or a timing attribution. V3
timeout therefore does not isolate expert-network difficulty from guard
encoding, bound optimization and search. No graph or decision rule is changed.

## Final execution freeze after1b

[R1 real freeze](gate_decoupling_stage4_freeze_20261025_r1.json), SHA256
`58f30f30d22ae334631514cec4a7fdde2567e8b9bb37df61ca96504411c448d5`, binds the
[complete catalog](gate_decoupling_sources_20261025_r2.json), unchanged engine,
20 source files and the accepted [control record](gate_decoupling_control_records_20261025.json).
R4 passed19 methods/3 prescribed API variants; cumulative R3+R4 cost11.004842s
and191,232bytes includes the preserved stdout-cap failure. This establishes
the diagnostic integration, not a trained-model improvement.

The full population is193 objects:1a55, B2 24 and1b114. The fixed386-call table
retains adjacent variants and alternating order. The250-call cap covers at most
125 paired objects (55+24+46); at least136 calls on68 objects remain unexecuted.
The same21h/300s/128MiB/8GiB policy applies; no new source selection or quota.
There are44 missing first-history identities (10 in1a,34 in1b), including29
among the first125 paired objects. Only4 complete range sources exist; missing
ranges are reconstructed within each call when the history identity is known.

This provenance/cap limitation is recorded **before diagnostic outcomes**:
at most96 pairs in the executable prefix have identified histories. With the
registered full193-object primary denominator, rule(a)'s timeout lower bound
above50% and rule(c)'s observed majority cannot be reached by this prefix.
Rule(b) or gray remains possible. Do not shrink the denominator, turn missing
records into timeouts, or infer absence of ReLU difficulty from an unreachable
stop statistic. Single-/multilayer strata and all missing/unexecuted counts
remain explicit. The evidence can still test local closure and inform the
registered report; it cannot supply complete-population attribution.

## Same-history graph and scope

For the canonical selected pair `(a_l,b_l)` at layer `l`, write
`m_l=r_a(h_l)-r_b(h_l)`, `g_l=sigmoid(m_l)` and
`h_(l+1)=E_b(h_l)+g_l(E_a(h_l)-E_b(h_l))`; the residual family also adds `h_l`.
Both static compilers retain the same routers, experts, original input, guards,
properties and affine final head. All classes and property constants remain.
The implementation is opt-in in
[gate_decoupling.py](../act/back_end/moe/gate_decoupling.py); it is not wired into
the live1b queue or an automatic production fallback.

Original-prefix CROWN records give the two ordered upper bounds
`U_ab >= r_a-r_b` and `U_ba >= r_b-r_a`; therefore the canonical first expert's
margin range is `[-U_ba,U_ab]`. Source identity must bind the original model,
materialized box, properties and source version, and the record must match the
exact history prefix. Missing, nonfinite or inverted needed directions do not
become a fabricated range. The helper owns immutable tuples, so changes to the
caller record cannot mutate an already constructed gate domain.

V2 replaces each gate equality by an independent interval variable. Every
original history assignment embeds by setting those variables to its original
gates. Later gate ranges need to cover original executions, not every artificial
state introduced by earlier decoupling. Guards are evaluated at each state in
the new graph; an original legal assignment still satisfies them. This proves
mathematical outer containment assuming valid original ranges and lowering.
The implementation uses floating sigmoid plus `nextafter` padding and numerical
CROWN bounds. This **does not prove** a correctly rounded transcendental envelope
or source-complete containment of the original network; accepted results stay
at the registered numerical-policy level.

For fixed original input and all earlier independent gates, the final output
property is affine in the last gate, including the registered residual addition
and affine head. The final-layer guard depends on its input, not that last gate.
Thus two endpoint implications `guard => qF+c>0` imply the same property for the
whole last-gate interval. Each property is compiled as
`max(qF+c, all outsider-minus-selected atoms)>0`. Ties give zero guard atoms and
remain obligations. The two endpoint graphs retain every property and share one
eventual V2 call budget; a missing/nonpositive endpoint cannot establish closure.

Earlier gates cannot in general be endpointized: a later ReLU can implement
`abs(g-1/2)-1/4`, positive at both endpoints but negative at the midpoint.
Consequently multilayer V2 retains its inner products and the nonlinear engine;
only a single-layer V2 endpoint graph is purely affine/ReLU. A nonlinear final
suffix is rejected by this implementation, not silently linearized.

V3 fixes every gate to its interval midpoint. It is an affine/ReLU counterfactual,
not a subgraph or outer envelope of the original function. Its receipt says
`DIAGNOSTIC_CLOSED`, never original-history closed. A relaxed/frozen violation
also does not establish original UNSAFE without full-model replay. Fixing gates
can change downstream guard reachability, including making a history vacuous;
V3 closure alone is not a unique diagnosis of easy experts. Likewise V2 failure
does not by itself distinguish relaxation weakness from search/guard/budget.

## Controls prepared; no numerical execution yet

Update after1b completion: R3 ran once and failed after4.413010s because native
model dumps exceeded the existing16KiB supervisor console cap. No final test or
API receipt was accepted. The original failure and all files remain at
`results/gate_decoupling_control_20261025_r3`; cleanup left no live owned child.
[R4](gate_decoupling_control_20261025_r4.json) repairs only the control entry's
non-evidence stdout handling to match the real worker, and carries R3 time and
all bytes inside the same300s/128MiB total. The19 source/semantic/ledger cases,
three prescribed API variants, numerical engine and acceptance rules are
unchanged. Three new variant calls are prescribed; R3's interrupted API work is
retained, not erased (at most three old plus three repair attempts). R4 has not
run at this registration. This is a tooling repair, not a scientific rerun or
mechanism result. The paragraphs below record the original pre-execution state.

The [R3 control freeze](gate_decoupling_control_20261025_r3.json) binds19 source
files,19 semantic/receipt/ledger test methods and exactly3 external API calls.
It includes the local worker's shared endpoint deadline, partial interruption,
artifact budget/reception and the stage queue's fixed-order/cap/denominator
checks. R1/R2 are preserved unexecuted, superseded before any control run.
Static parsing and source-hash checks pass; none of the19 tests
or3 API calls has run. This is a prepared prototype, not a passed integration
stage. R3 SHA256 is
`87d0246b19689cf7e71fc07854845c0fc0bed4594eccb3c958ef87a3bba0ade4`.

The control batch uses the existing owned-session supervisor, existing isolated
GenBaB CPU environment, unchanged engine configuration and one shared300s/128MiB
budget with8GiB per-process AS. No checkpoint, training data, GPU or installation.
There is no new supervisor. The
[fixed controls](../tests/test_gate_decoupling.py) cover direction/ties/range source,
missing/inverted/nonfinite bounds, alias isolation, all properties and both
endpoints, late reception, L1/L2/L3 graphs, residual/affine suffixes,10 classes,
property constants, midpoint-positive/original-negative behavior, and the inner
endpoint counterexample. Three API queries use one analytic-positive two-layer
fixture (V2_LO,V2_HI,V3) under the same control deadline. They test API semantics,
not speed, trained-model success, or the registered stage4 statistical gate.

Unexpected failure stops the control batch and retains its files. Only tool
defects may be repaired under the existing user policy; no changed acceptance
threshold or reset of the cumulative budget. Native controls wait until1b ends,
so this diagnostic preparation does not add a competing numerical process.
Static parsing/review is not recorded as native-control passage.

The [local worker](../scripts/gate_decoupling_worker.py) uses the existing
owned-session supervisor and numerical API. Its parent charges a single300s
call, with the existing6s worker/cleanup/reception reserve; V2_LO and V2_HI do
not receive separate budgets. Loading, original-prefix reconstruction,
construction, solver and original-model replay have append-only ENTER/RETURN
records. An unfinished range query cannot be counted as a V3 expert-solver
timeout. Ordinary UNKNOWN still permits the second endpoint within the same
remaining deadline; interrupted calls retain partial records without closure.
The parent rechecks the catalog, property roster, gate domain, endpoint files
and local aggregate, without loading the network. This checks identity/coverage,
not CROWN's numerical arithmetic. A frozen-midpoint positive remains diagnostic.
Candidates from extended V2 inputs are projected to the original input and
replayed on the original dynamic model; independent gate values never stand in
for an input witness. Final real catalog/call-table freeze, worker controls and
batch execution remain pending. The live1b immutable copy is unchanged.

The [stage driver](../scripts/gate_decoupling_stage4.py) calls that worker, using
the existing supervisor. It freezes only after completed/audited1b is included
and the synthetic integration batch passes. Inventory order is retained, with
adjacent V2/V3 calls and alternating within-object order; no result-dependent
reordering or extra endpoint budgets. The full catalog is kept even if250 calls,
21h or128MiB leave a suffix unexecuted. Starts without a terminal count as
attempts and remain distinct from calls never started. Tool errors preserve
their prefix for the authorized repair process.

Primary decision uses all fixed hard-core model-input objects; single-layer and
multilayer summaries use the same rules and their own full catalog denominators.
The user's ordered rules remain: V3 TIMEOUT Wilson95 lower>.5 stops; joint V2/V3
closure Wilson95 lower>.3 enters stage5; V3 closure with observed V2 UNKNOWN or
TIMEOUT in more than half the catalog calls for a mathematical contract and user
review; otherwise gray. ERROR, source-missing and unexecuted entries do not
become observed V2 failures. Prefix summaries make no decision. Total V3 TIMEOUT
keeps the registered rule while phase strata distinguish preparation from
solver timeouts; that decision is not a unique causal explanation. Missing
observations stay visible in the denominator/status table. These are descriptive
intervals on a selected set with repeated inputs/fixed models, not independent
population evidence. Costs include final audit/accounting when available; a
stopped prefix explicitly lacks a final cost receipt. No stage4 numerical result
has been inspected to choose these interpretations or alter the thresholds.

## Real diagnostic preparation

The [read-only source inventory](gate_decoupling_sources_20261025_r1.json) now
binds the79 already defined1a/B2 objects. It is **not an execution freeze** and
does not include the still-running1b cohort. Its generator is
[gate_decoupling_catalog.py](../scripts/gate_decoupling_catalog.py), using only
the standard library; checkpoint/tensor contents are not loaded. A separate
read-only cross-check passes619 distinct raw-file hashes, the55+24 denominator,
canonical product ordinals and all24 B2 pair/property rosters. Catalog SHA256:
`ae03542cc691ece2da75629431b1af7ffe066b75f472d17968a35f7298e147e1`.

| Cohort | Saved returned-unproved history | Saved first in-flight history | Saved pending/unvisited only | Earlier identity missing |
| --- | ---: | ---: | ---: | ---: |
| 1a,55 objects | 3 | 29 | 13 | 10 |
| B2,24 objects | 21 | 2 started-without-return | 1 | 0 |

The two B2 interrupted starts are retained ERRORs (seed0/4237 pair5,6 and
seed2/4374 pair0,3); seed1/4212 pair2,3 was unvisited after a verified prefix.
These are not24 mathematically unsuccessful single-history solves. Three B2
objects also have previously saved witness records, kept in the inventory;
their selected history need not be the witness's history. No new replay was
performed here and no local positive can become whole-request robustness.
Only3/79 have reusable complete prefix-range records;76 need charged range
reconstruction if a history can be identified. This distinguishes preparation
cost/missing provenance from expert or gate difficulty before any V2/V3 result.

The hard-core catalog remains1a55 plus B2 24 plus the pending1b nonpositive set;
known counterexamples, tool errors and missing sources retain separate labels.
The existing1a inventory has3 complete first-history/range records,42 saved-cut
first-pending ordinals and10 missing earlier-unproved identities. Do not replace
those10 by the last active history or pretend saved-cut counters recover later
unrecorded work. Necessary reconstruction must use the registered diagnostic
budget; an unresolved source stays explicit, not an easier substituted history.

The real call table is frozen after1b completes, before any V2/V3 outcome. It
must expose every object and any unexecuted suffix under250 local calls/21h,
retain the user's Wilson decision rules and single-/multi-layer grouping, and
charge loading, bound reconstruction, graph construction, all endpoint queries,
reception and cleanup to each300s call. Prefix/source checks and local closure
cannot retroactively complete an old request. The implementation and its tests
are not a real execution freeze or authority to exceed the existing call cap.
