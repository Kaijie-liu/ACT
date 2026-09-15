# Generic weighted top-2 evidence, version 1

This is a separately frozen optional evidence mode, not a change to production
optimal-status acceptance. The request names E experts, C outputs, the actual
represented box, model identity and a nonempty list of linear properties.
There is no input98, fixed pair, E4 or C10 assumption in the evidence checker
or generator. The separately preserved CROWN comparator remains C10 only.
Scope: output-layer selected-softmax top-2, ANY_LEGAL_TOPK, eval, CPU/float64.
No multi-layer/token routes or top-k>2 claim.

## Checked conclusion and trusted boundary

Let X be the represented input box and X_S its tie-legal pair domain. Assume
the stored shared expert HZ encloses the ordered expert outputs on X_S; the
membership HZ intervals enclose each expert on X_i; guards and infeasible-pair
exclusions are correct. The independently checked route partition lists every
unordered E-choose-2 pair exactly once. Unresolved routes cannot establish a
positive complete result. Exclusion truth is a trusted premise, not proved by
listing it or hashing it.

For each requested q,c and every feasible S={a,b}, either two membership
facts prove qE_a+c and qE_b+c positive (X_S subset X_a and X_b), or a checked LP
proves u+w positive, where u=qE_b+c and d=q(E_a-E_b) share the same HZ factors.
The independent checker reconstructs these rational projections, checked gate
and difference ranges, four McCormick inequalities, finite variable bounds,
continuous binary relaxation and exact rational dual lower bound. Convex
weights and complete pair/property coverage imply the registered output
properties for every tie-legal execution, conditional on the premises above.

| Independently checked | Still trusted |
| --- | --- |
| Request/source hashes, E/C widths, property and frame binding | Network and materialized input to stored HZ/intervals; ordered expert binding |
| Pair partition and no unresolved omission | Correctness of guard lowering and infeasibility exclusions |
| Scoped reuse arithmetic; exact projections and HZ→LP export | Upstream floating HZ propagation and stored coefficients |
| Checked order/difference ranges, rational McCormick construction and LP dual certificate | Runtime/interpreter and independently pinned checker/statement identities |
| Every necessary output obligation and aggregation | Not original deployed floating-point program equivalence |

Thus CHECKED_CONDITIONAL is neither HZ_POLICY_ACCEPTED nor
CROWN_NUMERICAL_FILTER, and none is automatically a deployed-float proof.
There is no remaining trusted floating F0 construction step. Nonpositive
checked bounds are UNKNOWN_NONPOSITIVE, not UNSAFE. Missing evidence is a
different state. The optional path currently seeks positive proofs only; it
does not run an additional attack. Baseline UNSAFE still needs full-model
replay in archival audit. Do not report incomparable solved counts as proof
superiority.

## Execution and controls

`moe_evidence.generate.capture` reads ordinary model/tensors/request/config and
intercepts the matched V2 prelude before weighted F0. It computes its own
routes and membership intervals, handles all feasible pairs and creates a
complete obligation roster before propagating residual pairs. Common source
intervals, not old classification verdicts, feed the requested properties.
`propose_all` uses deterministic pair/property order and the existing LP
proposal policy. The checker does not import/call the generator or a solver.

`moe_evidence.execution.run_request` owns one 300s monotonic clock, with the
same 2s outer terminal reserve as the previous evidence smoke. V2 retains its
5s inner reserve. LP calls have cap60s and reserve80s for checking. All loading,
capture, export, proposal, rational construction, precheck, packing, isolated
`python -I -S` check, hashing and terminal submission are charged. A proposal
reserve exhaustion may leave explicit missing evidence for checking; it must
not discard pending obligations. Outer watchdogs kill only owned process
groups; late files cannot promote a failed terminal. Separate archival
checking cannot rescue a deadline failure.

The bundle contains deduplicated sources, all necessary obligations and the
small isolated checking code; hashes/statement must be pinned independently.
No model, dataset, prior run, solver, network access or floating F0 builder is
needed by its checker. Transport integrity alone is not the mathematical test.

Control suite: `python -m unittest scripts.test_general_evidence -v`.
It covers E/C changes, explicit rational/nonzero-offset properties, all-tied
multi-pair controls, partial reuse, real ACT capture/proposal, cross-request
and property/source mutations, missing/duplicate coverage, exact gate ties,
nonpositive vs unavailable evidence, portable outside-checkout checking and
actual owned-child timeout. An analytic end-to-end matched/evidence test also
checks terminal packages and equal independently generated common facts.
No real convolutional verification request is used for these controls.

## New experiment (freeze only in this stage)

Protocol: `general_evidence_v1_protocol.json`. Twenty new ascending-index
clean-correct CIFAR10 inputs on the already selected conv epoch89 checkpoint;
exclude the frozen union of prior verification/development indices, including
failed/incomplete runs and the previous convolutional roster. No route count,
attack, bound or proof success is used for selection. Prior all-test telemetry
is disclosed: these are unused evidence-mode endpoints, not unseen images.

Three arms: unchanged matched V2, generic evidence, original plain CROWN;
20 input blocks/60 requests, fixed2/255 and300s, cyclic arm order, no pooling
with input98 or past cohorts. No tuning25%, support, thresholds or checkpoint.
All statuses, gained/lost complete positives, proof grades, costs, timeouts,
and missingness remain visible even for zero gains. New execution directory
required; no retry/resume/replacement. Full cohort launch is a later explicit
step after published freeze and controls; this stage performs clean-only
selection and separate-process reconstruction, not the60 verification calls.

Portable proof API and per-request terminal audit are implemented here.
Before full launch, the cohort-level lock/resource/roster supervisor and final
three-arm aggregation must be bound to the frozen selection and tested; do not
substitute the old input98-only launcher or launch individual selected inputs
out of order. This preparation boundary is explicit, not an executed experiment.
