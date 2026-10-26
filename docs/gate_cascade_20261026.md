# Gate-decoupling cascade: mathematical and experimental contract

User-approved sixth revision; original Stage4 remains GRAY. This is a proposed
full-request mechanism, not evidence of a complete-request improvement. All
reported positive solver results retain the numerical-policy grade. HZ/source
checking remains a separate G3 obligation.

The seventh revision supersedes the experimental queue sizes at the end of this
sixth-revision document: B4 now has1320calls/110h in four nontransferable parts.
The mathematics and frozen120s cap below remain current. M-B/M-C admission is
decided separately by their fixed Stage5a diagnostic, not by these controls.

## Containment and last-gate endpoints

Fix one canonical, tie-legal history S and the original input box X. At layer l,
write h_l = A_l(h_(l-1), lambda_l), where

    A_l(h, lambda) = rho_l h + (1-lambda) E_b,l(h) + lambda E_a,l(h),
    lambda_l = sigmoid(r_a,l(h)-r_b,l(h)).

rho=0 for ordinary layers and rho=1 for the registered residual layers. Keep the
same stem, experts, routers, head, input and all route guards. For each original
static prefix, a whole-box CROWN margin interval induces I_l=[lo_l,hi_l].
Replacing each gate equality by an independent lambda_l in I_l contains every
original legal execution, by induction using its original states and gates.
I_l need not contain gates evaluated on every *relaxed* earlier state: only the
original-graph embedding is used. Missing bounds never imply a zero-width gate.
The real-number containment theorem assumes valid enclosures. Floating CROWN
and sigmoid-plus-nextafter currently provide the existing numerical policy,
not an independently proved enclosure of the source network.

The last-layer output followed by the registered affine suffix is affine in the
last gate. Last-layer legality depends on its input, not its gate. For any fixed
input/inner relaxed gates that satisfy all guards, positivity of every output
property at both last-gate endpoints proves it throughout I_L. Each endpoint
must cover every property, with identical source/domain and strict tie semantics.
When guards are encoded as max(property, outside-minus-selected atoms), apply
the endpoint proof **on the legal subset**, where these atoms are nonpositive;
do not infer endpoint sufficiency from convexity of max alone. The endpoint
verifier proves the required stronger whole-box disjunction.

Inner gates remain variables and products. ReLUs after an inner gate destroy
multiaffinity, so enumerating only inner endpoints is not this mechanism. A
falsified relaxed endpoint is not original UNSAFE; only full dynamic-model replay
may establish that. V3 is never part of the cascade.

## Error bound and its limits

For a single layer at the *same* input/state h, with true gate g in [lo,hi],

    |q^T A(h,lambda) - q^T A(h,g)|
      <= (hi-lo) sup_h |q^T(E_a(h)-E_b(h))|.

If a valid difference enclosure is [d_lo,d_hi], use
D=max(|d_lo|,|d_hi|), not d_hi-d_lo. A constant nonzero difference has zero
oscillation but still produces nonzero gate-decoupling error.

For multilayer fixed-history graphs, let D_l bound the norm of E_a,l-E_b,l on
original prefix states, and K_l uniformly bound the state-Lipschitz constant of
A_l(.,lambda) for lambda in I_l over a domain containing both compared states.
For example, rho_l + max over I_l of ((1-lambda)L_b,l + lambda L_a,l) suffices.
Inserting A_l(h_original,lambda_relaxed) gives

    delta_l <= K_l delta_(l-1) + width(I_l) D_l,
    delta_0 = 0,
    delta_L <= sum_j width(I_j) D_j product_(k>j) K_k.

Multiply by the dual norm of the final affine property row for a scalar bound.
No router Lipschitz term is needed in this particular comparison: the other gate
is the original gate at the original state, known to belong to I_l. These are
conditional *pointwise* bounds, not free numeric certificates. Computing valid
D/K would be additional charged work; B4 does not silently add such queries.

This bound alone does not bound the difference between constrained optima on
different guard-feasible domains: decoupling can create additional legal relaxed
states. It also does not compare different dynamic histories. Source enclosures
and all-history coverage are still required. Wide gates and changing expert
ordering can hurt precision; neither is a necessary or sufficient failure test.
Observed wide-gate positives preclude declaring a universal narrow-gate cutoff.

## Cascade and cost

For each prefix surviving the **same original coupled prefix pruning** in both
arms, on tries the independent-gate graph with last-gate endpoints. This removes
the sigmoid gate equations but permits additional gate choices; coupled GenBaB
instead retains them and branches over its nonlinear graph. It is a change to
the proof obligation, not an assertion that one relaxation uniformly dominates.
Keep the existing numerical engine/configuration and per-property guard fold.

The on arm may reuse its own already computed prefix bounds, charged once to its
request. It must not read off-arm answers. V2 range gathering, graph construction,
all endpoints/properties, candidate replay and receipt handling share
min(history_entry+T1, remaining_request_deadline). If not closed, discard any
claim of closure and run the original coupled history on the remaining request
budget. It does not get another 300 seconds. A native call that fails to return by
T1 must be interrupted through the existing owned supervisor before fallback;
a soft timeout supplied to the numerical library alone is insufficient.

Off performs the same pruning and coupled solver directly. For a request,
T_on = T_common + sum(T_V2 + indicator_not_closed T_fallback) + T_reception/cleanup.
T_off includes all corresponding coupled calls. Savings must compensate V2
attempts on hard or unsafe histories; new UNKNOWNs count as losses, not speedup.
Both arms must cover all canonical tie-legal histories and all properties or
replay an original counterexample. A single local positive is never a request
positive. Source loading/graph conversion/launches remain within the 300 seconds.

T1 is frozen from the original 38 V2-positive full-call durations:
`docs/gate_decoupling_stage4_records_20261025.json`, SHA256
`3412bf5a07c3ff92099f151cb7cc7f63648fc01244e79b6bf8e2c0f1b161743b`.
Nearest-rank p90 is rank35 =241.00177495367825 seconds, capped at120 => **120s**.
No supplement result affects T1. This development choice is not an optimality claim.

## Approved studies and interpretation

Supplement: original97 objects (68 unexecuted,29 old source-missing),194 calls,
200/17h cap.44 of97 have no saved first-unproved history ID, including15 in the
old unexecuted suffix. Reconstruct canonical first CROWN-unexcluded history in
call; retain numerical prefix rows and new provenance. It may differ from an
old earlier unproved history and may already have been proved in that old run.
Never label those comparisons as recovered failures beaten; separate their
counts from original-ID objects. V2/V3 independently rebuild and bind the same
rule. Timeout or failed reconstruction remains unfinished, not SOURCE_MISSING
silently relabeled success. Full193 supplemental view is a new view with both
original and new costs retained; it does not overwrite the original GRAY table.

B4:18 models x20 new inputs x2 arms=720 calls/60h maximum. Primary15 multilayer
models;3 single-layer models secondary. Freeze exact sources, exclusion list,
query order, bootstrap unit/seed and all endpoints before execution. Primary
requires positive paired net gain with bootstrap95 lower>0, at least5pp higher
numerical-positive rate, and median all-request cost at most2x off. Report
UNSAFE/UNKNOWN/TIMEOUT/ERROR, losses, overlap and complete cost. Single-layer
zero gain is an expectation to test, not a result. B4 failure ends the algorithm
line as directed; a descriptive local increase does not bypass that gate.

Separate27-object2/4-slice diagnostic:54 planned calls, cap60/5h. All slices per
configuration share300s, cover first-layer margin with closed endpoints, retain
original source and guard. This factor is excluded from B4 on. No slicing code
or real run is asserted complete by this contract.

## Optional complete-request implementation and necessary controls

`verify_histories` now accepts an optional whole-history callback; default
enumeration, original-prefix pruning, tie coverage and full-model replay remain
the same. The request worker selects `GateCascade` only for explicit
`gate_cascade=true` on the pruned arm. A V2 child reloads the same bound model and
materialized tensors; this cost is deliberately included. It receives only
the parent's own prefix bounds and has no access to another arm's answers.
Both last endpoints and all properties must pass before skipping coupled work.

The existing owned process-group executor keeps the child inside the outer
request session, kills/reaps it at the phase limit and permits fallback only
after cleanup. Its optional development-policy mode makes missing RSS telemetry
explicit and does not use RSS as an acceptance gate; per-process8GiB address
space and the outer request's kernel usage accounting remain unchanged.
No new supervisor is introduced. Parent reception/serialization/cleanup share
the120s phase allowance and original request deadline. A nonfinite gate range
cannot authorize V2 but allows the unchanged coupled fallback.

Pre-execution control plan: four ordinary controls (nine-history coverage and
same-deadline fallback; partial/expired result; unavailable gate interval; actual
owned child cutoff/foreign-session rejection), then one fixed native two-layer
three-expert/three-class synthetic request through the production verify entry.
This request has analytically positive outputs and eight strictly excluded
histories plus one V2-closed history. Its weights are constructed, not trained,
and its input is synthetic. Include missing endpoint and late reception
mutations. One cumulative300s control allowance, original mathematical and
resource gates, retained tool errors; no real model queue or parameter tuning.

Control result: PASS, seven distinct tests (four ordinary, one native integration,
two unchanged-baseline regressions). The native test called the actual complete
request entry: nine histories covered, eight pruned, one closed by both V2
endpoints; no coupled fallback needed in this constructed example. Missing
endpoint and late reception mutations were rejected. Separate ordinary controls
exercise actual child cutoff/cleanup and fallback with the original deadline.
The native controller took14.660995s, including its14.587278s owned session;
the V2 phase took7.032251s. The cumulative preassigned control envelope was270s
(60ordinary+180native+30regressions), within300s. These are interface/control
measurements, not trained-model speedups, new full-request research certificates,
or an independent proof of solver bounds. No trained model/data/GPU was used.

The retained logs/receipt are in
`/data1/Kane/MOE/results/gate_cascade_controls_20261026_r1/complete.json`, SHA256
`e435ed97a115027b48b13593548cbbedead7b090918947ac6fd25488c07aa0b5`.
The public dispatcher still defaults off. B4 requires its later fixed paired
catalog, admitted M-B/M-C choices and offline complete-queue audit; these controls
do not authorize skipping that scientific comparison or change its endpoints.
