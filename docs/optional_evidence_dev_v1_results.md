# Optional evidence under one request budget: input98 development complete

Frozen execution `d944a9ed3` (full identity in the adjacent review), published
before launch. Raw root `data/moe/results/optional_evidence_dev_20260915_v1`.
Separate review: `docs/optional_evidence_dev_v1_review.json`, PASS, zero issues;
the reviewer rehashed recorded artifacts, audited the production package and
reran the portable proof in another isolated process. All old roots unchanged.

| Arm | Result | Evidence level | Request elapsed |
|---|---|---|---:|
|Original matched V2|TIMEOUT|Original HZ/HiGHS policy, no completed safety proof|295.697s|
|Opt-in evidence|9/9 positive, CHECKED_CONDITIONAL|Independently checked rational request evidence; upstream assumptions remain|185.317s|

Both are fresh executions of the same previously observed input98, model,
materialized2/255 domain and nine properties. Both independently recompute their
route/front-end facts. A saved-record comparison of `fact_view(common_facts)`
also returns equality. No historical source matrix or bound is borrowed.

## Evidence-arm costs inside the same300s

| Stage | Seconds |
|---|---:|
|Startup, fresh capture, propagation/support and source exports|29.274|
|26 proposals, construction and inline checking/writes|83.736|
|Local complete-request precheck|32.432|
|Portable packaging|9.381|
|Isolated independent portable checker process|30.225|
|Whole request including remaining orchestration/terminal work|185.317|

Proposal grants are capped by the remaining request budget minus the frozen80s
check/pack reserve; no clock is reset at stage transitions. Prechecking is paid
and is not substituted for the independent check. The final proof has eight
rational weighted LPs and one scoped interval fact; minimum remains exactly
`199593373867685/1125899906842624`. The fresh portable bundle is7,181,522 bytes.
The additional archival audit costs31.564s (including a30.059s fresh proof
recheck) **outside** the comparison; it does not rescue a timed-out request.

## What this establishes—and does not

This closes the engineering question on **one postselected positive control**:
fresh proposal → checked bounds → complete aggregation → portable independent
check can finish under the same nominal budget that the original production
path did not close in this run. It is not a population speedup, new-input
confirmation, route-changing proof (only pair{1,2}), or universal improvement.
Fixed arm order and one input do not justify a timing advantage claim.

The optional mode stays outside the production entry and optimal-status gate.
It is limited to this frozen single-pair request adapter, not a general evidence
backend. Network/input→HZ, ordered source binding, guards and route exclusions
remain trusted; deployed floating-point SAFE is not claimed. Input16 and the
three ACT-only requests have not been queried or relabelled. No new experiment
is queued; any broader development cohort needs its own fixed selection and
complete-obligation handling, rather than repeated tuning on input98.
