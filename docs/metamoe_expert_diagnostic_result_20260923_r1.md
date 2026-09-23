# Expert UNKNOWN localized: properties never reached the native solver

## Decision

This new observation resolves the **immediate execution stopping cause**: the
expert base-feasibility query exhausted its shared allocation before any native
output-property query. It does **not** demonstrate inadequate output relaxation,
model unsafety, or that more time would yield SAFE. Output certification remains
UNKNOWN. The old R4 record is preserved and still lacks its own native trace;
this is not a retrospective reconstruction of that run.

One original MNIST0 ACT request executed after freeze `d6c1b3997`, with
implementation `985bae727`, config SHA256
`52406e625f87537e57e78c43d7a2f677d9ec734a1325adaa491ed81fb73e926e`.
All494 source/input identities and fixed environments were revalidated.
No source rebind, new sample, native-query addition, solver-option change,
precision change, budget increase or retry. See the
[protocol](metamoe_expert_diagnostic_protocol_20260922_r1.md) and
[compact archive](metamoe_expert_diagnostic_archive_20260923_r1.json).

Raw directory: `/data1/Kane/MOE/baseline_runs/metamoe_expert_trace_20260922_r1`.
Separate full review: the adjacent `metamoe_expert_trace_20260922_r1_review.json`.

## What actually happened

| Stage | Observed execution | Consequence |
|---|---|---|
| Router propagation |2.248302s|Completed|
| Candidate analysis |95.079399s|Candidate1, excluded0, no unresolved branch|
| Expert propagation, 27 layer spans |2.976055s|Completed; no resource refusal|
| Expert base-feasibility native MILP |78.313156s, passed time_limit29.999643s|HiGHS reports time limit; returned point passes existing float feasibility/integrality check|
| First output property wrapper, row0 |remaining **−48.317046s**, 0.000843s|Returns UNKNOWN **without a native MILP invocation**|
| Remaining18 output properties |Not attempted|No output-property certificate|
| Selected-score support |30.058497s|Native limit; unchanged fast range still proves nonzero|

The expert allocation is **30 seconds shared by base feasibility and all19
properties**, not the300-second outer request limit. Base returns `feasible`,
not `base_unknown`. The first property wrapper then detects its already-expired
deadline; final expert reason is `violation_region_undecided`.

The saved `solves=2` counts TWO feasibility-wrapper invocations: one native base
solve and one immediate local-deadline return. It does **not** mean two property
MILPs were solved. Actual native output-property calls: **0/19**. Thus no
violating output region was solved to feasibility, no negative output optimum
was computed, and no candidate output witness was rejected here.

Both router-candidate and expert-base calls report native status1/time limit
but return incumbents that pass the original check. Retaining their `feasible`
results is existing behavior, not a relaxed gate. Their zero objectives and zero
reported gaps are about feasibility, **not output robustness margins**.

There are four native calls in total: excluded-candidate infeasibility,
candidate1 feasibility, expert-base feasibility, and selected-score support.
Three report status1; one reports status2/infeasible. In particular, the
candidate1 native call lasts95.016011s despite a29.997423s requested limit.
This establishes that the native time-limit argument was not a hard wall-clock
bound in these calls. We did not instrument HiGHS internal phases; zero reported
branch nodes does not uniquely attribute the delay to presolve or factorization.

## Representation and endpoint boundary

The expert HZ has5636 continuous and1282 binary factors; its incoming guarded
input already has5592 continuous and1260 binary factors. The underlying expert
query retains integer variables; this is not an explicitly relaxed LP-margin
optimization. `exact=true` / `exact_witness=true` are implementation flags,
not an independently checked source-to-HZ proof.

Candidate coverage, full20-class/19-property identity and selected-score range
`[3.345338179462453,4.30685851511294]` remain unchanged. Both range endpoints
use `fast_fallback`; this is not a completed optimal support solve. All138
sparse admission events were retained. The expert status remains UNKNOWN.
No new SAFE, UNSAFE, formal-cohort opening, or author-baseline comparison is
claimed. Network-to-HZ, guard lowering and HZ numerical-policy trust remain.

## Costs and independent saved-record audit

| Item | Seconds / size |
|---|---:|
| Whole charged request, including imports, tracing, serialization and cleanup |211.276962s|
| Batch through summary, also including validation and postflight |211.557075s|
| Four native calls, non-overlapping |203.425416s|
| All43 propagation layer spans |5.200763s|
| All four HZ→MILP lowerings |0.007470s|
| Trace writes observed before last event |0.069435s|
| First separate audit |0.270439s|
| Saved-only audit reconstruction during archiving |0.278739s|
| Sampled own-group peak RSS |2,396,061,696 bytes|

The native calls account for about96.3% of charged wall time. Nested parent
spans overlap: do not add propagation, expert/evaluate spans and native sums
together. Trace cost is already charged and excludes its last write in that
specific counter. Audit/reconstruction and archive work are separate from the
request; interpreter/interactive gaps and final archive write are not included
in those instrumented function clocks. This is not an observer/no-observer
speed experiment, and tracing may perturb deadline-sensitive execution.

Outer terminal COMPLETED/exit0, no outer timeout or RSS refusal.132 hash-chained
events, all spans closed, no partial tail. Separate terminal/trace audit PASS;
a fresh saved-only reconstruction agrees except for its own audit clock.
The compact archive binds11 raw files by hash/size, all native call records,
expert reason, property attempt count and aggregate cost. Audit PASS checks
identity and recorded execution, not all underlying numerical bounds.

One inherited generic trace label needs care: `property_index=1` on the
`verify_once` entry is actually its caller's **expert index**. Output property
identity comes from evaluate_spec `lane/row/M`; the archive explicitly annotates
this. No sealed trace was rewritten to cosmetically change the label.

## Continue / stop decision

**Stop precision attribution and further solving under R1.** This trace did not
test whether the output representation can prove the19 properties. Do not add
time or optimize sparse construction on the basis of this result: propagation
and lowering are not the measured long tail.

The next separately versioned implementation should target getting valid output
obligations to the solver within the SAME request budget, rather than tightening
a relaxation that has not been queried. Two engineering concerns should remain
separable:

1. Native-call observability/supervision must not assume `time_limit` is a hard
   deadline; retain native state, validated incumbents, expert reason and actual
   remaining time. A base feasibility check must not silently consume every
   opportunity for property verification.
2. A specific, testable representation-simplification opportunity exists for
   **complete singleton route coverage**: if the sole legal expert is i on the
   entire box, its membership domain equals that box. Verify its unconditioned
   expert graph without carrying the redundant router-factor system, binding
   the simplification to the completed route-coverage fact. This would preserve
   the mathematical property, not assume fixed routing without proof. The1260
   inherited router binaries motivate a control but do not prove a speed gain.
   Ties, multiple candidates, unknown exclusions or incorrect scope MUST refuse
   this simplification. Source and numerical trust levels do not improve merely
   by using it. No implementation or real comparison of this option occurred
   in R1; it needs its own controls and freeze, not a silent rerun.

The diagnostic gap is now closed for **this observed execution**. The scientific
question of output-representation sufficiency, and the practical task of getting
an accepted full-output certificate, remain open.
