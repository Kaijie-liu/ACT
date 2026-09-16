# Full upstream integration and small-study freeze (no real verification run)

Completed2026-09-16. Starting branch feat/moe-route-verification, clean/synced
HEAD c4f5ed13aec5e2357b452c1367f4568f940fe38b. All work is in a separate
namespace; original ACT/solver/checker/cache/portable/cohort sources unchanged.

## Completed implementation and controls

The full flow now actually loads the model and input, performs original
guarded capture/propagation, invokes reserve-aware proposals, then runs either
frozen V2 or V3 tail. Both arms share the same numerical and proposal policy,
cache ON, original300s total clock and298s owned work deadline. Both pay for
their own upstream work. Only duplicate full prechecking differs. The whole
driver, not just individual solver calls, is supervised. Tail evidence binds
the actual generated manifest and original request/start. Per-stage records,
query logs and publication clocks distinguish completed costs from censored
or unmeasured costs. Proposal timings overlap their enclosing stage.

Two immutable receipts are retained: attempt001 passed77 controls; after
adding an explicit deadline-at-upstream/tail-handoff control, attempt002 passed
78/78 in40.323s, zero failures/errors/skips. The latter binds final source and
protocol. Original72 mathematical, cache, relocation, process ownership,
handoff and cohort/accounting regressions pass unchanged. New controls include
real analytic model propagation and proposal in both arms, exact result equality,
invalid checkpoint binding, source/manifest/request changes, re-signed stage
cost tampering, expired original clocks, no-resume, late publication and
roster fail-stop/denominator preservation.

The analytic double/single executions both returned CHECKED_CONDITIONAL.
Recorded wall costs were4.926/3.974s; capture1.767/1.817s and proposals
1.016/1.015s. These are integration controls, not evidence of a convolutional
speedup. The full exact result matched. A later audit is structural consistency,
not an independent reimplementation of the LP proof checker. Network→HZ,
guard lowering and route exclusions remain trusted; deployed floats not proved.

## Frozen real-request design, execution not started

Read [protocol](upstream_portable_v1.md),
[control receipt](upstream_portable_v1_controls_attempt002.json),
[freeze](upstream_portable_v1_freeze.json), and
[independent clean reconstruction](upstream_portable_v1_selection_review.json).

Exactly4 new clean-correct CIFAR inputs207,209,211,214 were selected in order
after the old20-input selection, excluding recorded prior endpoints and all
parent exclusions. No route/bound queries were used. Same frozen convolutional
epoch89 model, materialized float64 boxes at2/255, all9 class properties and
all tie-legal pairs. Two arms,8 requests, alternating order,300s/request.
Both arms use new execution identities and independent upstream computation.
This is a small engineering development comparison, not a new population claim.

Selection was independently reconstructed in a separate process: identical
clean predictions, tensors and identity/exclusion union; PASS,0 issues. Freeze
SHA256625e06b8a2aa9d31dffc583c05bb610034bf9c401e33044db02e79b0d74c8552.
Raw selected tensors/exclusion inventory stay local and are not committed.
No old results were overwritten and no raw model/checkpoint/data was added.

Primary endpoint is completion of the independent checking process, with
missing/nonpositive/unresolved-route/timeout/error split. Conditional-positive
results and paired full costs are secondary. All8 roster slots remain visible;
ERROR stops subsequent jobs, UNKNOWN/TIMEOUT do not prompt retries. Separate
archival audit cannot promote a failed or late runtime result. No new SAFE,
real-request speedup or300s convolutional feasibility claim is made here.

## Next

After clean commit/push, execute the frozen8 requests once, then independently
audit and archive the final summary. Maximum nominal request budget40 minutes,
plus resource waiting and separately measured archival review. No execution
has started in this implementation/freeze turn. Output root must remain fresh:
`data/moe/results/upstream_portable_comparison_20260916_v1`.
