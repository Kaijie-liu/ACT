# Full-flow upstream reuse integration and timing freeze

2026-09-16. **CONTROL-TESTED; REAL COMPARISON FROZEN, NOT EXECUTED.**

Read `reuse_supervised_v1.md` for the fixed protocol and
`reuse_supervised_controls_attempt002.json` for code-bound controls.
96/96 tests passed, no skips; 53.47s harness elapsed is not a performance
comparison. Two actual analytic model requests traversed loading/capture,
optional-reuse proposals, portable packing, isolated complete checking and
outer admission, both CHECKED_CONDITIONAL with identical exact final results.
Both newly produced bundles passed a separate relocated `python -I -S`
check without model/solver imports. The relocation control is separately
charged control work, not free computation added to the original admission.

Original-start accounting, real near-deadline process termination, deadline
between phases, late terminal publication, invalid identities, unknown arms,
phase cost mutation, missing cost, error-stop roster and no-retry controls
passed. Partial JSON after a killed driver stays censored telemetry: absent
exact duration is null, not zero; an unaccepted partial candidate is not read
as an accepted proof. Nested proposal statistics fit inside the proposal
process window. Whole request cost equals recorded disjoint phase windows plus
residual observed clock cost. The outer work watchdog remains +298 and total
deadline +300, including admission/publication overhead. No clock resets.

Every old frozen source/experiment inventory still verifies. No upstream
reuse source was changed since its 86-test receipt. The new namespace binds
both upstream options OFF versus both ON; tail cache ON and single full check
are identical. All property/source/range/dual checks and support-first order
remain. Scoped reuse/scheduling numerical policies are not changed here.

## Preserved failures

`reuse_supervised_initial_failure.md` records an initial test import/harness
failure. Numbered attempt001 then ran96 tests with one relocation-control
error: its standalone command omitted the mandatory deadline/timeout option.
Attempt002 corrected the control invocation, not the checker requirement or
budget. Between receipts, cost collection was also hardened against partial
unaccepted candidate/stage JSON after watchdog interruption, with a dedicated
control. Attempt001 remains unchanged; no real request failed or was rerun.

## New timing freeze

- Input indices: **220, 222, 230, 232**; 4 inputs ×2 arms =8 requests.
- Same frozen convolutional E4/C10 weighted-top-2 epoch89 checkpoint, 2/255.
- Ordered clean-only selection after the sealed four-input study;902 excluded
  historical indices, including the previous207/209/211/214 and its exclusions.
- Same independently paid capture, same current checks and query order;
  no free source objects, matrices, certificates or LP answers cross arms.
- One CPU worker/thread, no GPU, alternating arm order,300s each; no retry,
  resume, range refinement, sample replacement or order optimization.
- Freeze: `reuse_supervised_v1_freeze.json`.
- Separate-process review: `reuse_supervised_v1_selection_review.json`,
  **PASS,0 issues**, exact input tensors and clean-only reconstruction.
- Real verification calls: **0**. Raw selection tensors are local-only.
- Result directory `data/moe/results/reuse_supervised_comparison_20260916_v1`
  does not exist at freeze completion.

This is a small engineering timing comparison, not a confirmation of general
speedup or additional SAFE. Coverage and cost are reported jointly, all
timeouts stay in the roster, missing/nonpositive remain distinct. Source
variation from repeated floating propagation must be reported, not silently
treated as identical mathematical inputs. Conditional independent LP checking
still trusts network→HZ, guard lowering and route-infeasibility exclusions.

## Next authorized execution step

After a clean pushed freeze, an explicit launch is available:

```sh
nice -n 10 /data1/Kane/miniconda3/envs/act-py312/bin/python -m reuse_supervised.study launch
```

The launch gate verifies source/input/control/selection identities, a clean
feature branch and synchronized upstream. Maximum nominal request allowance is
8×300s =40 minutes, plus resource waiting and separately reported archival
audits; this is not a measured ETA. This task did **not** invoke launch.
Do not mix support-order optimization or extra samples into this freeze.
