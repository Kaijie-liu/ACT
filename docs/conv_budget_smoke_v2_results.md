# V2 old-input smoke: complete terminals, unchanged solved outcomes

Execution `bd8610238` (2026-09-15), frozen protocol
`docs/conv_budget_smoke_v2.md`. Compact source-reconstructable review:
`act/pipeline/moe/results/conv_budget_smoke_review_20260915_v2.json`.
Raw root `data/moe/results/conv_budget_smoke_20260915_v2`.

All four registered requests finished; automatic, separate-process independent,
and fresh archival audits agree exactly: PASS, zero issues, V2 conformance gate
PASS. Four full packages (two per arm); two common-fact pairs equal; one complete
dynamic-model UNSAFE replay. No outer kills, no ERROR, no missing request,
no new SAFE. The full90 experiment has NOT started or been authorized here.

| Old input / arm | Old R1 terminal / seconds | V2 terminal / seconds | V2 full package |
| --- | --- | --- | --- |
| 0 / adaptive | UNSAFE / 113.036 | UNSAFE / 115.147 | yes |
| 0 / monolithic | outer TIMEOUT / 300.069 | internal TIMEOUT / 295.580 | yes |
| 1 / monolithic | outer TIMEOUT / 300.049 | internal TIMEOUT / 295.581 | yes |
| 1 / adaptive | outer TIMEOUT / 300.059 | internal TIMEOUT / 296.275 | yes |

This demonstrates better terminal/evidence closure on these controls, NOT
increased verification coverage. Old R1 remains failed and unchanged. The new
gate expressly accepts a complete non-error TIMEOUT package; it does not require
proof success. Both runs retain one solved UNSAFE and three unsolved requests.
These are sequential historical comparisons, not controlled speedup estimates.
V2 changes accounting and logging, with a5s reserve INSIDE300s, not extra time.

## What the new evidence reveals

The journals retain5,993 events,2,593 native calls and37 property results:
1/9/9/18 in execution order. All37 property results are UNKNOWN with solver
status1. Corresponding replay observations are NOT37 valid counterexamples;
only one full request has a valid dynamic-model witness. No positive-property
or partial-coverage shortcut has been introduced.

Every native call returned in these four controls. The independent standard-
library checks find zero instances where the recorded passed native limit
exceeds its remaining absolute deadline. All actual property scopes are bound
to encoding pair sets/linear rows. Full packages bind the final journal hash;
all incomplete-coverage and numerical acceptance requirements remain unchanged.

There is a material limit: compliant *passed allocation* is not an enforced
native wall-clock cutoff. The largest observed native-return overruns of local
deadline are4.166s,0.142s,0.169s,54.956s respectively. For input1/adaptive,
the call entered at29.540s, received23.357s (deadline52.897s), but returned
at107.853s with status1. It is a Tier1 native call, not an F0 positive proof.
Return timing includes the wrapper observation, not a machine-checked timing
trace; the large overrun cannot be interpreted as extra authorized budget.
No cause within the native implementation is established by these logs.

Thus the original stale-limit defect is addressed at the call boundary, while
nonpreemptible native work remains possible. The process-group300s watchdog is
still required. A5s reserve worked here; this does not prove it will always
suffice. Do not relax status0/dual/numerical gates based on primal values or
these execution checks.

## Reproducibility and next boundary

The74-test prelaunch suite passed, including real descendant cleanup, ERROR
fail-stop, late-result refusal, exact V2 request binding and source-defined
actual package/journal checks. An initial orchestration test mocked the shared
subprocess module too broadly and intercepted provenance Git reads; its mock
was narrowed to the runner namespace before freeze. No real query had run.

The final auditor reloads the two exact materialized inputs and checkpoint,
checks package identity/coverage with the existing auditor, replays UNSAFE,
and runs each journal checker in a separate Python -S process. This is evidence
and accounting audit, not independently checked network/HZ/MILP soundness.
All54 parent smoke/timing-root artifacts remain hash-identical; the old
archival inventory's53 listed artifacts are also retained. Raw checkpoints,
tensors and external repositories are not committed.

`python -m scripts.review_conv_budget_smoke_v2 --check` independently rebuilds
the compact archive without new solving. Frozen outer/worker/auditor/protocol
sources are not changed during or after this execution. No worker remains.

This requested freeze-and-smoke stage is COMPLETE. Any full convolutional
comparison must separately bind V2 for both ACT arms and preserve the frozen
CROWN arm/semantics and whole-request costs; do not use the old R1 gate or mix
V2 packages into it. No automatic full90, timeout expansion, support change,
25% retuning, replacement inputs or sealed-backend search follows this result.
