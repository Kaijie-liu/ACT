# Full-flow upstream reuse timing protocol V1

New namespace `reuse_supervised/`; existing source/experiment freezes remain
unchanged. Both arms use identical current-source validation, exact checks,
proposal algorithm/order and the same single full isolated portable checker.
`reuse_off` disables upstream decoded-source and exact-CSR retention;
`reuse_on` enables both. Tail exact-CSR caching is ON for both arms. No
precheck removal, change of scheduling or weakened numerical acceptance is
part of this ablation. Cached and uncached proposal code/timers are identical.

## Clock and evidence

The parent starts the clock before request construction. All child phases
receive that original start: loading, propagation, routing/support/export,
proposals, current-source reads, checks, cache work, serialization, packaging,
isolated `python -I -S` checking and in-budget structural admission are charged.
Work watchdog at +298 seconds, total +300 including terminal overhead. Owned
processes only are terminated; no retry/resume or late positive rescue.
Cache reserve handoff stays 80 seconds and each proposal cap stays 60 seconds.

Terminal audit binds plan/request/manifest/report/checker and phase order,
checks cache options/limits/scope/cleanup, report elapsed against the proposal
window, and nonnegative nested exclusive/inclusive timings. No terminal audit
claims independent validation of network→HZ, guard lowering or route exclusions.
Conditional rational checks are not deployed-float SAFE. Missing, nonpositive,
route-incomplete, TIMEOUT and ERROR stay separate.

Whole observed cost is measured at outer publication, after outer terminal
serialization. A publication reaching +300 invalidates admission. Phase windows
are disjoint; whole cost equals their recorded sum plus residual clock time.
Absent/cut-off phases have null exact time and an observed window when known,
never a zero-duration successful phase. Query and nested timer costs overlap
proposal time and must not be added again. Adapter report elapsed excludes its
own write, but the enclosing proposal process window includes it. Residual
clock cost includes setup, inventories, in-budget audits, publication and
unrecorded interrupted work. Subsequent archival audits/rechecks cost separately
and cannot rescue a deadline. Cooperative clock checks alone are not a watchdog.

## Frozen timing design (after passing controls)

Four new ordered convolutional clean-correct inputs, same epoch89 checkpoint,
2/255, two arms = eight independently charged requests. Start after the sealed
previous four-input selection; exclude its full exclusion set and all four
used indices, plus the existing historical exclusion inventory. No selection
on routes, bounds, proof readiness or outcome. Materialized tensors, dataset,
checkpoint, configs, source hashes, control receipt and ordered jobs are bound.
Separate-process clean reconstruction must pass before any launch.

Single CPU worker/thread, no GPU, resource gate; alternating arm order within
each input block. Same 300s request budget, 60s cap, 80s reserve, no repeat or
sample replacement. ERROR stops, with every remaining roster slot recorded
NOT_RUN_AFTER_ERROR. Both arms independently recompute upstream HZ; no free
sharing of source objects or evidence between runs. Floating propagation may
produce differing source coefficients; compare source identities/semantics
afterward and do not assume byte-identical upstream problems from input identity.

Primary: paired complete independent-check outcome AND full observed request
cost (include timeouts). Report conditional positives, missing/nonpositive
obligations and all failure states separately. A faster UNKNOWN is not a faster
proof. Completion/cost tradeoffs are acceptable outcomes, not grounds to retune.
Secondary: exclusive decode/hash/copy/CSR/construction/check/native/serialization
costs and retained cache statistics where reports exist; missing reports stay
missing, not zero. Four inputs support descriptive engineering comparisons,
not statistical superiority or broad speedup claims. No effect-size threshold
is used to select samples or reopen this run. Source-only/matrix-only attribution
and query-order optimization require separate future protocols.

This task freezes only, not executes the real comparison. Output will be
`data/moe/results/reuse_supervised_comparison_20260916_v1`. Old studies remain
sealed. Launch only after a clean pushed freeze and independent selection review.
