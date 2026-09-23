# Separate selected-score nonzero precheck

Scope: same old MNIST0/checkpoint/physical box, repaired BN, `2/255`, 19 output
properties, 300-second hard request cap, <=30 seconds per support query, 8 GiB
sampled group RSS and 2 GiB representation policy. Two requests in a NEW root:
`support_native` then `support_precheck`. Checked routing and checked expert
base are ON in BOTH arms. No sample expansion, larger budget, new relaxation,
deduplication, property skipping or solver acceptance change.

One explicit source rebind extracts the existing selected-score support call
into a named function in `class_separated_top1.py`. Its default operation is
unchanged; the new context patches ONLY this definedness callsite, not general
support, router feasibility, expert safety or weighted F0. Old manifests stay
sealed and require their historical source for replay.
Nonfinite unresolved endpoints are stored as null, without changing the finite
strict-sign acceptance predicate; JSON serialization must not erase a timeout.

The precheck computes the same zero-budget generator enclosure already used
as native support's fallback. For current stored HZ coordinate
`c + Gc xi_c + Gb xi_b`, all factors have absolute value at most one. Dropping
constraints gives radius `sum(abs(Gc))+sum(abs(Gb))`, a superset of every retained
guard. If finite endpoints already satisfy `lower>0 OR upper<0`, independently
check that sign using exact binary-rational stored coefficients. This extra
check may veto the shortcut, never make a crossing floating enclosure pass.
It does not tighten the range, lower a margin threshold or certify source
construction. A concrete feasible point is never used as a global bound.

Record the FULL current sparse HZ (including guards, factor/frame metadata),
scope, score row and result. Both arms pay snapshot costs. Unsupported dense
objects, nonfinite/cross-zero bounds, sign disagreement or failed checks fall
back to the original support policy. No saved range or old proof is an input.

Snapshot+precheck+publication are capped at min(original query deadline,
query start+3 seconds). Native fallback receives ONLY remaining original
time; it retains the old soft native semantics. Outer watchdog remains hard.
Late shortcut publication cannot accept a result. Partial artifacts, exceptions,
timeout and stop-after-error are preserved and audited. No automatic retry.

Controls before freezing: positive/negative/zero/touching/crossing ranges,
guard-only positivity needing native support, binary generators, duplicates,
changed dimensions/row, nonfinite coefficients, exact-sign mismatch,
unchanged default/general support, identity/guard/factor contamination,
partial/late publication, fallback remaining budget, outer cutoff/costs.
All tie-legal routes and all output properties remain required.

After tests and commit/push: freeze, commit/push, resource check, execute once,
independent saved-only audit and archive. Audit recomputes exact generator sign
without calling the production checker, binds the source HZ to the separately
saved routed matrix, accounts for all expert properties and full request costs.
Successful finite controls do not require the real endpoint to improve.

Only one old route-stable request is measured twice. No population speedup,
new route-changing certificate, author-tool advantage or source-complete SAFE
may be claimed. Network-to-HZ, guard lowering and native infeasibility remain
trusted. No other bottleneck will be optimized inside this comparison.
