# Current guarded routing assignment: separate finite control

Rationale from the sealed checked-base run: the selected router branch's native
feasibility call took 92.2026 s despite a requested ~29.9974 s soft native
limit. It returned status1 with a valid point; the other branch was infeasible
in ~0.039 s. Score support's 30.044 s call is an OPTIMIZATION (nonzero objective),
not redundant feasibility. A feasible point cannot replace its global bound.

This version changes routing feasibility ONLY. It does not change support,
representation, property scheduling, tolerances, native solver options, or
BN repair. Expert checked-base remains ON in BOTH arms. No cross-query cache,
dedup, model selection, expanded cohort, smaller epsilon or more time.

The hook is only `hz_routing.hz_check_feasibility`, on the fully guarded HZ.
Lowering and matrix saving are charged in both arms. Enabled mode constructs
one fresh zero-free-factor assignment with request/input/nonce/full-matrix
binding and checks every retained constraint/domain/binary. All membership
constraints are part of that matrix; a point for an unguarded or different
guarded object is not accepted. Unsupported selectors/encodings or rejected
points cause native fallback, NOT exclusion. Every tie-legal route is still
visited. The hook does not return bounds, SAFE, or original-model UNSAFE.

Proposal/check/publication cap: min(original query deadline, query start+3s).
Fallback receives the ORIGINAL deadline and original feasibility policy.
Native local time limits were and remain SOFT in both arms: late native returns
are recorded explicitly, rather than quietly claiming a hard 30s local limit.
The 300s request watchdog remains HARD and overrides incomplete/late results.
Hardening local native fallback would be a separate change/control.

Before real execution: identity/guard contamination, ties/all-sets, unsupported
encoding, unchanged support, partial/late/exception controls and native
differentials, then commit/push and freeze. Freeze exactly old MNIST0 twice:
`router_native`, then `router_checked`, same source checkpoint and tensor,
2/255, same 19 global output properties, 300s request, 8GiB sampled RSS / 2GiB
representation budget. Native-first order is not a population speed test.

New result root, no retry/resume; stop following error/source change. Both
arms independently pay imports, propagation, routing, assignment checks,
serialization, expert queries, support, terminal and cleanup. Postflight and
audit costs are separate. No archived route results or assignments are inputs.

Save each routing matrix, candidate/check, native entry/return, all properties,
full request traces and terminal evidence. Audit checks every saved accepted
point with scalar CSR rows, cross-arm matrix identities, tie-inclusive route
coverage, native statuses, all 19 expert obligations, unchanged numerical
policy, outer precedence and full cost. Native infeasibility and network/guard
lowering still belong to the stated trusted basis; this is not a source-complete
or rational proof. Real success is optional: archive unknowns/errors too.

If successful, report the endpoint and cost difference on this single old
route-stable request only. Do not relabel historical runs or infer a general
speedup. Score-support optimization remains an independent next question.
