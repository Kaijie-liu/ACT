# First ordinary terminal attempt after qualified C9 live ReLU78

Prerequisite: C9_LIVE_RELU_AUDIT_20260908.md and immutable qualification SHA
17a78cc29c59acb7eba60cd97a9e95766c1e90a9a86ed8d42cec2404484d6d9e,
tests/worker exits0, every live gate passed. Earlier failed candidates stay closed.

One fresh Tiny143 original-model/input traversal, same corrected graph and
C5+c9_live_runtime_v1 rule and settings. No checkpoint substitutes for a live
network prefix. The representation runtime and all construction/resource gates
are unchanged and frozen. Its new ReLU78 HZ must content-hash match the fully
audited predecessor (all centers, sparse values/indices/indptr, all predicates,
widths and exact frame) before the ordinary affine suffix can advance to the
terminal solver. Original slot preservation and new-slot disjointness are
checked too. This binding is diagnostic authorization, not an iid-dependent
representation choice. Retain the complete C9 definition and value-view state.

The previous run already exhaustively audited this same post-HZ. This new run
checks exact content identity to that evidence instead of repeating the same
75M-entry source audit; it does not change any candidate computation or weaken
the exactness obligation. Any differing post-HZ fails closed before solving.

At actual terminal entry, require original framed input HZ and the complete
output HZ. Recollect all actual live state including original model/input/spec,
TF caches/phase slots/bounds, retained C9 metadata/views and final affine output.
Rebuild the SAME two-largest distinct-content reachable-Conv reference witness
under the same64M/1GiB bounds and require strict bytes AND entries decrease.
Then inspect actual FINAL-HZ native ingestion: zero lost/changed matrix
coefficients, bounds and integrality unchanged. No measured construction
follows this inspection. Only after these checks call the unchanged ordinary
HZSolver.evaluate_spec with45s terminal budget and native options. No attack,
PGD/BaB/input-phase split/backward/dual/LP-status rescue or parameter retry.

376 prerequisite tests plus boundary-binding guards before target. Worker240s,
tests60s, CPU1, memory16GiB; all C9 work/storage/transient and native coefficient
window/thresholds unchanged. No speed or formal-retention claim from one run.
Exclusive results/c9_first_terminal_20260908_v1/, all source/library/native-
backend and model/spec provenance frozen; automatic logs/checkpoint/result/
failure/exit retention even if terminal timeout or invalid witness occurs.

A new ADV is capability evidence only if the unchanged original concrete
network/spec replay accepts it. Any invalid witness is explicitly retained and
UNKNOWN, never a gain. UNKNOWN/TIMEOUT/ERROR yield gain0. No CIFAR or shadow
advancement on an unqualified terminal failure. Formal1870/2413 and E061/400
stay unchanged until their separate full single-path retention/promotion gates.
