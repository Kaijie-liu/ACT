# Combined checked execution vs author sufficient adapter: separate smoke

Freeze BEFORE execution. Two original inputs CIFAR10/0 and MNIST/0, four
requests in order ACT/author then author/ACT. No result-dependent input,
method, property, tolerance, support or time changes. No retry/resume. Stop
remaining batch on ERROR, preserve timeout/resource/partial evidence.

Same full original checkpoint and materialized CPU float64 normalized box
`2/255`, 19 global classification margins at `1e-7`, zero-filled unselected
classes, raw-score selected top-1 (requires nonzero selected score). ACT
covers every tie-legal route. Author's unchanged backend/adapter proves the
stronger sufficient condition of strict route dominance and selected-score
sign, together with all global margins. It is NOT direct dynamic-MoE export
or a verbatim reproduction of the paper's table. Numerical grades separate.

ACT explicitly enables three request-local, checked hooks: expert base point,
guarded route feasible point, and generator-box nonzero precheck. Defaults
remain off. Independent current scope/nonce, all extra constraints, property
queries and original native fallback remain. No deduplication, relaxation
adjustment, proof reuse from another request, or new solver gate. Existing BN
repair shared with the previous control; no new production-source rebind.

Both arms: hard 300-second process-group budget, sampled 8 GiB RSS, CPU/two
threads, separate pinned environments. ACT sparse representation 2 GiB,
expert <=30s and 10% base allocation, route/support <=30s; checked attempts
spend existing allocations. Clock includes launch/imports, file/environment
validation, original model load/forward, conversion, artifacts/checks, native
solving/cleanup and result publication. Receipt, terminal, hashes and batch
publication are recorded as postflight/batch cost, never hidden solver time;
independent audit/replay and offline selection are separate. No prior timing
is substituted for a newly executed arm. Full raw streams/partial artifacts
retained, raw matrices and weights not committed.

Tests must cover disabled/partial/full options, hook restoration/nesting,
same-request identity, default native behavior, all tie routes/properties,
remaining budgets, late/partial candidate precedence, errors, complete cost,
source mutation, immutable directory and complete roster. Reuse prior detailed
point/sign controls; new audit independently checks physical property rows,
VNNLIB endpoints/disjuncts, recorded native queries and exact stored-HZ sign.
UNSAFE must pass separately bound full original model replay. Positive records
are not independently re-proved network certificates.

Smoke gate: all four executions and independent audit/replay valid, center
violations replayed on CIFAR10/0, complete author numerical and ACT policy
positives on MNIST/0. A well-recorded timeout is an audited failure, not a
passing smoke. This is an interface/regression gate, NOT expected superiority.
Passing permits ONLY separate preparation/freeze of a small clean-correct
cohort, not automatic launch. The later selection must use raw index order,
exclude all previously used MetaMoE verification inputs and use no route/bound
outcomes. Freeze selection scope before forward-only selection.

Upstream conversion impact is a separate saved-identity analysis. Finite BN
conformance does not prove all-domain network-to-HZ inclusion; guard lowering,
solver numerical acceptance and deployed float semantics stay distinct.

Pre-freeze controls: 132 focused ACT tests and 97 overlapping pinned-worker
tests pass. Initial new restoration test inspected the general solver symbol
instead of the intentionally routing-local binding and failed; corrected the
test to observe `hz_routing.hz_check_feasibility` and additionally assert that
the general native feasibility function is untouched. No implementation,
acceptance gate or tolerance was relaxed to pass the test.
