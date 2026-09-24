# Opt-in construction parsing in the full proof supervisor

This is a separate execution version. The old `scoped_proof/` implementation,
its sealed CIFAR4088 timeout, input98, historical23 source-gap gains and all
previous experimental results remain unchanged. The user authorizes integration,
controls and a real comparison **freeze**, not real execution in this stage.

## One factor and one proof contract

Both arms run the complete pipeline: timed checkpoint/center intake and declared
graph capture; source/guard/output construction; independent source checking;
one native LP proposal per necessary output; fresh exact source/bound checking
and complete aggregation. Dispatch uses the original hard supervisor and original
intake/proposal/aggregate workers. Only construction and its provenance-check
entry are separately versioned. No module-global monkeypatch is used in execution.

Arms are the same instrumented construction adapter with exact CSR parsing reuse
**disabled** and **enabled**. Neither arm caches source validation, checking,
solver results or mathematical facts. Both use the previously tested original
construction algebra and original miss parser. The enabled limits remain 64
entries,64MiB serialized payload,2,000,000 cells. Immutable parsed rows are bound
to current bytes, request and policy; returned containers are fresh. Cache state
is cleared on normal return/exception; hard cutoff kills the owned process.

The disabled arm is not the original uninstrumented builder. The earlier
three-way synthetic result separately disclosed adapter overhead. This new
comparison isolates enabling reuse in the full workflow; it must not be
presented as an original-production-versus-new-tool benchmark.

Every completed construction publishes an exclusive receipt binding invocation,
whole spec, mode/policy, source and matrix hashes/size, parser counters and trace
timings. The receipt is validated within the timed source-check child before
mathematical checking. All hashing/serialization counts. The independent checker
stays uncached, runs under `python -S`, and cannot import the new constructor,
model, numerical solver or external execution. The final exact aggregate uses
the unchanged checker and acceptance; mode provenance never licenses a bound.

## One budget and explicit partial/late handling

Each real arm has 300 seconds total, two numerical CPU threads (not affinity
isolation), sampled parent plus owned-process-group RSS8GiB. The original2s
publication reserve is INSIDE300. Proposal receives half of actual remaining
work time; every lexicographic pair/property gets one equal-remaining-share
LP attempt. There are no new per-pair300s budgets, retries or extra stages.

All per-request intake/source hashes, model and center loads, conversion,
construction, receipt validation, independent checking, proposal, aggregation,
serialization and owned cleanup are charged. The supervisor return clock includes
the final cost-ledger write; a late ledger invalidates success. One-time batch
startup, configuration freeze/resource admission and later administrative audit
are separately outside per-request costs; do not label their sum batch wall time.

Timeout/exception during construction preserves events and any partial file but
does not start proposal without a complete, checked construction. After a proposal
timeout/exception, remaining time may check its saved prefix. Missing obligations
cannot become a complete proof; a proposer error remains ERROR even if partial
checks run. A valid full positive artifact that arrives too late does not override
TIMEOUT. Malformed/wrong-run/wrong-source evidence is rejected. Nonpositive bounds
mean NOT_CLOSED, not UNSAFE or a proof of relaxation impossibility.

Batch execution retains both registered terminals. Resource refusal before a
request is `NOT_STARTED_RESOURCE`, with no invented zero duration. An incomplete
supervisor receipt is preserved as `SUPERVISOR_ERROR` and cannot pass full cost
audit. Missing/duplicated/reordered terminals, policy drift or different matrix
identities are audit failures. No automatic resume or success-directed rerun.

## Controls before freeze

The new controls cover both full synthetic checkpoint-to-positive paths with
identical source/matrix/bound identities; a fully checked nonpositive case;
unchanged worker dispatch; one absolute deadline/proposal share/cost closure;
construction cutoff and exception; proposal partial prefix on timeout/error;
late complete positive evidence; receipt publication overrun; missing construction
receipt/wrong invocation/policy; changed matrix; resource/dispatch failure; and
complete batch accounting including missing, duplicate and unstarted requests.
The 17 parser/isolation tests and 53 original proof/source tests remain required.
Synthetic LP calls are allowed in these controls; no real checkpoint is loaded.

## Two real calls to freeze, not run

Use seed0 and **rank1, CIFAR10 index4096, label8**, the next rank after sealed
rank0 in the already archived100-input manifest. This is a preselected observed
engineering request, not an untouched holdout, high-accuracy model or certificate-
selected sample. The rule reads manifest order and identity only, never route
counts, old bounds or old verdicts. Do not substitute another sample if either
arm fails. The two modes run once each, disabled then enabled, in a fresh batch
directory. One pair and fixed order cannot establish stable speed superiority.

Bind the old stored float64 center and same seed0 checkpoint by SHA-256.
Load only center/weights at execution, not historical rounded endpoints,
matrices, bounds, exclusions, reuse facts or witnesses. New requested endpoints
are exact rational2/255 around that center, clipped[0,1]. The margin stays the
exact binary64 value of1e-7. All28 unordered tie-legal pairs and all9 competing
classes per pair are retained:252 freshly constructed and checked obligations
per arm. Universal gate range[0,1], original outward input/affine/ReLU/guard/
McCormick rules and continuous relaxations remain unchanged. No support search,
tighter range,25% tuning, pair exclusion, selected property or acceptance change.

The source object is newly generated and checked. Even a later complete positive
result would be `CHECKED_DECLARED_REAL_GRAPH_REQUEST`, with declared graph/program
correspondence, stored-center preprocessing and checker/runtime still disclosed
as trusted boundaries. It is not native floating-point proof. Enumerating all
pairs is coverage, not evidence of two reachable routes; no route-change claim.

## Reporting and stopping rule

Report both complete terminals, all252 statuses/prefix availability, exact-bound
counts, source-check completion, phase/event times, complete per-request costs,
sampled memory and receipt identities. Compare source/matrix hashes only where
both constructions complete; missing is not equal. Never resurrect a missed
budget by an offline audit. No additional samples, time, retries or tuning.

If both stop before LPs, report the observed construction/check stage and saved
trace; do not infer a solver or relaxation cause. If they complete, report the
actual positive/nonpositive/missing evidence, not a required success. This
engineering comparison neither upgrades the historical23 gains nor addresses
external-tool competition by itself. Freeze review hashes real assets only;
actual execution requires a later explicit user authorization.

Output: `data/moe/results/scoped_parse_proof_source4096_compare_20260924_r1`.
Config: `configs/backend_controls/scoped_parse_proof_compare_r1.json`.
