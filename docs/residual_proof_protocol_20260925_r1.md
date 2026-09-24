# Shared residual evidence in the complete 300-second proof pipeline — R1

## Decision and scope

The user requested full-budget integration and a test of whether the saved
router budget yields complete output certificates. Synthetic route-only
timing and exact differential are archived in
`shared_route_residual_result_20260924_r1.md`; they do not predict a real result.
New namespace `residual_proof/` changes no previously frozen source. No original
production verifier, numerical policy, output relaxation or training is changed.

Two execution modes use the same source intake, fixed final-affine route
candidates, lazy retained source construction, native LP candidate provider,
exact source/LP checking, lexicographic obligations and budget allocation:

- `pairwise`: old per-margin candidate and exact checker, 56 physical files.
- `shared`: 8 equality potentials in one bound file, exact shared-residual
  subtraction deriving the same 56 ordered margin lower bounds.

Both modes have seven phases: intake → route proposal → route check → retained
construction → source check → output proposal → exact aggregation. The change
is evidence organization, not free evidence, stronger ranges, a cache or a new
solver. Routes are checked again inside construction and subsequent source/
aggregation checks in both modes. No check is silently removed to gain speed.

## Proof and acceptance contract

The unchanged source checker validates the input outer box and affine/ReLU
enclosures, shared factors, joins, pair guards, property projections and rational
McCormick output LP construction for the declared real graph. The shared route
checker validates signed EQUALITY potentials only, differences residual vectors
before box bounding, and never infers strict dominance from a zero lower bound.
See the residual composition lemma in
`shared_route_residual_protocol_20260924_r1.md`.

Route candidates bind source, request, prefix, invocation, mode and file index;
the shared certificate also binds factor order. A complete file manifest and
independent route receipt are required BEFORE retained construction. Partial
route evidence remains inspectable offline but is not an online acceptance.

All 28 tie-legal unordered pairs × 9 classification properties remain in the
252-row denominator. A row is discharged only by a checked strictly positive
outsider-minus-insider route margin proving its pair impossible, or by a fresh
strictly positive, residual-corrected rational output lower bound. Missing and
nonpositive bounds do not close. Retained properties keep original global
indices; no reindexing, property dropping or old-bound reuse. A partial/failed
candidate phase cannot produce a full positive request even if it leaves files.

The independent checker runs under `python -S`, forbids model/solver/producer
imports and external process execution, and checks all necessary sources again.
Native LP status/objective is not proof. Positive acceptance is for the declared
real network graph, conditional on graph/program correspondence, stored-center
preprocessing and checker/runtime. It is NOT deployed floating-point execution
verification. Multiple retained pairs do not prove route change. No UNSAFE is
inferred from a lower bound or a relaxation assignment.

## Unified deadline, reception and accounting

Every phase shares one absolute 300 s deadline, two-thread CPU environment and
sampled parent-plus-owned-group 8 GiB cap. Reserve 2 s for terminal publication;
output proposal receives half of the actual remaining work time, as before, to
leave time for exact checking. No stage can renew the deadline. Native output
queries split their actual remaining proposal budget evenly across the remaining
registered obligations. Loading, parameter/input hashing, conversion, source
construction/checking, candidate generation, native solves, exact aggregation,
serialization, process imports, owned cleanup and receipt publication are charged.
Final ledger writing and later administrative audit are identified separately;
ledger overrun invalidates acceptance. The supervisor kills only its own group.

Source and route receipts, partial files, exceptions, resource refusals and
outer timeouts survive. Complete-cost audit checks phase order, nonrenewed
deadlines, terminal/file hashes, counters, sampled RSS, original coverage and
status consistency. Offline recovery never upgrades the budgeted terminal.

## Controls and regression gate before real freezing

23 integration controls plus 98 unchanged math/source/watchdog controls. Cover
synthetic checkpoint intake through complete proof, same-object two-arm retained
matrix equality, all-tied routes, original-index gaps, negative bounds, missing
or transplanted route/output evidence, missing source layers/properties, partial
publication, late positives, cutoff, owned descendants, RSS, parent/worker
exceptions, relocation-independent saved-only audit, full-cost corruption,
terminal overrun and original batch denominators. No real dataset/checkpoint is
decoded by these controls; synthetic native LPs exercise the actual provider.

Initial development run: 20/21 controls passed; one copied test still read the
old `frontier_only_positive` report key. Corrected the assertion to the new
`shared_only_positive` key and expected equality on the synthetic control. This
was a test migration defect before freezing, not a mathematical discrepancy or
real run. Added source-binding and cost/RSS mutation controls before final gate.

## Finite real comparison (freeze after passing controls)

Exactly two calls, pairwise then shared, each once at 300 s; new directory
`data/moe/results/residual_proof_source4099_compare_20260925_r1`.
Use seed0 from the existing 100-input selection, next sequential rank3,
CIFAR test index4099, label4, original materialized center, exact 2/255,
clip [0,1], and the exact binary rational representation of the existing 1e-7
margin. Bind checkpoint, center, selection manifest, source-gap ledger, tested
implementation, protocol and environment hashes. Selection uses no route count,
bound or outcome. It is an observed engineering request, NOT a new holdout.

Only the old center is read from its materialization, not old source matrices,
route census, bounds or witnesses. All new matrices require their own evidence.
Both modes calculate their own source/routes; no cross-arm answer reuse. Older
inputs98/4088/4096/4098 stay sealed, with their failures and the historical23
source-containment limitations unchanged. This experiment is not an external
baseline run and cannot establish superiority over public tools.

Commit/push implementation and test gate, freeze and independently review the
two-call configuration, commit/push the freeze, then execute the fixed roster
once under this user's request. No automatic retries, more samples, longer
deadlines, changed margin, or parameter/relaxation tuning after results.

## Endpoints and decision

Primary: is a complete source-accounted output proof accepted WITHIN the same
request budget in each arm? Report both positive/NOT_CLOSED/timeout/error/resource
states and all252 original duties. Also report online route receipt, retained
pair/expert count, construction/source/proposal/aggregation completion, native
query and checked-bound count, full and per-stage costs, and sampled memory.
Compare fresh source identities and retained matrices when both exist. If one
does not publish matrices, state that differential is unavailable, not passed.

If only routing becomes cheaper but complete output proof remains absent,
report precisely that and the saved stopping stage. Do not diagnose output
relaxation without output evidence; do not reopen a sealed input. Independently
recheck saved evidence and archive both calls, including failed attempts. Only
then propose the next bounded step. A source-complete real-graph certificate,
actual route-changing evidence, external competitive advantage, and an ISSTA
acceptance claim remain distinct requirements.
