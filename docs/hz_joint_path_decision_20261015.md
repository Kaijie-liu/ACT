# Joint source, consumer and portable-checker decision

Read-only review of `822fc2c4e65f0a73c9479cb8fe3a2558eef40842`, 2026-10-15.
**The approved candidate-8/check-2 consumer is implemented and control-checked;
the complete direct-source execution path is not.** This document sets out the
remaining interface decisions for approval, not an execution protocol or a new quota.

The [consumer controls](hz_pair_scoped_consumer_controls_20261014.md),
[implementation capacity](hz_pair_scoped_capacity_20261015.md) and
[consumer-only stopping decision](hz_consumer_only_decision_20261015.md) remain
unchanged. No model, checkpoint, saved numerical array, optimizer or GPU was
loaded for this review. G1–G6 remain open.

## What actually connects today

| Boundary | Implemented path at the reviewed commit | Missing connection |
| --- | --- | --- |
| Producer → source format | `wire_forest.encode` consumes the old V3 JSON; intake reconstructs ordered views and checks the old logical hash/byte limit | Direct source identity and unique-root construction |
| Source → checker | `CheckedWireForest` calls the old `Checker.run`; repeated view uses remain subject to the old root limit | Independent direct-format semantic checking, with unique values distinct from all ordered memberships |
| Native proposal → consumer | `scaled_forest_endpoints.produce` creates a plain `Session`; `Operator` allocates global COO/transpose/reference structures | Native proposal publication into the optional `Driver`; its global allocations are not bounded by the pair arena |
| Portable manifest → consumer | `wire_proof.check_manifest` uses a plain `Session`, expands candidate tiles to old JSON and reconstructs the old logical proof | A versioned portable entry using direct source checking and `Driver.receive` |
| Driver → relocation | The bounded-runtime test copies a small fixture and invokes the Driver in an inline `python -I -S` program | Production manifest, full source/candidate orchestration and terminal supervision |
| Supervisor → portable audit | `run_scaled_dense_capacity` supervises the old query and a separately budgeted offline relocation check | One new online execution graph with an absolute deadline and explicitly separated offline audit |

Source anchors: [wire forest](../scoped_source/wire_forest.py),
[native operator](../act/back_end/moe/scaled_forest_endpoints.py),
[portable checker](../scoped_source/wire_proof.py),
[relocation control](../scripts/test_bounded_runtime.py),
[historical supervisor](../scripts/run_scaled_dense_capacity.py).
These are missing integrations, not evidence that the historical controls or
separately labelled offline audits were invalid.

The proposed 96.8 MB single-store envelope and the implemented 110.56 MB owned
consumer arena describe **different representations and different resources**.
Their sum is not an end-to-end disk or RSS bound. In particular, the arena does
not include the forest, native tensors, caller-owned candidate dictionaries,
Python/Fraction objects, checkpoint snapshot or a relocated copy. The portable
checker currently has a different call graph again.

## Proposed single execution graph

This is a recommendation for a separately authorized implementation, not a
claim that its cost is closed:

`source capture → direct construction → independent source check →`
`candidate publication / Driver acceptance → portable recheck → terminal`

1. **Use a new domain-separated direct-source identity.** Preserve the source
   bytes, centers, affine compensation, factor ownership, guards and ordered
   memberships. Deduplicate stored root values, not mathematical obligations.
   Do not replay old JSON to define the new identity. The independent checker
   must establish the constructor's source/alias correspondence, not call the
   constructor and compare its own output. Historical evidence retains its
   historical checker; no old bound is imported into a new matrix.
2. **Expose a sealed checked-source interface.** Both source bytes and mapping
   storage need immutable ownership, not merely read-only object fields. Bind
   source, request, property, ordered factors, pair and cumulative accounting;
   fail closed after transfer, release, abort or foreign binding. No native
   proposer receives a writable alias to checked storage.
3. **Keep proposal and acceptance distinct.** Candidate width remains eight;
   the optional pair-scoped exact checker uses blocks of two. Publish complete
   bound frames before acceptance. Retain both proposed and zero roles, all
   rows, intermediate exact checks and separate objective/dual propagation.
   A symbolic-zero format is only a possible separately versioned encoding;
   it is not implemented by this review and must not erase polluted legacy
   values. General rationals must not be rounded into a binary64 profile.
4. **Make the portable entry use this same declared contract.** It must rebuild
   the checked source and all necessary results from the package alone under
   `python -I -S`, without a model, solver, historical directory or trusted
   online receipt. Bootstrap only the package's hash-bound code path; do not
   inherit `PYTHONPATH` or ambient import locations. New typed frames must not
   quietly round-trip through old JSON.
5. **Use one online 300-second deadline.** The proposed online success contract
   includes source work, native setup, acceptance, serialization, independent
   portable recheck, terminal reception and cleanup. Phase subprocesses inherit
   remaining time, never a fresh allowance. A later copied-package relocation
   audit is separately charged offline and cannot rescue online failure.

The restricted live-checkpoint/source-capture design is still unimplemented;
synthetic captured-source controls would not close that part of the graph.
See [source lifetime](hz_source_lifetime_protocol_20261014.md) and
[exact-domain limits](hz_exact_domain_20261014.md). The online portable choice
above needs explicit adoption before implementation; it does not reinterpret
the historical offline experiment.

## Ownership and cost obligations

| Phase | Identity and live ownership to bind | Costs that must remain visible | Release / failure condition |
| --- | --- | --- | --- |
| Source capture | Checkpoint/source declaration, materialized input and preprocessing; private snapshot | Reading, parsing, copying, hashing; original plus snapshot retention | No arbitrary executable deserialization; partial snapshot unaccepted and charged |
| Construction / source check | Direct graph, unique root values and every ordered membership | Source coefficient visits, exact arithmetic, descriptors, construction/check overlap, hashes and I/O | Source becomes consumable only after independent complete check; no identity-only shortcut |
| Native proposals | Checked source and ordered query roster; native COO/transpose/workspace | Setup, tensor copies, proposals, buffers, serialization and publication | Producer exit/descendant lifetime accounted; a partial frame is not a receipt |
| Exact consumer | Same checked source, candidate frames, pair maps, check blocks and results | All row/factor visits and exact operations, frame access, planning, sealing and invalidation | Transfer invalidates the prior owner; any later-block fault prevents full acceptance |
| Portable recheck | Package identity, source, complete candidate inventory and checker version | Fresh source check and all necessary exact rechecks, file reads and process startup | No receipt shortcut; missing, changed or late evidence remains incomplete |
| Reception / archive | Attempt identity, complete obligation roster, terminal references | Logging, inventory/hashing, retained prefixes/copies, cleanup and process reaping | Exactly one terminal; no late success replacing timeout; cleanup failures retained |

Work, bytes, owned buffers and RSS are separate axes. Parsing, hashing, I/O and
native operations must be costed even when they do not increment today's Work
counter. Resource policy comes from a trusted frozen protocol, never from the
evidence being checked. Numeric quotas and current production gates stay
unchanged. No per-tile reset or uncharged fallback is allowed.

An allocation-lifetime bound must cover native and exact-checker coexistence,
temporary arithmetic, caches and the complete process tree. An RSS/AS stop gate
is a refusal mechanism, not proof that a complete request fits. A bounded
format does not guarantee 300-second completion under arbitrary OS/I/O behavior.
The new path still has **no complete implementation-bound Work/RSS upper or
approved absolute resource ceiling**.

## Comparison boundary

The initial representation differential should fix the same checked source,
candidate batch width eight, candidate values, all obligations and resource
policy. Compare dense-owned acceptance with pair-scoped check-two acceptance;
also retain the legacy consumer/expanded oracle for exact result checking.
Account for changed access counts and failure order; do not claim invariant
Work or speedup from equal answers.

This is distinct from the historical width-one versus width-eight native
experiment, which changes both proposal batching and checking. A later
whole-path comparison needs both complete arms and all common preprocessing
charged independently. It cannot reuse an old JSON/native timing as the new
direct-path reference. No performance or real-model experiment is frozen here.

## Decision and next authorization

Stop isolated consumer capacity controls. The remaining useful implementation
unit is the direct source/independent checker/ownership interface **together
with its production and portable connections**, not another local cursor.

Before writing that implementation, request approval for a default-off version
and bounded synthetic CPU/standard-library differential and fault controls.
This explicitly moves implementation-correspondence work ahead of final runtime
cost closure; it is not implied by the previous consumer approval. Required
controls include source and membership mutation, alias pollution, same-candidate
exact agreement, missing/foreign/partial/late evidence, relocation, ownership
invalidation, deadline and complete accounting. Their purpose is to test the
declared path, not to increase a passing-test count.

Such approval would **not** raise quotas, change the 300-second budget or
mathematical/numerical gates, authorize real checkpoint/data or native/GPU
experiments, admit the eight-expert model, or establish an algorithmic benefit.
Initial boundary controls use fixed candidates or standard-library stubs only;
they must not execute Torch or the native proposer under a different label.
Native dispatch, allocation and trajectory correspondence remain open after
those controls and require a separately scoped decision.
Full resource analysis and any policy/execution approval remain later gates.
If implementation is not authorized, retain this decision and stop; do not
repeat static refusal arithmetic as new experimental progress.

## Review and validation

Two read-only source reviews independently identified the old Session/JSON
joins and distinguished local payload bounds from full lifetime bounds. This
is code review, not independent proof of every network bound. The existing
project-layout and review-handoff suites passed all 13 tests using
`act-py312/bin/python -B -S -m unittest scripts.test_project_layout scripts.test_review_handoff`.
All local links in the three changed documents resolve, and `git diff --check`
passes. These are navigation/historical-binding checks only; no new solver,
capacity execution or microbenchmark is part of this stage.
