# Direct candidate lifecycle and hard supervision controls

Frozen implementation scope from `d02c8ea7a747ddb1bffbc42c1ffdf627968cd7d3`.
The user approved default-off lifecycle, read-only checked roots, matched
production/portable consumer identities and hard supervision, using only fixed
candidates and standard-library stubs. No native optimizer, Torch, checkpoint,
dataset, real model or GPU may run. Existing resource and mathematical gates
remain unchanged. This stage does not establish full-topology admission.

## Versioned connection

Preserve the existing direct V1 interfaces and frozen results. Add a separate
lifecycle version, with candidate width eight and two explicitly registered
consumer arms: dense-owned and pair-scoped check-two. Each arm uses its own
consumer in production and fresh portable replay. Bind producer configuration,
both consumers, source, properties, protocol and invocation; reject mixed arms
and unknown producer configurations, including when outer hashes are recomputed.

One fixed proposer is created and set up per request. Only Driver-prepared roots
are exposed through read-only, revocable leases bound to the source object,
request, contract, task order, tile token and Work. Saved mappings and iterators
must fail after revocation. Values returned by the accessor must not be writable
aliases of the checker's Fraction values. Root access is charged; no off-ledger
root recomputation. This is a trusted in-process API, not a Python memory sandbox.

Setup failure, proposal failure, successful proposal and abort all revoke leases.
Close is attempted exactly once even after setup failure. Successful production
closes the proposer and releases its owners before portable replay. A failed or
late close cannot authorize a result. Candidates remain proposals: unchanged
independent exact checks accept both proposed/zero roles and all obligations.

## Unified supervised execution

A new default-off standard-library entry accepts only the frozen synthetic case,
arm, candidate profile and finite fault roster, never an arbitrary command,
module or callback. A fresh `act-py312 -I -S -B` worker performs source creation,
loading, direct construction, checking, lifecycle calls, encoding, publication,
fresh portable replay and cleanup. One Work and absolute deadline cover these
steps. No phase or tile resets counters. The parent clock begins at API entry,
including binding, spawn, imports, reception, cleanup and terminal bookkeeping.

Use the existing bounded owned-process-group supervision primitive. Reserve
cleanup/reception time within the at-most-300-second deadline, not after it.
Retain the existing 2 GiB control RSS observation ceiling; sampled RSS is not an
allocator upper or hard instantaneous memory guarantee. Fixed stubs must not
escape the process group. No general native containment claim is made.

All source copies, package, logs, candidate output and terminal metadata share
the original 128 MiB request storage ceiling, with explicit reserved metadata
space subtracted from the worker archive. Candidate/arena/source limits and
8M Work remain unchanged. Source fixture creation and copying are timed; logical
bytes count every retained copy, not only the portable package. Successful
completion requires timely exit, confirmed owned-group cleanup, bounded output
reception, matching package/method/coverage/cost records and timely final receipt.
File presence or worker status alone never grants acceptance. Offline relocation
has separate costs and cannot rescue online failure. Failed prefixes are retained
and missing progress counters reported as unavailable, never zero completed work.

## Finite controls

Use the existing `mixed`, `deep`, `half`, `guarded` synthetic sources, without
selection on proof results. Compare the two consumers at fixed binary64 and
general-rational candidate profiles with the old scalar oracle and V1 results.
Supervised normal controls use `half` and `guarded`, both arms and profiles.
No timing claim or large synthetic capacity run follows from these controls.

Required fault and differential groups:

- roots: identity, task order, foreign/reused/stale tokens, source/Work mismatch,
  readonly attributes, mappings, saved iterators, returned-value alias pollution;
- lifecycle: one setup, tile-bound proposal, setup/propose/close exceptions,
  re-entry, duplicate close, source mutation, no post-close reuse;
- complete obligations: explicit missing candidates stay UNKNOWN; truncated,
  duplicate, absent or foreign candidate frames and mixed method bindings reject;
- hard supervision: blocked setup/proposal/close/replay, startup exception,
  partial output, late candidate output, late reception and cleanup refusal;
- cost: preconsumed counters, one cumulative Work/deadline, archive aggregate,
  phase costs within whole-request costs, missing counters on a killed worker,
  no offline rescue and no success on late final bookkeeping;
- relocation: copied package checked under isolated standard-library Python,
  without original source/repository/solver; same method required on replay;
- unchanged direct/source/wire/consumer/navigation regression tests.

Retain first failed control reports and explain fixes. Runtime corrections may
rerun the same finite controls, but do not change a mathematical gate or select
new sources. Completion means the approved interface and supervision controls
pass; G1–G6, native dispatch, full Work/RSS bounds and real/GPU admission remain
open. The next step is reconciliation of this actual lifecycle, not automatic
native experiments or another quota inferred from partial bounds.
