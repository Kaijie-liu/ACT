# Frozen paged-result implementation controls — 2026-10-15

Starting branch `feat/moe-route-verification`, clean HEAD
`0e00a7668833c1bf8dfaf52dd5303535e3fb983f`. The user authorizes this
default-off implementation and fixed-candidate, standard-library controls.
Commit and push this protocol before implementation. No real model, GPU,
native optimizer, dependency change, quota increase or numerical-gate change.

## Contract

Add a closed, versioned shared-source method: candidate width 8, scoped check
width 2 (dense control still 8), result pages of at most 8 rows and 64 KiB,
index at most the existing 1 MiB cap. Keep all current Work, membership,
arena, archive, bit, report and 300-second limits. Shorter deadlines are
allowed only for failure controls. Old methods/formats remain default.

Writer traverses all private-plan duties without assembling a complete result
array. Every row retains its pair, property, canonical exact lower bound or
explicit missing sentinel, and strict positive flag. Source reseal precedes
index publication. Missing evidence is a complete UNKNOWN only when every
required duty is represented; an interrupted prefix is never acceptable.

Context binds source, forest, forest-index, contract, method, invocation and
the independently rebuilt endpoint roster. The final index binds the ordered
candidate references; manifest binds index; bundle binds all members. There
is no circular bundle identity. Current shared format defines forest identity
as the index hash: these are different binding roles but **equal hash values**.
This corrects the earlier static proposal's misleading “distinct” wording;
the frozen historical proposal is not rewritten.

Production, replay and parent reports carry summary plus result reference,
not copies of all rows. Portable replay must reconstruct the source and plan,
consume every original candidate, independently recompute every endpoint,
and compare every resulting row and summary. Parent separately checks the
source-derived pair/property roster, exact signs and aggregate counts; it is
not a second independent endpoint proof. Retain all inventory/census checks.

Use at most two new portable modules and one new protocol binding. Publish
pages within existing phases. All construction, source reseal, serialization,
I/O, hashing, replay, reception and cleanup stay within the original request
deadline and budget. New-mode errors use bounded fixed codes, not arbitrary
exception stringification. RSS observation is not a memory proof. Static
capacity estimates remain conditional; verify actual traversal and report
sizes after implementation, and report deviations rather than changing caps.

## Finite controls and acceptance

Use only existing `half` and `guarded` end-to-end fixtures. Execute two sources
× two arms × two profiles (8 successful combinations), compare exact logical
results to the old same-candidate path. Guarded has 12 duties (8+4 pages).
Retain the existing four-fixture source/equation differential as regression,
not an expanded experiment. No timing-performance claim or parameter search.

Required controls: default-off/version binding; complete missing UNKNOWN;
nonpositive/zero and one/two endpoint aggregation; page size/row/bit/type
prechecks; alias mutation and duplicate seal; missing/duplicate/reordered or
extra duties; context/source/contract/method/invocation/candidate binding;
source seal failure; partial candidate; page-write exception/block; index
publication expiry; replay, parent reception and final publication expiry;
preconsumed Work and retained failed-file accounting.

On both arms, mutate a real lower bound and recompute page/index/manifest/
bundle hashes. Fresh replay, externally anchored to the same method, source
and invocation, must reject at exact recomputed-row comparison, not merely
at an old hash. Relocate each arm's package, hide its old location and run
`python -I -S -B` using only copied checker code and standard library;
forbid reads from the repository/original request. Check all duties again.

Run existing lifecycle/shared-source/consumer regressions. Preserve failed
attempts and compact logs. Archive actual pages, summary/report sizes, phase
costs, Work, source ownership and partial-result outcomes for controls.
Offline replay costs are separate and cannot rescue an online timeout.

Completion means interfaces and controls pass, documentation is updated and
branch is committed/pushed. It does not admit the eight-expert real workload,
close G1–G6, establish speedup or authorize subsequent real/GPU execution.
