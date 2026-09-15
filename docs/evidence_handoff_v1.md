# Evidence reserve handoff V1: separate engineering revision

This stage implements the next bounded engineering step after the archived
new20-input evidence study. **It does not rerun that cohort or change its
20 TIMEOUT results.** Source namespaces `act/`, `scripts/`, `moe_evidence/`,
`portable_proof/` and `evidence_cohort/` remain unchanged. Existing experiment
freezes continue to verify. No new real-model request, sample selection,
bound proposal or holdout reopening is authorized by this document.

## Control-flow change and unchanged proof contract

`evidence_handoff/proposal.py` delegates to the original deterministic proposal
loop. It retains the300s request clock,60s per-proposal cap and80s downstream
reserve. A narrow budget proxy distinguishes these outcomes:

| Condition | Outcome |
|---|---|
| A grant with reserve80 expires but request work time remains | Return a recorded reserve handoff; continue to unchanged precheck |
| Whole request work time is exhausted (including2s terminal reserve) | Original budget exception; TIMEOUT, not handoff success |
| Invalid binding, mathematical check failure, I/O or other exception | Propagate failure; do not disguise it as budget exhaustion |
| Original proposal loop returns normally | Continue with exactly its committed evidence |

The first case includes expiry during source decoding/validation, weighted
construction, or between the two grant checks around proposal-journal writing.
The proxy catches only its own `ReserveHandoff`, not arbitrary
`EvidenceBudgetExpired` raised inside a solver or checker. It never increases
the deadline. A proposal journal can retain a PENDING entry and an unreferenced
construction file; neither is adopted as a proof. Only the original manifest's
atomically published references participate in checking. Already committed
proof obligations survive the handoff, while all missing obligations remain
missing. `handoff.json` is diagnostic only and has no acceptance authority.

The unchanged checker still covers all tie-legal pairs and all requested
properties, using scoped reuse or rational range/McCormick/dual checks.
Missing evidence returns UNKNOWN; a nonpositive bound does not prove UNSAFE.
The portable isolated checker, property thresholds, trusted upstream
network-to-HZ/source/guard/route exclusions and production optimal-status
policy are unchanged. No floating-F0 assumption is reintroduced.

## Integration boundary

The new optional worker replaces only `propose`; capture, precheck and package
delegate to the old phase implementations. The evidence-only driver retains
capture→propose→precheck→package→isolated-check and the same acceptance function.
It records schema `EVIDENCE_HANDOFF_EXECUTION_V1`, not an old execution identity.
Its output is a **candidate terminal**. A caller must supply the same outer
start time and a298s whole-driver watchdog; an inner phase timeout alone is
insufficient for hashing/serialization cleanup. Analytic integration controls
exercise this driver under the existing owned-process watchdog and audit the
complete positive proof with the old auditor.

There is deliberately **no new cohort launch entry or real-model execution
freeze**. The old cohort supervisor continues to name the old driver. These
controls establish an optional implementation, not a registered population
experiment or a completed production rollout. A future execution protocol must
bind this new driver/worker and preserve outer terminal audit before any new
real-request run; it must not edit the old run directory or relabel old rows.

## Controls and regression scope

New controls test reserve-only versus actual deadline exhaustion; expiry during
support checks, weighted construction and the second grant; preservation of a
previously proved property; UNKNOWN with missing properties through a portable
`python -I -S` check; propagation of invalid evidence and I/O failures; one-shot
execution; and rejection of a late/incomplete isolated check.

Under nonbinding time budgets, original and new proposal paths yield identical
manifests and checked results on analytic E/C variants, tied multi-pair routes,
partial reuse and a nonpositive control. A separate tiny source-defined model
exercises actual capture, worker processes, proposed bounds, packaging,
isolated checking and a fresh terminal audit. This is not a CIFAR query.
Existing general-proof, cohort-supervisor and cost-accounting regressions are
also run; `docs/evidence_handoff_v1_controls.json` pins the source identities,
test names and counts. No test drops a necessary obligation or modifies a gate.

## One saved-evidence cost profile

The profiling subject was fixed to the first saved precheck in the archived
cohort: rank0, input114. The parent index is pinned by SHA256
`67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe`.
This chooses an already observed record by order, not a new input or positive
bound. `evidence_handoff/profile.py` checks its bound file identities, invokes
the **unchanged** checker with `cProfile`, and compares the result exactly with
the saved precheck. No checkpoint/data load, model verification, solver call,
range tightening or packing rerun occurs. It has a separate300s profiling
watchdog/280s cooperative limit; this is not extra time for the original request.
The original TIMEOUT cannot be promoted. Raw profile records are kept in a new
directory; no retry or overwritten profile is permitted.

The recheck reproduced UNKNOWN_NONPOSITIVE,3 positive and6 nonpositive
obligations, with102.8523s instrumented wall time. Findings for this one checker:

| Operation | Calls | Instrumented time |
|---|---:|---:|
| File loader | 61 distinct files,61 calls | 2.6003s inclusive |
| Rational conversion | 23,257,441 | 53.9831s cumulative |
| Sparse matrix `_entries` parsing | 232 | 36.4919s cumulative |
| Source/LP canonical identity | 166 | 10.1779s cumulative |
| Sparse dual evaluation | 29 | 40.4244s cumulative |

**Cumulative times overlap and must not be added.** Fraction-library exclusive
time was48.8550s. About733.00MB of logical files were read, but each referenced
file was loaded only once in this pass. The repeated work is not repeated reads
of a single pathname: source matrices occur embedded in different records and
are repeatedly converted/checked; identity construction reserializes content.
This observation supports investigating immutable exact parsed-source reuse,
not assuming that faster storage or a dictionary of file contents will solve
the issue. It does not yet quantify an optimization's savings.

This is one instrumented saved checker, not an uninstrumented timing pair,
cross-model profile, end-to-end speedup or new positive proof. Profiling affects
cost; comparison with the old uninstrumented precheck would be invalid. The
profile does not show how much of the entire cohort can be saved. The other12
saved prechecks and all original terminals remain unchanged and are not silently
rechecked or completed. See `docs/evidence_handoff_v1_profile.json`.

## Next bounded decision

The reserve-handling defect now has a separate control-tested implementation;
it is not evidence that the old cohort would acquire SAFE. The prior7 complete
but nonpositive local prechecks remain a reason not to make that inference.
Keep this fixed while considering a **separate** exact-checker optimization:
reuse immutable parsed matrices keyed by checked content/source identity, with
explicit per-reference source, frame, objective, guard and property checks.
Test stale-cache, changed coefficients, changed factor order and cross-request
bindings against an uncached reference before evaluating any timing benefit.
Do not remove the independent check or rely on unchecked solver status.
No cache optimization or new real-model run is included in this stage.

Commands in the existing ACT environment:

```
python -m evidence_handoff.controls
# One-shot profile of the pinned stored record, not a verifier experiment:
python -m evidence_handoff.profile
```
