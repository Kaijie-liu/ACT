# Repaired source-cost profiling: hard-budget synthetic diagnosis R1

Scope: implement and control-test supervision, then independently freeze the
two finite synthetic repair-follow-up calls. No real request, native LP solve,
output lower-bound generation, encoder tuning or relaxation change. The old
38-call batch stays sealed: 36 publication completions, 0/2 source-profile
completions. This new execution is not a replacement or success relabeling.

## Frozen objects and measurement question

Exactly once each, in order: E4/C3/width4/depth1/seed724; then
E8/C10/width8/depth2/seed724. These are the two previously intended synthetic
sources, explicitly new repair-follow-up identities and directory. The fixture
generator and source identities are bound before execution. Online regeneration
is charged. No selection by positive bounds or timing outcomes; no retries,
additional sizes or repetitions. Controls use ONLY E2/C2/width1/depth0/seed91.

Use the unchanged repaired `source_cost_controls.profile.run` and unchanged
OLD `bounded_evidence.stream` serialization. Do not mix batched-writer speed
measurements into this profile. All source, route, expert, guard, projection,
weighted construction and independent source checks remain. No math change.

Record identity/prefix/proposal/construction/publication/source-check phases;
within construction and checking, record router rechecks, expert propagation,
factor joins, guards, projection, weighted LP construction and their checks.
Only nonoverlapping sibling component times are summed. Parent/child timers
are not added together. Other phase costs remain explicit, not discarded.

## Supervision, acceptance, and costs

One 300s monotonic budget per call, 2s finalization reserve, sampled 8GiB
parent+owned-process-group RSS, 2 CPU threads, no GPU. Existing act-py312;
no dependency changes. Owned process groups only are terminated. The inherited
work deadline is shared by profile AND a separate receipt receiver; no renewed
300s per phase. Large hashing/JSON validation occurs in the watched receiver,
not an unbounded parent read. Preserve partial files and every failed terminal.

The enclosing monotonic wall starts before spec validation, directory creation
and launch. It includes worker imports, generation, construction, all existing
checks, publication, logging, hashes, receipt validation and owned cleanup.
Small ledger/terminal writes and terminal hash are measured by the enclosing
return; late publication invalidates COMPLETED. The batch caller records its
own enclosing wall as well. Batch preflight/configuration and batch summary
are reported separately; subsequent independent replay/audit is not hidden in
the online timing. Budgets are acceptance/cancellation limits, not guarantees
that OS cleanup or a blocked filesystem returns by an exact nanosecond.

Successful child exit alone is insufficient: invocation/spec/source binding,
all evidence hashes/sizes, the complete phase/component inventory, original
excluded+retained output-duty counts and no-output-proof flags must pass
reception. A complete diagnostic means construction and its checking finished;
it is NOT a complete positive output proof. No native output solves occur.

Append/flush ENTER/EXIT/ERROR events before/after components and phases survive
ordinary hard cutoff. Journals are not fsync-per-event crash-durable: a torn
last record stays explicitly partial. Incomplete durations are null/right-
censored with observed lower time, never zero or treated as completed work.
Cooperative deadline-error records can be late and are disclosed, never accepted
as timely completion. Instrumentation/extra logging overhead is charged; there
is no inferred uninstrumented speedup from these profiles.

## Required controls and immutable evidence

Tiny full run and fresh `python -S` relocated source recheck without model,
producer or solver imports; construct/check/publication cutoff; partial file;
receiver cutoff and exception; missing/transplanted/overclaimed/mutated receipt;
missing output obligation; source identity failure; sampled resource limit;
owned descendant cleanup; late ledger and terminal; invalid budgets and
no-overwrite; changed costs/journal identities/order/torn tail. Retain unchanged
interface and prior construction/check/publication regression suites too.

Commit tested implementation, then generate/review/commit the independent
freeze before executing this separate two-call diagnosis. Preserve every
old source binding and old failed log. No resume/late-success rescue. Whole
denominator is two, even if resource admission prevents one call starting.
Saved-only audit independently rechecks successful source constructions;
failure audits verify cost and partial records, not missing proofs.

## Decision rule, fixed before valid measurements

Rank construction/checking/publication only from completed, identity-valid
profiles and report absolute seconds, dimensions, obligations and full call
cost. A cutoff permits only observed completed substeps and censored intervals,
not an extrapolated dominant root cause. Any interface error stops causal
interpretation until separately repaired. No outcome-triggered expansion.

If a concrete repeated operation dominates, propose ONE finite, separately
controlled intervention preserving all checks and mathematical obligations.
Otherwise stop and report insufficient evidence. Synthetic component costs
cannot establish the real4099 bottleneck or a full-MoE speedup/certificate gain.
Do not reopen sealed98/4088/4096/4098/4099; do not change production defaults,
acceptance gates, historical23 source-gap claims or external/ISSTA claims.
