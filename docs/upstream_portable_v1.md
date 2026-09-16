# Full upstream / portable-tail V1 engineering protocol

Separate opt-in namespace `upstream_portable/`. No frozen production, original
20-input study, cache mathematics or V2/V3 source is modified. Preparation and
clean-only selection do not launch verification. Run only after controls,
selection reconstruction, clean commit/push and an explicit launch step.

## Contrast and budget

Both arms independently load the SAME checkpoint and materialized CPU/float64
input, compute exact tie-legal route coverage and common facts, propagate the
shared guarded experts, export sources and propose rational LP evidence. Both
use the unchanged `evidence_handoff` reserve-aware proposal path (60s cap,
80s tail reserve), same proposal order and cache ON with the frozen limits.
Capture may itself exhaust the request budget; it does not receive free time
or a guaranteed 220s completion bound. Missing evidence remains missing.

Only the tail differs: double_check uses frozen cached-portable V2 (full
precheck, package, isolated full check); single_check uses frozen V3 (package
unverified evidence, isolated full check). Thus this is a duplicate-check
engineering ablation, NOT a comparison against original production matched
monolithic, CROWN, or the sealed original evidence experiment. Handoff/cache
improvements are shared, not attributed to V3. Both arms recompute all sources;
neither can read the other's results or borrow historical range certificates.

The original monotonic clock starts BEFORE request construction and disk
serialization. All imports, loading, routing, support, propagation, export,
proposal, transport, checking, in-budget audits and final admission share300s.
An owned whole-driver watchdog at298s includes upstream and the complete tail,
including nested native children. Each phase also receives this same original
clock; no reset at tail entry. Cache is request-local and cleared after check.
Only successful, timely complete independent checks are admitted. Missing,
nonpositive and incomplete route evidence remain distinct UNKNOWN states;
no negative bound becomes UNSAFE. Capture/proposal failures, killed/late
drivers and late publication cannot be promoted by later audit. Raw candidate
files and the old diagnostic handoff.next_stage string have no authority.

Execution audit binds request, manifest, stage times, process status, tail
source, clock, portable hashes and full checker scope. This is structural
review, not a second independent implementation of all LP mathematics.
The conditional guarantee still trusts network→HZ, guard lowering and route
exclusions. The frozen checker checks the supplied rational construction and
all required outputs; it does not prove deployed floating-point execution.
No production optimal-status gate is relaxed.

## Controls before selection

Actual analytic-model controls execute BOTH complete flows with real capture
and proposal, compare exact results, and reconstruct costs. Additional controls
cover exhausted clocks, invalid model binding, changed upstream source/request,
forged stage costs, late publication, arm and no-resume rules, rotated roster,
ERROR stop with unrun rows retained. All72 prior mathematical, parser pollution,
dimensions, tied/multiple pairs, partial reuse, missing obligations, relocation,
deadline, reserve handoff and owned-process controls run again. Numbered control
receipts preserve failures. No real convolutional verification occurs here.

## Small real-request comparison (freeze after controls)

Exactly4 NEW convolutional clean-correct inputs, ordered from the index after
the original20-input selection's maximum, excluding the recorded prior endpoint
union and all original selections. Same frozen epoch89 checkpoint,2/255, all
required classification properties. No route-count, bound or solver outcome
selection. Full-test telemetry is disclosed: these are new verification
endpoints, not necessarily unseen images. Reconstruct clean predictions and
materialized tensors independently before launch. Bind source, dataset,
checkpoint, input, config, controls and exclusion hashes. No sample replacement.

Eight requests, two per input, alternating order by rank; one CPU worker,
one thread, no GPU, nice10 at launch. No retry/resume; new output directory
`data/moe/results/upstream_portable_comparison_20260916_v1`. Resource gate:
16GiB available RAM,5GiB disk,load/core<=0.5; bounded24h wait,30s polling.
Resource wait is recorded separately, not hidden as verification time.
TIMEOUT/UNKNOWN continue; ERROR stops new requests and records every remaining
roster slot as NOT_RUN_AFTER_ERROR. Keep all8 slots in denominators.

Primary: complete independent checks, with complete-nonpositive, missing,
route-incomplete, timeout and error separated. Secondary: conditional-positive
counts and paired total costs; no promised SAFE improvement or statistical
population claim from four inputs. Report intersections/one-sided complete
checks and conditional positives, not property counts alone. Proposal logs are
nested in the propose phase; do not add costs twice. Capture includes routing,
support, propagation and export; retain budget journals for narrower analysis.
Interrupted stages have null duration/right-censored windows, never zero cost.
The whole clock includes unassigned startup/I/O/audit costs. Additional archival
audits are separately timed and cannot rescue failed runtime admission.

Commands (existing act-py312 only):

```
python -m upstream_portable.controls
python -m upstream_portable.study freeze --controls docs/upstream_portable_v1_controls_attemptNNN.json
python -m upstream_portable.study reconstruct
# Only after clean commit/push; NOT part of freeze:
nice -n 10 python -m upstream_portable.study launch
python -m upstream_portable.study audit
```
