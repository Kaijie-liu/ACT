# Full upstream two-tail comparison: completed, no new positive certificate

Execution used published17c2ec9f31753a8f40620a756e39ed436a8cff65, unchanged
[protocol](upstream_portable_v1.md) and [freeze](upstream_portable_v1_freeze.json).
All8 requests ran once on inputs207,209,211,214. No resource-driven replacement,
retry, increased deadline or code modification occurred. Single CPU worker,
one thread, nice10, no GPU. The old20-input and offline114 results are untouched.

Read [compact archive](upstream_portable_v1_execution_results.json) and
[source identity clarification](upstream_portable_v1_source_comparison.json).
Raw root: `data/moe/results/upstream_portable_comparison_20260916_v1`.
Raw proofs, source matrices and checkpoint remain local, not committed.

## Complete-request results

| Input | Legal pairs | Double check: result / charged seconds | Single check: result / charged seconds |
|---|---:|---|---|
|207|2|TIMEOUT /298.045|UNKNOWN_MISSING_EVIDENCE /297.940|
|209|1|TIMEOUT /298.032|UNKNOWN_MISSING_EVIDENCE /290.390|
|211|1|UNKNOWN_NONPOSITIVE /252.058|UNKNOWN_NONPOSITIVE /220.137|
|214|3|TIMEOUT /298.035|TIMEOUT /298.048|

Completed independent checker executions: **1/4→3/4**, two gains(207,209),
no losses, common completed set{211}. Both arms have **0/4 conditional-positive
requests**. No UNSAFE claim is made from negative bounds. All8 requests stay
in denominators, including4 actual outer timeouts; no ERROR or unrun rows.
This is a four-input engineering experiment, not a superiority/significance
claim or certified accuracy. A complete checker can legitimately report missing
evidence: completion of that process is not completion of the requested proof.

Accepted checker details:

| Input/arm | Required outputs | Positive | Nonpositive | Missing |
|---|---:|---:|---:|---:|
|207 single|18|0|0|18|
|209 single|9|0|6|3|
|211 double|9|0|9|0|
|211 single|9|0|9|0|

Do not adopt any partial or late output from timed-out checker processes.
The results maintain the conditional trust in network→HZ, guard lowering and
route exclusion. Runtime rational checking verifies supplied evidence, not
the whole deployed floating-point network implementation.

## Actual full cost, not simulated upstream

All times include startup, model loading, routing, propagation/support/export,
proposals, packaging, independent checking and in-budget admission. Tail entry
uses the original request clock. Total charged time2252.686s(~37.54min), not
eight free tail checks. Resource waiting and retrospective archival review are
separate. Mean whole-request time: double286.543s, single276.629s. Paired median
single-minus-double−3.874s; mean difference−9.914s. Four observations do not
justify a general speedup, and different completion outcomes are not identical
solved work.207 single completed with only~2.06s total deadline slack.

| Input | Capture double/single | Proposal double/single | Extra precheck double | Pack single | Sole check single |
|---|---:|---:|---:|---:|---:|
|207|69.40/69.75|150.61/150.75|46.80|22.91|53.47|
|209|46.71/46.45|174.05/175.15|43.05|19.90|47.85|
|211|35.13/35.54|134.77/134.18|31.87|14.34|35.03|
|214|121.90/121.60|99.81/99.01|65.88|41.34|censored|

Times above are seconds; components are measured subprocess costs. Remaining
whole-clock overhead includes source/terminal I/O and audits. Do not add nested
query logs to phase costs again. Interrupted phase durations remain null, not
zero.207/209 double were interrupted in check;214 double in packaging;214
single in check. The retained post-last-completed windows include interstage
and cleanup time, so are not exact native interrupted-phase durations.

211 saves~31.92s observed full cost with the same NONPOSITIVE category.207/209
gain complete checking but not positive proofs.214 still does not fit. This
supports retaining the single-check path as an opt-in engineering improvement,
not claiming it resolves upstream proof-generation/strength limitations.

## Independent executions did not yield identical proof bytes

The initial archive intentionally reports exact full source/reference comparison
as false. A separate derived comparison clarifies why, without rewriting it:

- Requests and common-fact references match4/4. Route records differ only in
  per-branch elapsed fields; after excluding exactly that field, they match4/4.
- All7 guarded router source pairs match byte-for-byte.
- Three of7 joint expert HZ sources differ:207 pair{0,1} in Ac/Ab;
  209 pair{0,1} and211 pair{1,2} in Ac/Gc/b/c. Other four match.
-211's nine exact checked bounds therefore are not identical between arms;
  observed single-minus-double differences range~0.000253–0.007882, all still
  nonpositive. This is not the fixed-saved-proof equality control from V3.

These are same-policy independently computed flows, not replaying one identical
proof through two tails. The current read-only analysis has NOT isolated the
cause of numerical source variation. Do not assign every outcome or runtime
difference exclusively to redundant-check removal. No code or result was
adjusted to hide the discrepancy, and no new solve was used to investigate it.

## Audit and preservation

Frozen final roster audit PASS(0.084s). A separate process regenerated the full
summary exactly (apart from the separately recorded audit duration), verified
each outer/inner admission, clock, input and package binding, and created the
compact hash-bound archive(~0.881s). This is structural execution/evidence
audit, not another mathematical proof implementation. Aggregation controls
reject deleted/duplicate rows, changed inputs, negative costs and promoted
timeouts. A further read-only source comparison verified every compared source
against its manifest SHA. No checkpoint/data loading or solver call was made
in these archival steps. Hash inventory binds raw launch, summary, terminal,
phase, manifest and bundle artifacts; failed/partial logs remain available.

## Next decision, grounded in these records

Do not enlarge the cohort or80s reserve to turn this result positive. The next
bounded engineering investigation is the upstream evidence-generation phase:
recorded query windows total~41–70s per request while the phase takes~99–175s.
Their difference includes exact checks, rational construction, serialization and
other overhead; it is NOT evidence that native LP solving alone dominates.
Use the existing logs to select a narrowly scoped profiling/control task before
changing any algorithm.207's18 missing and209's3 missing outputs indicate
incomplete evidence;211's9 nonpositive bounds after complete checking indicate
a different limitation.214 also retains a large capture/transport burden.
Any new optimization needs separate controls, identity and budgeted development
freeze; this eight-request experiment is now sealed.
