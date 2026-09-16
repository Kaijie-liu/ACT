# Frozen upstream reuse timing comparison: completed

Execution HEAD: `805e19372730e41e2cc93f17cf2419f41ac2d858`.
Eight requests executed once, original selection, alternating arms, same
300s total /298s watchdog /60s proposal cap /80s reserve. No retries, tuning,
extra samples or order changes. All8 terminal records retained: no ERROR,
no outer TIMEOUT. Automatic final audit and fresh-process saved-record review
both **PASS**. Archive: `reuse_supervised_v1_execution_results.json`.

## Full-request outcome and cost

Every input has9 required output obligations and one exact legal top-2 pair.
The four new inputs were selected by clean correctness/order, not route count.
Times include independently charged capture, proposals, packing, isolated
checking and in-budget admission/publication observation. Both arms use the
same tail and retain all mathematical checks. `reuse_on` enables both upstream
caches; `reuse_off` disables both, with identical instrumentation.

| Input | Off seconds | On seconds | Off output obligations | On output obligations |
|---|---:|---:|---|---|
|220|275.62|177.84|2 missing,7 nonpositive|0 missing,9 nonpositive|
|222|273.90|176.79|2 missing,1 positive,6 nonpositive|0 missing,2 positive,7 nonpositive|
|230|274.87|249.98|9 missing|0 missing,4 positive,5 nonpositive|
|232|281.17|244.39|8 missing,1 nonpositive|0 missing,9 nonpositive|

- Mean observed cost: **276.391→212.251s**; total **1105.563→849.003s**,
  a23.21% observed reduction on this four-input engineering study.
- All four on-minus-off cost differences are negative; paired median
  **−66.947s**, not the ratio of means or a general acceleration theorem.
- Complete independent checker executions: **4/4 in each arm**, unchanged.
- Output evidence missing: **21/36→0/36**. Checked positive rows:1→6;
  checked nonpositive rows:14→30.
- Complete conditional-positive requests: **0/4→0/4**. Neither arm proves
  complete output safety. A positive individual margin is not a SAFE request.
- Off returns4 UNKNOWN_MISSING_EVIDENCE; on returns4 UNKNOWN_NONPOSITIVE.
  Nonpositive checked relaxation bounds do not prove the network unsafe.
- Total charged request time:1954.567s(~32.58min); resource waiting and
  post-request archival work are separate, never refunded into requests.

The improvement is **less time and less missing evidence**, not additional
certificates or additional completed checker processes. In this cohort the
on path reaches the nonpositive-bound limitation for every request. Do not
rename this a route-changing certification result: all four have a single
legal pair, and all remain UNKNOWN.

## What the measured costs support

Disjoint full-process windows, summed across the four inputs:

| Phase | Off seconds | On seconds |
|---|---:|---:|
|Capture/loading/routing/propagation/export|156.30|156.68|
|Proposal generation including construction/checking/I/O|736.28|428.27|
|Portable packaging|61.08|74.61|
|Isolated complete checking|146.97|185.24|

The remaining clock time is startup, inventories/admission and publication.
Packaging and checking grow because on produces more evidence; their cost is
not hidden. Do not add nested query or timer durations to this table.

Selected **exclusive**, nested upstream counters:

| Category | Off seconds | On seconds |
|---|---:|---:|
|CSR rows + entries access/serialization/parsing|472.78|76.23|
|Exact dual evaluation excluding its timed children|93.96|118.06|
|Canonical identity computations|65.48|81.75|
|Native `linprog`|21.05|28.15|
|Source decoding + immutable freezing + return copying|7.79|19.65|
|Save/reference serialization|8.81|13.80|

CSR categories include canonical current-content serialization, exact parse or
cache retrieval, and caller row copying; they are not pure Fraction conversion.
These are different amounts of completed work:95 queries off versus116 on;
15 weighted queries off versus36 on. All entered queries returned PROPOSED.
Recorded exact dual evaluations are285 and348, exactly3 per proposal;
source/property/range/dual acceptance checks were not removed.

The dominant measured reduction is in CSR access/parsing. Source decode
caching itself has visible freeze/copy overhead and does not show a lower
aggregate cost here. This joint ablation does **not** establish that source
caching alone helps or that matrix-only execution has the same endpoints.
Any source-only/matrix-only ablation must be frozen separately; no post hoc
option changes were made. Native solving is not the dominant measured time in
these runs, so increasing solver limits is not supported as the next remedy.

## Identity and audit boundary

Fresh review recomputed the frozen summary exactly from saved terminal files,
then bound raw records by SHA-256. Across all4 pairs of executions:

- Request identity and common-fact references match.
- Route records match after removing **only** branch elapsed times.
- Router sources and joint-expert HZ sources are byte-identical,4/4 each.

Unlike the earlier separate tail comparison, no differing stored HZ source
coefficient was observed here. Generated output evidence is still not identical:
on completes more weighted obligations. Do not claim this was a benchmark of
two identical *complete proof bundles*.

Independent-process archival review took4.514s; automatic final summary audit
took0.154s, plus individually recorded archival audits. A further exact archive
reconstruction passed. The new aggregate control rejects missing/duplicate
rows, wrong inputs, negative cost, timeout promotion and partial-positive
promotion. No model execution, proposal or new mathematical bound check was
performed during archival. Raw matrices/checkpoints remain local, not in Git.

The existing isolated rational checker validates supplied-source obligations.
Network→HZ, guard lowering and route exclusion remain trusted; JSON/structural
review does not independently prove those transformations or deployed floating
execution. Production acceptance gates remain unchanged and reuse remains opt-in.

## Disposition

Seal this study. No extra samples, larger budget or difficult-row replay.
The engineering reuse combination now has a small real-model cost/evidence
coverage result. It has no complete positive-certificate gain. Next decisions
should distinguish (a) the source cache's overhead from exact matrix reuse,
and (b) completed nonpositive LP evidence from unavailable evidence. Query-order
optimization remains a separate ablation and is not part of this result.
