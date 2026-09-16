# V3: package unverified evidence, then one authoritative isolated full check

User authorized the next separately scoped engineering step after V2's90.497s
saved-proof tail exceeded the old80s reserve. This version removes a duplicate
mathematical precheck, NOT any obligation inside the remaining complete check.
No old code, numerical threshold, model, real-request experiment, reserve or
archived result changes. Cache policy/caps stay frozen and optional (off by
default). New implementation namespace: `single_check_portable/`.

## Proof and execution contract

Let C(e,r) be the unchanged deterministic rational checker on immutable supplied
evidence e and requested theorem r. For a complete positive it checks all legal
pair/property obligations, scoped reuse, support/range provenance, rational
McCormick construction, dual signs/residual-corrected bounds, threshold and
aggregation. The existing network→HZ, guard and route-exclusion trust remains.

V2 first computed p=C(e,r), packaged p, then accepted only an independent
q=C(e,r) with q=p and complete positive status. V3 packages e,r without a
claimed result, and accepts only q=C(e,r) from the isolated checker, with the
same complete positive status and timely-successful process requirements.
For fixed valid bytes and unchanged C, the mathematical q is the same. The
second evaluation and equality test were diagnostic redundancy, not a further
inequality or soundness premise in the proof. Removing them does lose that
cross-execution consistency diagnostic; it does NOT establish protection from
bugs in the shared checker implementation. Conditional guarantee is unchanged.

Transport identity is NOT weakened to compensate: the externally requested r,
source manifest hash, all logical source hashes, exact parser mode and all
executable bytes are bound. V3 metadata schema is
`PORTABLE_WEIGHTED_TOP2_SINGLE_CHECK_V3`, with decision contract
`PACK_ONLY_THEN_ISOLATED_FULL_CHECK`. An `expected_result` field is forbidden,
even if null; a V2 package cannot silently enter the V3 runtime. The pack API
has no result parameter and never reads a historical independent/precheck
result as an oracle. The output explicitly says mathematical precheck=False
and checked_result=None. Packaging invalid mathematical evidence may succeed:
its sole status is transport produced, NEVER SAFE or checked positive.

The V2 content-addressed transport is reused without a math invocation. Its
temporary unpublished null result field is removed before publishing V3
metadata. Output goes only to a fresh directory; source changes during packing
reject. The raw source manifest reference must match both the job and package.
The runtime uses the identical frozen complete checker and exact cache code.

## Sole check and result acceptance

`python -I -S verify.py --bundle-hash ... --statement-hash ...` retains the V2
read-only bundle/stdlib restriction, forbidden solver/model imports, no network
or subprocess, finite timeout/absolute deadline and bootstrap code hashes.
The caller must trust the checker entry and external hash channel; this is not
a sandbox for arbitrary attacker-provided Python. Only the response contract
and runtime metadata schema differ from V2. Each check has its own empty cache.

Execution has exactly two stages: package, check. Original full-check functions
are untouched. All malformed proof/range/dual/source cases still reject in
check; missing, unresolved-route and nonpositive cases stay UNKNOWN; static
expert violations never become UNSAFE. No second math pass runs in the auditor.
It instead independently reconstructs the result's pair/property coverage,
counts, reported-bound signs, aggregate and trusted-assumption flags, without
treating those reported bounds as independently re-proved LP bounds.

`supervise()` owns the whole driver at original_start+298s. The same clock
includes packaging, isolated checker startup/work, I/O, inventories, candidate
serialization, cleanup and outer admission review; total300s is unchanged.
No default new start, no extra budget at handoff. Candidate files alone never
admit. Incomplete/nonzero/killed/late checker or driver exits, late publication
and missing stages fail closed. Publication/OS latency limitations remain as
documented for V2; late logs may be written, but cannot promote positives.

## Controls before saved-evidence execution

New controls cover: no checker calls/result-oracle reads in packing; no result
API argument; V2/reference/V3 exact differential with cache on/off, dimensions,
ties, partial reuse, missing/nonpositive/unresolved evidence; source/property/
threshold/range/dual/constraint/order mutations; rejection of injected expected
results and wrong external requests; relocation after source deletion; one
complete checker-adapter call; deadline after math but before response; actual
two-stage inherited-budget execution and audit; pack-only/failed-check never
accepted; aggregate tampering without a precheck oracle. All prior61 cache,
portable, handoff, ownership/cohort and accounting regressions also run.
Numbered receipts preserve all failures. Old controls/source freezes are
verified before and after; no prior experiment is re-executed.

## Fixed single offline comparison (after clean commit/push)

Use ONLY the archived rank0/input114 evidence already used for V2. Parent SHA
`67cf5602723221df1b6ac1d10a04363de8b71bd385722c8a87f272353b9a21fe`;
V2 comparison receipt SHA
`c8ce46cdce1172885c866143833e7cc8893dacabc5fecafede49ffa9c8bb945f`.
Same input/domain/properties/bounds; no solver proposal, checkpoint load or
training. Cache ON; same caps, CPU one thread, nice10, no GPU; resource gate
before launch. New output root:
`data/moe/results/single_check_saved114_20260916_v3`.

Simulate220s of prior work by passing original_start=tail_start-220. This
leaves80s through the total deadline and78s before the work watchdog, NOT a
fresh300s offline window. Record simulated prior time separately from observed
tail time. No actual upstream propagation/proposal is performed. ERROR/TIMEOUT
or different result is retained without retry, enlarged reserve or substitution.
The expected exact result is UNKNOWN_NONPOSITIVE,3 positive/6 nonpositive.
Comparison to the saved original is POST-execution archival review only; it
cannot influence checker acceptance. Historical TIMEOUT is never promoted.

Report packaging, isolated process/check time, whole-tail cost, byte size,
cache counters, result identity and whether the inherited80s window closed.
V2's earlier90.497s is a descriptive historical reference, not an interleaved
paired speed benchmark. No general or full-verifier speedup follows from this
one saved case. Even a successful result only motivates a separately frozen
development experiment; do not reopen a sealed cohort or declare new SAFE.
