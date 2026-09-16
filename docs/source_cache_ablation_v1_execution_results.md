# Source-cache attribution V1 — completed, sealed

Execution HEAD: `f855c0b713fe9874aeb7d43913a54ff64063865e`.
Frozen protocol: [source_cache_ablation_v1.md](source_cache_ablation_v1.md).
Independent archive: [execution JSON](source_cache_ablation_v1_execution_results.json).
Raw directory remains local: `data/moe/results/source_cache_ablation_comparison_20260916_v1`.

**8/8 run once, automatic final audit PASS, separate-process archival review
PASS (0 issues). No ERROR, TIMEOUT, retries, source changes, sample expansion,
range/precision changes, or order tuning.** Both arms retained 300s total,
298s watchdog, 60s proposal cap, 80s tail reserve and the same sole full check.
Pre-launch 110 controls remain valid; the new archival aggregate mutation control
also passes. Archival review cost is separate, not refunded into request cost.

## Primary result: identical evidence, small cost difference

| Input | Matrix only (s) | Both caches (s) | Both minus matrix (s) | Final result, both arms |
| --- | ---: | ---: | ---: | --- |
| 220 | 177.177 | 178.026 | +0.849 | UNKNOWN_NONPOSITIVE |
| 222 | 174.028 | 176.641 | +2.613 | UNKNOWN_NONPOSITIVE |
| 230 | 247.490 | 252.505 | +5.015 | UNKNOWN_NONPOSITIVE |
| 232 | 238.125 | 240.873 | +2.748 | UNKNOWN_NONPOSITIVE |
| Total | 836.820 | 848.045 | +11.225 | 4/4 complete checks per arm |
| Mean | 209.205 | 212.011 | +2.806 | 0/4 complete positive requests per arm |

Paired median difference: **+2.681s**. Both caches cost **1.341% more** than
matrix-only in this small, already observed development cohort. All four paired
differences have that direction, but there is one execution per arm per input,
not repeated timing or a population-performance guarantee. We do not add samples
or retries to strengthen this result.

Each arm checked **36/36 output obligations: 6 positive, 30 nonpositive, zero
missing**. All four full mathematical checker results and all 36 exact output
bounds agree between arms. Request identity, common-fact references and route
records (excluding only branch elapsed time) agree. All four router/joint-expert
source pairs are byte-identical. Raw checker-log hashes differ because logs also
contain timing and cache metadata; this is not a mathematical-result mismatch.

All four requests have a single legal pair. There are no new complete positive
certificates, no route-changing SAFE claim, and no UNSAFE conclusion. The earlier
30-row diagnosis remains **UNRESOLVED_CANDIDATE_VS_LP_RELAXATION**. Cache timing
does not separate weak candidate bounds from intrinsic LP relaxation loss.

## Attribution: source decode savings are outweighed by freezing/copying

Both arms entered **116 identical ordered queries**, including 36 weighted
queries, all PROPOSED. Each performed **348 exact dual evaluations** (three per
proposal), 72 construction checks and 160 export checks. No checking was skipped
to obtain a lower time. These are actual matched workloads, not different numbers
of budget-completed obligations.

Exclusive upstream timers, summed over four requests:

| Category | Matrix only (s) | Both (s) |
| --- | ---: | ---: |
| Source decode | 9.597 | 7.840 |
| Source freeze | 0 | 10.323 |
| Source copy | 0 | 1.590 |
| Decode + freeze + copy | **9.597** | **19.753** |
| CSR entries + rows | 75.974 | 76.299 |
| Native linprog | 28.354 | 28.317 |
| Exact dual evaluation (exclusive) | 118.405 | 117.497 |

The source cache reduces decodes from **116 to 84**, but introduces 84 freezes
and 32 hit copies. Its measured source decode/freeze/copy increment is **10.155s**;
the total proposal-phase increment is **10.190s**. This directly supports the
narrow observation that this source-cache implementation/configuration did not
pay for itself in this cohort. It does not assign every whole-clock fluctuation
to the cache or establish a general impossibility of useful source reuse.

Source-cache telemetry records **32/116 hits and 78 evictions**, no oversized
objects. Per-request evictions are 19,19,20,20 under the frozen 8-entry/64MiB/
2,000,000-node cap. The log does not identify which capacity constraint triggered
each eviction; do not claim a uniquely isolated capacity cause. Matrix-cache
telemetry is identical in both arms: per request 745 hits/768 lookups, 23 parses,
zero evictions. No capacity or admission-policy tuning was performed.

## Full cost accounting

| Charged phase totals | Matrix only (s) | Both (s) |
| --- | ---: | ---: |
| Capture / propagation | 157.976 | 156.684 |
| Proposal pipeline | 418.273 | 428.463 |
| Packaging | 74.551 | 75.553 |
| Final independent check | 181.117 | 183.172 |

Startup, serialization outside those phase windows, admission and terminal
publication remain charged in the whole request clock. Exclusive nested timers
are explanatory breakdowns inside the proposal pipeline: do not add them again
to phase totals. Automatic final audit and fresh-process archival review are
separate post-terminal costs, available in JSON. No censored or missing costs
were silently replaced with zero.

## Decision and boundaries

Seal this follow-up. **Matrix-only is the better observed engineering option
for this frozen cohort**: same complete evidence, slightly lower full cost.
The evidence supports using it as a candidate configuration in a separately
scoped future integration, not claiming source caching is universally useless.
No production/default switch is made by this experiment; source reuse remains
optional. Do not tune cache capacity or query order on these four requests.

The next scientific gap is still bound strength, not missing work: all required
obligations are now present. Before changing representation, a separately scoped
diagnostic would need independently checked primal feasibility / LP upper-bound
evidence alongside lower bounds. This run authorizes no such new solver search,
retraining, extra budget, model substitution or threshold change.

The archive checks saved execution identities, accounting and consistency; it
does not independently reprove network-to-HZ propagation, guard lowering or
route exclusions. Existing rational-check conditional guarantees and the
distinction from deployed floating-point SAFE are unchanged.
