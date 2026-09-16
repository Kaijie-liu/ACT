# Portable cached checker V2: controls and one saved-evidence integration

Completed 2026-09-16. Protocol, implementation and61 passing controls were
published at `bdae1da5d` before the fixed offline replay. Read
[V2 contract](cached_portable_v2.md),
[controls attempt001](cached_portable_v2_controls_attempt001.json), and
[saved114 receipt](cached_portable_v2_saved114.json).

## Delivered

New optional `PORTABLE_WEIGHTED_TOP2_CACHE_V2` packages bind the exact parser
policy, cache enabled flag, statement, logical sources and executable bytes.
Legacy V1 and production/cohort code are unchanged. `supervise()` owns the
entire evidence tail and is the admission entry; inner candidate files alone
are not success. Original request start is mandatory: no fresh300s at the
handoff. Both precheck and isolated check use separate request-local caches.

61/61 controls PASS, zero skips/errors/failures in27.516s. Fourteen new
portable/clock tests plus47 original cache, general-evidence, handoff,
ownership/cohort and accounting controls. Old source/execution freezes were
verified before and after. New control coverage includes:

- Analytic proof copied to a new location with spaces, original source tree
  removed, then all required obligations checked with `python -I -S`.
- Cache enabled/disabled/original exact differential, multiple tie-legal pairs,
  changed dimensions, partial reuse, missing and nonpositive evidence.
- Source/property/request/code/cache-option/inventory tampering rejected;
  forbidden model/solver import, external reads and writes rejected.
- Expiry during packing or before response; owned native-call cleanup; no
  promotion from failed/late output or missing stages.
- Actual whole-tail subprocesses inherit simulated270s prior cost. Both modes
  completed around271.22s on their shared clock, NOT after another300s. This
  simulation does not spend270s or establish a real upstream model runtime.
- Publication crossing300s invalidates positive candidates; independent
  terminal accounting rejects reset allowances or omitted upstream/stage cost.

These are control successes, not new real-model SAFE certificates or a
performance comparison between cache modes on real requests.

## One fixed archived real-evidence tail

The already observed rank0/input114 source was fixed in advance, identical to
the previous matrix-parsing timing subject. One new output directory, no retry,
no proposal, checkpoint loading, dataset loading or new model verification.
All evidence, properties, ranges and numerical gates remain unchanged.

| Charged portion | Observed wall seconds |
|---|---:|
| Cached precheck subprocess | 35.032 |
| Packaging subprocess | 15.695 |
| Isolated cached checker subprocess | 38.882 |
| Other supervisor/terminal/review costs through return | 0.888 |
| **Offline tail, including outer publication** | **90.497** |

The internal checker reports38.814s; its process wall time also includes
interpreter startup and exit. The packaging function reports15.641s, while
its process costs15.695s. The driver exited at89.680s; the in-budget outer
terminal review and submission are included in90.497s. A later read-only
archival review is separately recorded as0.009s; it cannot promote a failed
execution. Clock-accounting observation points/last-record I/O limitations
are explicit in the contract, not claims of real-time OS scheduling.

Result: `UNKNOWN_NONPOSITIVE`, all9 obligations checked,3 positive/6
nonpositive, exact full result equal to the saved original. Its canonical
SHA256 remains
`987cff13fc4b2df9e3fb8469823c2d74d6f950cce44efaa18bae8f2dcd580ec0`.
No killed process or missing stage. Both caches clear at return; the isolated
check records267 hits/290 lookups and23 parses. The original request's TIMEOUT
and all previously frozen results remain unchanged. No new SAFE is claimed.

The new package totals10,827,127bytes (10.326MiB). Original logical evidence
totals733,028,660bytes (699.071MiB); content-addressed uncompressed payload is
44,108,079bytes (42.065MiB). Deduplication/compression comes from the existing
transport, not a new benefit attributed to the parse cache. The checker reads
only the new package/stdlib, not the source experiment directory. Actual
copy-and-source-removal testing was analytic; this saved-real-evidence run
checks the newly built bundle in place under the same isolation policy.

Raw package and logs (not committed):
`data/moe/results/cached_portable_saved114_20260916_v2`.
Portable directory: `tail/portable` under that root.
Bundle hash: `4eeeac3c643055573ea8dffa6ed339d2eea161ff4a92cd5807f3af22619babf0`.
Statement hash: `cae8e865a3d79ea9b1a3e939d5b1711c0fda17c4081321d023849f5309693b13`.

## Archival checks and interpretation

Both the in-budget outer admission review and subsequent read-only review
passed. A separate shell/jq review checked all raw file hashes, phase/grade
counts, total cost and original request/terminal hashes. All source hashes in
the control receipt remain unchanged. These reviews are structural and cost
checks; the isolated rational checker, not the JSON audit, checks the supplied
mathematical proof. Network→HZ, guard lowering and route-exclusion trust remain.

Controls receipt SHA256:
`f509cd96402031766e035555f2e7b2343228dc55ef3a4df17405513c5f83a5d9`.
Saved integration receipt SHA256:
`c8ce46cdce1172885c866143833e7cc8893dacabc5fecafede49ffa9c8bb945f`.

This90.497s cost EXCLUDES upstream propagation and proposals: they were not
executed. It is not a new full verification run or an end-to-end speedup.
It also exceeds the old80s evidence-tail reserve by10.497s in this one observed
case. Therefore a successful offline tail does not demonstrate that attaching
it after220s of upstream work will meet the original300s request budget.
The roughly35s precheck and39s isolated check are both charged; neither may be
silently omitted from a claimed production cost. Do not compare this total
to the previous33.90s checker-only median or102.85s cProfile time as a speedup.

## Next decision (not executed)

Keep cache optional; no new cohort/holdout was opened. Before a real-request
study, separately examine the duplicate full-check cost: whether packaging can
avoid a prior full mathematical precheck while retaining the authoritative
isolated full-obligation check, source binding and fail-closed terminal audit.
That would need its own interface/proof contract and controls; it is NOT an
implemented gate relaxation or a decision to remove checks here. Do not simply
increase the80s reserve or claim cached90.5s fits it. Any real-request comparison
still needs a separately frozen execution identity, budget and selection.
