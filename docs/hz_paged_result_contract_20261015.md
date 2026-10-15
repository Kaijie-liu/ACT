# Paged result format and metadata bounds

The [proposed result interface](hz_next_runtime_decision_20261015.md) can retain
all 252 properties while fitting the existing per-file caps: a full page is at
most 40,057 bytes, the index 5,431, and the worker report 164,297. These are
**static bounds for the specified successful layout**, not emitted files,
timings or permission to run the full topology. Runtime implementation remains
unapproved; all mathematical checks, limits and 300 seconds are unchanged.

Base `3aad7f87f61497cc61905483d9b2c953de6f279b`, clean
`feat/moe-route-verification`. The [calculator](../scripts/assess_paged_result_contract.py)
uses symbolic JSON lengths, registered dimensions and source hashes; its
[tests](../scripts/test_paged_result_contract.py) do not import production code.
The [bound report](hz_paged_result_contract_20261015.json) binds these files and
the previous archive calculation. No source, candidate, worker, model or GPU runs.
Schema examples maximise field widths independently; they are not valid proof
records or a claim that their status/count combinations could be accepted.

## Identity and complete checking

The exact proposed field sets are in `schemas()` in the calculator. Each page
contains at most eight unchanged `pair/other/lower/positive` rows. Every page
except the last contains eight; 252 duties need 32 pages. Canonical rational
strings retain the original 8,192-bit bound, including negative and zero values.
Missing-candidate `None` rows remain complete UNKNOWN evidence; an interrupted
prefix is not a complete result.

| Object | Binding or contents |
| --- | --- |
| Context | source, checked forest identity, forest index file hash, contract, method, invocation, checked roster |
| Page | context hash, ordinal, first-row offset, property rows |
| Index | full context, ordered candidate-reference hash, ordered page references, complete summary |
| Reports and receipt | one result-index reference plus summary; existing lifecycle and cost fields retained |

The checked forest identity and `forest/index.json` hash are **distinct** and
both are required. Production and fresh replay independently derive the roster
from checked source and plan: all unordered pairs, other-class properties and
their actual endpoint slots. A caller-supplied pair list cannot remove duties.
The parent independently rebuilds the pair/property order from the source
request; it does not perform a third complete source/endpoint proof.

Pages bind the context known before candidate generation. The final index binds
the canonical hash of all ordered tile references. Manifest references index;
bundle contains both. Index does not bind its containing manifest/bundle, so
publication does not create a circular hash. New wrapper schemas are mandatory:
the closed old V1 manifest and full-result readers remain unchanged.

Fresh replay still checks the complete source, contract, tiles, endpoint math,
receipt identity and source seal, and compares every exact result row. Rehashing
tampered results must not bypass this comparison. Publishing an index is not
acceptance: timely replay, parent reception and final cost/deadline checks remain.

## Conditional successful storage demand

All bounds use the previous fixed captured-source layout, fixed JSON-safe
relative paths (`A-Z`, `a-z`, digits, `_`, `.`, `/`, `-`) of at most 200 characters,
and the same bounded phase/progress field counts. At most two
new portable modules are reserved: up to 28 checker files each meeting the
existing 64 KiB file cap. Up to 47 implementation bindings include the existing
44, two new checker modules and a separately bound protocol file. No other new
execution file is covered. This code upper
is a design allowance within existing file caps, not actual new code size or a
resource-limit increase. An implementation needing another layout must be
reassessed before freezing; this calculation is not its execution contract.

| Generated component | Byte upper |
| --- | ---: |
| One full eight-row page | 40,057 |
| All result pages | 1,261,868 |
| Result index | 5,431 |
| Manifest / bundle | 36,907 / 111,352 |
| Worker candidate report | 164,297 |
| Parent receipt | 37,984 |
| Contract / inner terminal / invocation | 4,550 / 1,102 / 13,510 |
| Progress / successful silent worker log | 64,400 / 0 |
| **All success metadata** | **1,701,401** |

With the full 28-file code bound, the conditional whole archive is 89,164,506
bytes for binary64 candidates or 2,897,855,706 for general-rational candidates.
The latter still exceeds 128 MiB. The former does not establish admission:
Work/membership/dense-arena refusals, the missing full-source adapter and full
memory contract remain. No measured compression or speedup is claimed.

The old standalone result was outside the package. Replacing it with 32 pages
plus an index inside the package adds 32 artifact files but **33 bundle members**.
Including two proposed code files gives 353 artifacts, 351 bundle members and
358 whole-request files. Source ownership-transfer event count stays unchanged;
all new files and archive events are included. No result array is serialized in
worker/replay/parent reports; summaries and references are still charged.

## Reads and remaining limits

Preserving the existing supervision chain entails one write and ten complete
reads of each result page: two worker inventory scans, fresh replay comparison,
two archive audits, parent inventory, parent aggregation, and three census scans.
The conditional page-read volume is 12,618,680 bytes, excluding indices, code,
other metadata, source and candidates. This is a design call-count inventory,
not physical device I/O or a timing result; implementation must verify it.

Archive/hash loops currently use deadline checks rather than charging every byte
as a Work unit. Retain that distinction and charge all elapsed time; do not
silently change Work accounting or omit hashes because data are stored once.
The final census may still cause a late result to become TIMEOUT. Partial writes,
unsealed indices and failed prefixes stay in outer accounting and cannot be
recovered into online success by a later portable check.

General failure-message generation, arithmetic/JSON temporaries, complete heap
and RSS, native capture/proposal, and new implementation Work/time remain open.
The successful schema does not bound arbitrary OS exception strings or a failed
writer before serialization. These gaps require explicit handling in any later
implementation freeze, not higher quotas or a claim that all costs are closed.

## Validation and next gate

The initial six static controls passed. Read-only review identified two missing
assumptions: new nonportable protocol bindings also need space, and arbitrary
ASCII paths may expand under JSON escaping. The revised proposal separately
counts that binding and uses a fixed JSON-safe filename alphabet; no production
admission rule was altered. A seventh control checks that adding a nonpackaged
binding increases invocation bytes without changing package file counts.

The seven static controls cover symbolic length, last-page counts,
file placement, distinct forest identities, versioned wrappers/guarantee fields,
cap classification, invalid sizes and parent-report drift. They validate a
format proposal, **not** runtime mutation/deadline/replay correctness.
Together with 37 existing ledger/navigation regressions, all 44 tests pass;
the saved report is exactly reproduced and current documentation links resolve.

The next gate is still explicit approval for default-off result streaming and
fixed-candidate standard-library controls. No further full-shape attempt,
ownership-only patch, source/candidate-domain restriction, quota adjustment or
real/GPU execution follows from this analysis. G1–G6 remain open.
