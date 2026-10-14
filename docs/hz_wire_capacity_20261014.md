# Implemented wire path: capacity and lifetime assessment

Static assessment at `aa7bc74a95f8a49ede1858bb2fd698a0eab39bb6`,
2026-10-14. No model/source construction, solver, native numerical code or GPU
was run. The [finite interface controls](hz_wire_interface_controls_20261014.md)
remain completed; this follow-up determines what they do **not** admit.

**Decision:** the registered eight-expert topology still cannot complete under
the unchanged policy. Stop capacity retries on this version. Lossless wire
storage does not remove logical-byte, root, Work or producer-memory obligations.
No quota, acceptance threshold, 300-second deadline or historical result changes.

## Scope and reproducibility

The [stdlib analysis](../scripts/assess_wire_capacity.py) binds 18 implementation,
metadata and historical-result identities. Its [JSON record](hz_wire_capacity_20261014.json)
recomputes from shape metadata and current format definitions, without invoking
the former analyses' now-historical code identity checks. Old reports are
hashed and left unchanged; their old Work totals are not transplanted into the
new lazy path.

The same complete roster is retained: 28 unordered pairs, 252 properties, up to
504 endpoints and 1,008 proposed/zero candidate roles. Bounds do not assume
zero input centers, absent compensation, equal factor scales or sparse duals.
This is a capacity assessment, not evidence that all listed pairs are reachable.

## Physical representation: actual format, not the earlier design reserve

The current forest format uses a 24-byte prefix, an at-most-8-MiB header,
8-byte scales, 16-byte roots, 64-byte node descriptors and 16-byte inline or
reference entries. Source descriptors occupy their existing node records;
there is no second descriptor bank. With the registered shape bounds,

\[
B_F\le24+8\mathrm{MiB}+8(32048)+16(136232)+64(19496)
       +16(48048+277280)=17,277,720\ \mathrm{bytes}.
\]

This is conditional on the header passing its own bound. It does not admit
the root roster or assert actual checkpoint sparsity/encoded size.

Native candidates use fixed lower literals `0`; slot/query IDs are 64-character
hashes. A complete record's header is 228 bytes. For a tile of width \(w\),
the header is at most \(626+229w\) bytes, including its binding and logical hash.
Both zero and proposed roles remain checked. Generic long lower literals are
not covered by this native-specific header bound.

| Component bound (bytes) | Width 1 | Width 8 |
|---|---:|---:|
| Maximum tile header | 855 | 2,458 |
| Maximum physical tile | 22,467 | 175,270 |
| All physical candidate tiles | 11,323,368 | 11,055,408 |
| Source + forest + tiles, one copy | 88,510,912 | 88,242,952 |
| Same components, two retained copies | 177,021,824 | 176,485,904 |

Width 8 uses groups `8,8,2` for each pair, not three full eight-row tiles.
The source component includes 55,715,520 raw bytes plus its at-most-4-MiB
manifest. These sums exclude a complete production inventory of code, result
records, logs, partial attempts and terminal metadata. The full archive upper
therefore remains **null**. A component upper above 128 MiB does not prove an
actual archive will overflow; it does mean this upper cannot supply a two-copy
capacity guarantee. The former 110,611,136-byte design is not the implemented
size or a measurement.

## Four independent refusal or incompleteness boundaries

1. **Root capacity is a definite shape obstruction.** At least 126,632 roots
   are needed, whereas `intake()` still permits at most 65,536. It may reject
   before doing any of the projected downstream work below.
2. **Logical bytes remain bounded.** Wire intake replays the original canonical
   forest identity and 128-MiB logical limit; candidate replay retains the
   4-MiB tile and `16 MiB - 8192` cumulative candidate limit. Physical compression
   does not exempt these checks. The binary64 candidate tile upper for width 8
   is 7,541,057 bytes: this is an upper, not an observed or necessary overflow.
3. **A concrete logical-encoding counterexample exists.** Let every first-layer
   weight be the exactly representable binary64 value \(2^{-53}\), whose exact
   literal is `1/9007199254740992`. IDs 0–3071 contribute 11,178 decimal digits.
   A first-layer local array then needs
   \(1+11178+24\cdot3072=84907\) bytes. Across 2,176 rows, just these arrays take
   **184,757,632 bytes**, exceeding the logical forest limit by 50,539,904 bytes.
   No source was instantiated, and these are not observed trained weights.
   Other gates may reject first. This disproves a uniform logical-byte guarantee
   for arbitrary captured coefficients in this format, not any model's safety.
4. **The finite-control terminal is not a full-scale result protocol.** The
   driver embeds `outcome.results` in an 8-KiB terminal. Even the shortest legal
   252-row result array needs 13,357 bytes before identity and cost fields.
   A full runner needs a separately bounded result member and a small bound
   terminal reference. That layout has not been implemented or authorized here;
   neither has a larger terminal allowance.

## Changed-path Work and source-reading costs

`NodeTable` bounds each decoded row; it also makes formerly cheap tuple accesses
perform fresh decoding. Two components are now explicit:

| Complete-path component | Additional or mandatory count |
|---|---:|
| Logical identity replay + initialization, first-layer source positions only | at least 13,369,344 coefficient visits |
| Two input-node accesses per first-layer coefficient during semantic checking | 13,369,344 row + 13,369,344 coefficient visits |

The first row is \(2W_1\), for \(W_1=6,684,672\), including zero-valued source
positions. It already exceeds 8M. The second row is an additional lazy-decoder
cost relative to the prior resident tuple bank: **26,738,688 Work**, without
removing the original semantic checks. These are requirements of completing
the respective code paths, not measured runs or costs incurred after an earlier
refusal. Do not add them to an old total and call that a complete new ledger.

Each `SourceRows.row()` scans that tensor's chunk descriptors, rereads/hashes
intersecting complete chunks, and retains no cross-access cache. For one complete
first-layer traversal, a valid conservative byte bound is

\[
8W_1+2(1\mathrm{MiB})(2176)=4,616,880,128.
\]

This counts repeated userspace reads/hashes, **not** physical disk traffic,
peak memory or elapsed time. The two input-node accesses above are in-memory
row decoding, not two source-file reads. Gate, objective and exact-consumption
pullbacks can reread source ancestors; native preparation reads all nodes again.
Once the native operator is built, its 129 evaluations use resident tensors;
source read volume must not be multiplied by 129.

The complete new Work upper remains null. `produce()` still constructs a
default retaining session and a JSON proof, whereas the wire lifecycle controls
use fixed candidates and `retain_raw=False`. There is no frozen complete native
+ wire producer whose lifecycle can yet be assigned a total. A future integration
must include encoding, both role checks, terminal source reseal, copying,
publication, inventories and any independent replay in the same request budget.

## Resident memory: what did and did not change

The consumer no longer retains a full decoded Fraction coefficient bank; wire
bytes, root/view metadata, norms, objective/residual dictionaries and one tile
still live together. The migration encoder retains the old raw JSON and decoded
document, builds bytearrays, then joins the wire result. It is not bounded live
source construction. Live checkpoint capture is unchanged.

The native `Operator` also retains whole Python `rr/cc/vv` lists during global
COO construction, coalescing and transpose. Historical pinned-source named
payload components remain 161,585,280 bytes for one COO, 323,170,560 for COO plus
transpose, and 592,479,360 for the coalescing/sort branch. These are conditional
named-array bounds, **not** full native/RSS bounds or verified installed-binary
dispatch. Python allocation, source construction and native workspace lifetimes
remain open. The RSS/address-space stops are not completion guarantees.

## Bounded next decision

The approved controls-only stage is finished. Do not launch a registered source,
GPU, capacity retry or another small performance run to bypass these obstructions.
Do not silently increase a limit, discard checks or mark G1–G6 complete.

If development continues, the next scope must explicitly distinguish:

- logical-resource/representation policy, including roots and logical identities;
- direct bounded source construction, native preparation and complete Work/RSS;
- whole-request result/terminal and retained-copy lifecycle.

A repeated-access optimization alone cannot remove the root refusal or establish
the absent full resource contract. Any changed resource policy must be separately
approved and shared by the two internal reference arms. Code correctness controls
remain necessary, but do not replace MoE algorithm, GPU, same-source proof or
external comparison evidence. New proposed quotas, absolute ceilings and
wall-time forecasts remain null; existing limits are unchanged.

## Checks

Ten new static tests pass: record arithmetic, header/JSON length identities,
logical counterexample, changed Work components, terminal lower bound, immutable
historical anchors and no-admission semantics. They use `python -B -I -S`; no
production/numerical module is imported. Together with 19 project, review-handoff
and manuscript-preservation regressions, **29 tests pass**; all three links in
this report resolve. Two read-only agents independently
reviewed storage and lazy Work/lifetime paths. This is not human technical review.

```sh
/data1/Kane/miniconda3/envs/act-py312/bin/python -B -I -S -c 'import sys,unittest; sys.path.insert(0,"/data1/Kane/MOE/ACT"); unittest.main(module=None,argv=["unittest","scripts.test_wire_capacity","-v"])'
```

The prior 69 interface/regression tests and four archived fixture runs are not
recounted as new experiments. No scientific evidence or result directory was
deleted or modified.
